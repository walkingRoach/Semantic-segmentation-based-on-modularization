import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import BaseModel


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CBR, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InputProjectionA(nn.Module):
    def __init__(self, samplingTimes):
        super(InputProjectionA, self).__init__()
        self.pool = nn.ModuleList()

        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


class BR(nn.Module):
    def __init__(self, out_chanrels):
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(out_chanrels)
        self.act = nn.PReLU(out_chanrels)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class CB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CB, self).__init__()

        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        '''
        :param in_channel: number of input channels
        :param out_channel: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(in_channel, out_channel, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        x = self.conv(x)
        return x


class CDilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CDilated, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
                              dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSamplerB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplerB, self).__init__()
        n = int(out_channels / 5)
        n1 = out_channels - 4 * n
        self.c1 = C(in_channels, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.c1(x)

        d1 = self.d1(x)
        d2 = self.d2(x)
        d4 = self.d4(x)
        d8 = self.d8(x)
        d16 = self.d16(x)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        output = self.bn(combine)
        output - self.act(output)
        return output


class DilatedParallelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, in_channels, out_channels, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(out_channels / 5)
        n1 = out_channels - 4 * n
        self.c1 = C(in_channels, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BR(out_channels)
        self.add = add

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(x)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = x + combine
        output = self.bn(combine)
        return output


class ESPNetEncoder(nn.Module):
    """
    ESPNet-C encoder
    """

    def __init__(self, out_channels=256, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParallelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParallelResidualBlockB(128, 128))
        self.b3 = BR(256)

        self.classifier = C(256, out_channels, 1, 1)  # 这一层可以省略

        self._initialize_weights()

    def forward(self, x):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(x)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return classifier

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ESPNet(pretrained=False, **kwargs):
    model = ESPNetEncoder(**kwargs)
    return model


