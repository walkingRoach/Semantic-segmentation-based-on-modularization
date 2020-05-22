import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, BatchNorm):
        super(_ASSPModule, self).__init__()

        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(self.atrous_conv(x))

        return self.relu(x)


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride, BatchNorm):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16]
        if output_stride == 8:
            dilations = [1, 6, 12, 18]
        elif output_stride == 16:
            dilations = [1, 12, 24, 36]

        self.aspp1 = _ASSPModule(in_channels, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASSPModule(in_channels, 256, 3, padding=0, dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASSPModule(in_channels, 256, 3, padding=0, dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASSPModule(in_channels, 256, 3, padding=0, dilation=dilations[3], BatchNorm=BatchNorm)

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x
