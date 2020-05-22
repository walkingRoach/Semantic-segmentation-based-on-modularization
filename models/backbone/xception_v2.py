'''
来自：https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from ..modules import SeparableConv2d


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, BatchNorm=nn.BatchNorm2d,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_channels

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(out_channels))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            print('using grow_first')
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(out_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(out_channels))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(out_channels))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, output_stride, blocks, BatchNorm,
                 pretrained=True):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        self.block_num = blocks[1]

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.entry_flow = self._make_layer(blocks[0], suffix=1, in_channels=64, out_channels=[128, 256, 728], reps=[2, 2, 2],
                                           dilations=[1, 1, 1], strides=[2, 2, entry_block3_stride],
                                           norm_layer=BatchNorm, start_with_relu=False, grow_first=True, is_last=True)

        # Middle flow
        self.middle_flow = self._make_layer(blocks[1], suffix=blocks[0]+1, in_channels=728, out_channels=[728, 728, 728], reps=[3, 3, 3],
                                            dilations=[1, 1, 1], strides=[1, 1, 1],
                                            norm_layer=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        self.exit_block = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1024, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1024)  # 1536

        self.conv4 = SeparableConv2d(1024, 1024, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1024)  # 1536

        self.conv5 = SeparableConv2d(1024, 1024, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(1024)  # 2048

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.entry_flow(x)

        # middle flow
        x = self.middle_flow(x)

        # Exit flow
        x = self.exit_block(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x

    def _make_layer(self, blocks, suffix, in_channels, out_channels, reps, dilations, strides, norm_layer=None,
                    start_with_relu=True, grow_first=True, is_last=False):
        layers = torch.nn.Sequential()

        # start layer
        layers.add_module("block"+str(suffix), Block(in_channels, out_channels[0], reps=reps[0], stride=strides[0], dilation=dilations[0],
                            BatchNorm=norm_layer, start_with_relu=start_with_relu, grow_first=grow_first))
        in_channels = out_channels[0]
        # middle layer
        for i in range(1, blocks-1):
            layers.add_module("block"+str(suffix+i), Block(in_channels, out_channels[1], reps=reps[1], stride=strides[1], dilation=dilations[1],
                                BatchNorm=norm_layer, start_with_relu=True, grow_first=True))
            in_channels = out_channels[1]

        # exit layer
        layers.add_module("block"+str(suffix+blocks-1), Block(in_channels, out_channels[2], reps=reps[2], stride=strides[2], dilation=dilations[2],
                            BatchNorm=norm_layer, start_with_relu=True, grow_first=True, is_last=is_last))

        return layers

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        # 需要修改, 还需要争着一下
        for k, v in pretrain_dict.items():
            print(k)
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block1.') or k.startswith('block2.') or k.startswith('block3.'):
                    model_dict['entry_flow.'+k] = v
                elif k.startswith('block') and self.block_num < 9 and not k.startswith('block12'):
                    model_dict['middle_flow.'+k] = v
                elif k.startswith('block11') and self.block_num > 8:
                    for i in range(self.block_num - 8):
                        model_dict['middle_flow.'+'block'+str(i+12)] = v
                elif k.startswith('block12'):
                    model_dict['exit_flow.'+k.split('block12.')[-1]] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def xception(blocks=(3, 8), pretrained=True, output_stride=16, BatchNorm=nn.BatchNorm2d):
    model = AlignedXception(pretrained=pretrained, blocks=blocks, output_stride=output_stride, BatchNorm=BatchNorm)

    return model
