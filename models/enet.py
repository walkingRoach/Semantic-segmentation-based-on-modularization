from core.base import BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class InitalBlock(nn.Module):
    def __init__(self, in_channels):
        super(InitalBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16)

    def forward(self, x):
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, inter_rate=4, downsample=False, upsample=False,
                 asymmetric=False, dilate=1, regularize=True, p_drop=0.01):
        super(BottleNeck, self).__init__()
        self.upsample = upsample
        self.downsample = downsample

        self.pad = out_channels - in_channels

        inter_channels = out_channels // inter_rate
        # Main
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        elif upsample:
            self.spatial_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # bottleneck
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU(inter_channels)

        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(1, 5), padding=(0, 2)),
                nn.BatchNorm2d(inter_channels),
                nn.PReLU(),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=(5, 1), padding=(2, 0))
            )
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, padding=1,
                                            output_padding=1, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1,
                               dilation=dilate, padding=dilate, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU()

        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU()

    def forward(self, x, indices=None, output_size=None):
        # Main
        identity = x
        if self.upsample:
            assert (indices is not None) and (output_size is not None)
            identity = self.bn(self.spatial_conv(identity))
            if identity.size() != indices.size():
                pad = [indices.size(3) - identity.size(3), 0, indices.size(2) - identity.size(2), 0]
                identity = F.pad(identity, pad, "constant", 0)
            identity = self.unpool(identity, indices=indices)
        elif self.downsample:
            identity, idx = self.pool(identity)

        if self.pad > 0:
            extras = torch.zeros((identity.size(0), self.pad, identity.size(2), identity.size(3)))
            if torch.cuda.is_available(): extras = extras.cuda(0)
            identity = torch.cat((identity, extras), dim = 1)

        # Bottleneck
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)

        # When the input dim is odd, we might have a mismatch of one pixel
        if identity.size() != x.size():
            pad = [identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2), 0]
            x = F.pad(x, pad, "constant", 0)

        x += identity
        x = self.prelu_out(x)

        if self.downsample:
            return x, idx
        return x


class ENet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freze_bn=False, **kwargs):
        super(ENet, self).__init__()
        self.initial = InitalBlock(in_channels)

        # stage one
        self.bottlenect10 = BottleNeck(16, 64, downsample=True, p_drop=0.01)
        self.bottlenect11 = BottleNeck(64, 64)
        self.bottlenect12 = BottleNeck(64, 64)
        self.bottlenect13 = BottleNeck(64, 64)
        self.bottlenect14 = BottleNeck(64, 64)

        # stage two
        self.bottlenect20 = BottleNeck(64, 128, downsample=True, p_drop=0.1)
        self.bottlenect21 = BottleNeck(128, 128, p_drop=0.1)
        self.bottlenect22 = BottleNeck(128, 128, dilate=2, p_drop=0.1)
        self.bottlenect23 = BottleNeck(128, 128, asymmetric=True, p_drop=0.1)
        self.bottlenect24 = BottleNeck(128, 128, dilate=4, p_drop=0.1)
        self.bottlenect25 = BottleNeck(128, 128, p_drop=0.1)
        self.bottlenect26 = BottleNeck(128, 128, dilate=8, p_drop=0.1)
        self.bottlenect27 = BottleNeck(128, 128, asymmetric=True, p_drop=0.1)
        self.bottlenect28 = BottleNeck(128, 128, dilate=16, p_drop=0.1)

        # stage three
        self.bottlenect31 = BottleNeck(128, 128, p_drop=0.1)
        self.bottlenect32 = BottleNeck(128, 128, dilate=2, p_drop=0.1)
        self.bottlenect33 = BottleNeck(128, 128, asymmetric=True, p_drop=0.1)
        self.bottlenect34 = BottleNeck(128, 128, dilate=4, p_drop=0.1)
        self.bottlenect35 = BottleNeck(128, 128, p_drop=0.1)
        self.bottlenect36 = BottleNeck(128, 128, dilate=8, p_drop=0.1)
        self.bottlenect37 = BottleNeck(128, 128, asymmetric=True, p_drop=0.1)
        self.bottlenect38 = BottleNeck(128, 128, dilate=16, p_drop=0.1)

        # stage four
        self.bottlenect40 = BottleNeck(128, 64, upsample=True, p_drop=0.1)
        self.bottlenect41 = BottleNeck(64, 64, p_drop=0.1)
        self.bottlenect42 = BottleNeck(64, 64, p_drop=0.1)

        self.bottlenect50 = BottleNeck(64, 16, upsample=True, p_drop=0.1)
        self.bottlenect51 = BottleNeck(16, 16, p_drop=0.1)

        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, padding=1, output_padding=1,
                                           stride=2, bias=False)

    def forward(self, x):
        x = self.initial(x)

        x1_size = x.size()  # (-1. 16, w/2, h/2)
        x, indices1 = self.bottlenect10(x)
        x = self.bottlenect11(x)
        x = self.bottlenect12(x)
        x = self.bottlenect13(x)
        x = self.bottlenect14(x)

        x2_size = x.size()  # (-1, 64, w/4, h/4)
        x, indices2 = self.bottlenect20(x)
        x = self.bottlenect21(x)
        x = self.bottlenect22(x)
        x = self.bottlenect23(x)
        x = self.bottlenect24(x)
        x = self.bottlenect25(x)
        x = self.bottlenect26(x)
        x = self.bottlenect27(x)
        x = self.bottlenect28(x)

        x = self.bottlenect31(x)
        x = self.bottlenect32(x)
        x = self.bottlenect33(x)
        x = self.bottlenect34(x)
        x = self.bottlenect35(x)
        x = self.bottlenect36(x)
        x = self.bottlenect37(x)
        x = self.bottlenect38(x)

        x = self.bottlenect40(x, indices=indices2, output_size=x2_size)
        x = self.bottlenect41(x)
        x = self.bottlenect42(x)

        x = self.bottlenect50(x, indices=indices1, output_size=x1_size)
        x = self.bottlenect51(x)

        x = self.fullconv(x)
        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

