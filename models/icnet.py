'''
code: from https://github.com/liminn/ICNet-pytorch/blob/master/models/icnet.py
是一种图像的联级网络，同时保存了快速的特点
'''
from core.base import BaseModel
from .backbone import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(_ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, x):
        feat = x
        h, w = x.shape[2:]

        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(x, output_size=bin_size)
            x = F.interpolate(x,  size=(h, w), mode='bilinear', align_corners=True)
            feat = feat + x  # 这里同pspnet有不同的地方,
        return feat


class CascadeFeatureFusion(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.con_low_cls = nn.Conv2d(out_channels, num_classes, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.con_low_cls(x_low)

        return x, x_low_cls


class _IChead(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d):
        super(_IChead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, num_classes, norm_layer)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, num_classes, norm_layer)

        self.conv_cls = nn.Conv2d(128, num_classes, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = {}
        x_cff24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs["x24"] = x_24_cls
        x_cff12, x_12_cls = self.cff_12(x_cff24, x_sub1)
        outputs["x12"] = x_12_cls

        up_x2 = F.interpolate(x_cff12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)

        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs["x"] = up_x8

        return outputs


class ICNet(BaseModel):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, dilated=True,
                 multi_grid=False, **kwargs):
        super(ICNet, self).__init__()

        self.backbone = backbone
        assert self.backbone in ['resnet50', 'resnext50', 'resnet101']

        if 'resnet' in backbone:
            self.encoder = getattr(resnet, backbone)(pretrained, dilated=dilated, multi_grid=multi_grid,
                                                     back_layer_features=True)

        self.sub1 = nn.Sequential(
            _ConvBNRelu(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            _ConvBNRelu(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            _ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        )

        self.pyramid = PyramidPoolingModule()

        self.head = _IChead(num_classes)

    def forward(self, x):
        x_sub1 = self.sub1(x)

        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, sub2_output = self.encoder(x_sub2)
        x_sub2 = sub2_output["x2"]

        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, sub4_output = self.encoder(x_sub4)
        x_sub4 = sub4_output["x4"]

        x_sub4 = self.pyramid(x_sub4)

        output = self.head(x_sub1, x_sub2, x_sub4)

        return output

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
