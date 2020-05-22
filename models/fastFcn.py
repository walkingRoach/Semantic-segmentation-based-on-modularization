from core.base import BaseModel
import torch
import torch.nn as nn
from models.backbone import vgg
from models.backbone import resnet
from models.neck import JPU
import torch.nn.functional as F
from itertools import chain


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = BatchNorm(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        inter_channels = in_channels // 4
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.output(x)


class FCNFast(BaseModel):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, freeze_bn=False, dilated=True,
                 multi_grid=False, **kwargs):
        super(FCNFast, self).__init__()

        self.backbone = backbone

        if 'resnet' in backbone:
            self.encoder = getattr(resnet, backbone)(pretrained, dilated=dilated, multi_grid=multi_grid)
        elif 'vgg' in backbone:
            self.encoder = getattr(vgg, backbone)(pretrained)

        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=nn.BatchNorm2d)
        self.decoder = Decoder(2048, num_classes, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        if 'resnet' in self.backbone:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            c1 = self.backbone.layer1(x)
            c2 = self.backbone.layer2(c1)
            c3 = self.backbone.layer3(c2)
            c4 = self.backbone.layer4(c3)

            x = self.jpu(c1, c2, c3, c4)
            x = self.decoder(x)

            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

            return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return chain(self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.jpu.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
