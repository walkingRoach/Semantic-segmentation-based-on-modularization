from core.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet
from models.decode import UpConv2d, UpConv2dWithCat
from itertools import chain


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()
        self.down1 = Encoder(in_channels, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.up1 = UpConv2d(1024, 512, up_sample=True)
        self.up2 = UpConv2d(512, 256, up_sample=True)
        self.up3 = UpConv2d(256, 128, up_sample=True)
        self.up4 = UpConv2d(128, 64, up_sample=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1, x = self.down1(x)  # (-1, 64, w/2, h/2)
        x2, x = self.down2(x)  # (-1, 128, w/4, h/4)
        x3, x = self.down3(x)  # (-1, 256, w/8, h/8)
        x4, x = self.down4(x)  # (-1, 512, w/16, h/16)
        x = self.middle_conv(x)  # (-1, 1024, w/16, h/16)
        x = self.up1(x, x4)   # 逻辑是先压缩再拼接再压缩
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.bridge(x)
        return x


basicBlock_model = ['resnet18', 'resnet34']


class UNetResnet(BaseModel):
    def __init__(self, backbone, num_classes, pretrained=True, freeze_bn=False, multi_grid=False, **kwargs):
        super(UNetResnet, self).__init__()
        model = getattr(resnet, backbone)(pretrained, deep_base=False, norm_layer=nn.BatchNorm2d, dilated=False)
        base_channels = 512 if backbone in basicBlock_model else 2048

        self.layer0_block = nn.Sequential(model.conv1, model.bn1, model.relu)  # (-1, 128, w/2, h/2)
        self.layer0_pool = model.maxpool
        self.layer1 = model.layer1  # (-1, 256, w/4, h/4)
        self.layer2 = model.layer2  # (-1, 512, w/8, h/8)
        self.layer3 = model.layer3  # (-1, 1024, w/16, h/16)
        self.layer4 = model.layer4  # (-1, 2048, w/32, h/32)
        # self.input_block = nn.Sequential(*list(model.children()))[:3]
        # self.input_pool = list(model.children())[3]

        self.bridge = Bridge(base_channels, base_channels)

        self.up1 = UpConv2dWithCat(base_channels, base_channels/2, block_num=2)
        self.up2 = UpConv2dWithCat(base_channels/2, base_channels/4, block_num=2)
        self.up3 = UpConv2dWithCat(base_channels/4, base_channels/8, block_num=2)
        if base_channels == 2048:
            self.up4 = UpConv2dWithCat(128+64, 128, 256, 128, block_num=2)
            self.up5 = UpConv2dWithCat(64+3, 64, 128, 64, block_num=2)
        else:
            self.up4 = UpConv2dWithCat(64+64, 64, 64, 64, block_num=2)
            self.up5 = UpConv2dWithCat(64+3, 64, 64, 64, block_num=2)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = x  # (-1. 3, w, h)
        x = self.layer0_block(x)
        x2 = x  # (-1, 64, w/2, h/2)
        x = self.layer0_pool(x)
        x = self.layer1(x)  # (-1, 64, w/4, h/4)
        x3 = x
        x = self.layer2(x)  # (-1, 128, w/8, h/8)
        x4 = x
        x = self.layer3(x)  # (-1, 256, w/16, h/16)
        x5 = x
        x = self.layer4(x)  # (-1. 512, w/32, h/32)

        x = self.bridge(x)

        x = self.up1(x, x5)  # (-1. 256, w/16, h/16)
        x = self.up2(x, x4)  # (-1. 128, w/8, h/8)
        x = self.up3(x, x3)  # (-1. 64, w/4, h/4)
        # 这里需要特殊处理
        x = self.up4(x, x2)  # x:(-1, 64, w/4, h/4) x2:(-1. 64, w/2, h/2), (-1. 64, w/2, h/2)
        x = self.up5(x, x1)  # x:(-1, 64, w/2, h/2) x1:(-1, 3, w, h), (-1, 64, w, h)

        x = self.final_conv(x)

        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return chain(self.layer0_block.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters(), self.bridge.parameters())

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
