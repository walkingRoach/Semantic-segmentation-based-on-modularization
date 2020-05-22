from core.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neck import ASSP
from models.backbone import resnet, mobilenet, xception
from torchvision import models
from itertools import chain


class Decoder(nn.Module):
    def __init__(self, backbone, num_classes, BatchNorm):
        super(Decoder, self).__init__()

        if backbone is 'resnet50':
            low_level_features = 256
        elif backbone is 'resnet32':
            low_level_features = 64
        elif backbone is 'xception':
            low_level_features = 128
        elif backbone is 'mobilenet':
            low_level_features = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_features, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )

        self._init_weight()

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))

        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(BaseModel):
    def __init__(self, num_classes, in_clannels=3, backbone='xception', pretrained=True,
                 output_stride=16, freeze_bn=False, **kwargs):
        super(DeepLab, self).__init__()

        assert backbone in ['resnet32', 'resnet50', 'xception', 'mobilenet']
        if 'resnet' in backbone:
            self.backbone = getattr(resnet, backbone)(pretrained=pretrained, dilated=False, multi_gird=False,
                                                      deep_base=False, back_layer_features=True)
            out_channels = 2048
        elif backbone == 'xception':
            self.backbone = getattr(xception, backbone)(pretrained=pretrained, output_stride=output_stride)
            out_channels = 1280
        elif backbone == 'mobilenet':
            self.backbone = getattr(mobilenet, backbone)(pretrained=pretrained, output_stride=output_stride)
            out_channels = 1280
        else:
            raise NotImplementedError

        self.ASSP = ASSP(in_channels=out_channels, output_stride=output_stride, BatchNorm=nn.BatchNorm2d)
        self.decoder = Decoder(backbone, num_classes, nn.BatchNorm2d)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x, output = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, output["x1"])

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

