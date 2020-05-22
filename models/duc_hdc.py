from core.base import BaseModel
from torch import nn
from models.backbone import resnet
from models.decode import DenseUpsamplingConvModule
from itertools import chain


class ResNetDUC(BaseModel):
    # the size of image should be multiple of 8
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, freeze_bn=False, dilated=True, multi_grid=False, **kwargs):
        super(ResNetDUC, self).__init__()
        encoder = getattr(resnet, backbone)(pretrained, dilated=dilated, multi_grid=multi_grid)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.duc = DenseUpsamplingConvModule(num_classes, 8, 2048)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return chain(self.layer0.parameters(), self.layer1.parameters(), self.layer2.parameters,
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return self.duc.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class ResNetDUCHDC(BaseModel):
    # the size of image should be multiple of 8
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, dilated=True, multi_grid=True, **kwargs):
        super(ResNetDUCHDC, self).__init__()
        encoder = getattr(resnet, backbone)(pretrained, dilated=dilated, multi_grid=multi_grid)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.duc = DenseUpsamplingConvModule(num_classes, 8, 2048)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.duc(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return chain(self.layer0.parameters(), self.layer1.parameters(), self.layer2.parameters,
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return self.duc.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
