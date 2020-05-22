from core.base import BaseModel
import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision import models
import torch.nn.functional as F


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        batch_norm = False
        if str(model).endswith('bn'):
            batch_norm = True
        super().__init__(make_layers(cfg[model], batch_norm=batch_norm))
        self.ranges = ranges[model]

        self.remove_fc = remove_fc

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # print(x.size())
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        if not self.remove_fc:
            output = self.classifier(output)

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37)),
    'vgg11_bn': ((0, 4), (4, 8), (8, 15), (15, 22), (22, 29)),
    'vgg13_bn': ((0, 7), (7, 14), (14, 21), (21, 28), (28, 35)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19_bn': ((0, 7), (7, 14), (14, 27), (27, 40), (40, 53))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg11_bn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg11', remove_fc=remove_fc)
    return model


def vgg16(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg16', remove_fc=remove_fc)
    return model


def vgg13(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg13', remove_fc=remove_fc)
    return model


def vgg19(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg19', remove_fc=remove_fc)
    return model


def vgg11_bn(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg11_bn', remove_fc=remove_fc)
    return model


def vgg13_bn(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg13_bn', remove_fc=remove_fc)
    return model


def vgg16_bn(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg16_bn', remove_fc=remove_fc)
    return model


def vgg19_bn(pretrained=True, remove_fc=False):
    model = VGGNet(pretrained=pretrained, model='vgg19_bn', remove_fc=remove_fc)
    return model
