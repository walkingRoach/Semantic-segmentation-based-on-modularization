# Origian code and chechpoints by Hang Zhang
# https://github.com/zhanghang1989/PyTorch-Encoding

import math
import torch
import os
import sys
import zipfile
import shutil
import torch.utils.model_zoo as model_zoo
from torchvision.models.utils import load_state_dict_from_url
import torch.nn as nn
from ..modules import BasicBlock, Bottleneck
from collections import OrderedDict
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50_deep_base': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b509.zip',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.zip',
    'resnet152': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, remove_fc=True, dilated=False, multi_grid=False,
                 deep_base=True, groups=1, width_per_group=64, norm_layer=nn.BatchNorm2d, back_layer_features=False):
        self.inplanes = 128 if deep_base else 64
        self.groups = groups
        self.base_width = width_per_group
        self.bace_layer_features = back_layer_features
        self.remove_fc = remove_fc

        if groups > 1:
            multi_grid = False

        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)  # (-1, 64, w/2, h/2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            if multi_grid:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer, multi_grid=True, name='layer3')
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True, name='layer4')
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,   # 原版512, stride=2
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)   # 512 * block.expanseion

        if self.remove_fc:
            del self.avgpool
            del self.fc

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False, name=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if name == 'layer3':
            multi_dilations = [1, 2, 5, 9]
        elif name == 'layer4':
            multi_dilations = [5, 9, 17]
        else:
            multi_dilations = [4, 8, 16]

        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, groups=self.groups, base_width=self.base_width,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, groups=self.groups, base_width=self.base_width,
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, groups=self.groups, base_width=self.base_width,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                if name == 'layer3':
                    layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                        dilation=multi_dilations[i%4], norm_layer=norm_layer))
                elif name == 'layer4':
                    layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                        dilation=multi_dilations[i], norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                    dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output = {}
        x = self.layer1(x)
        output["x1"] = x
        # low_level_features = x
        # print('layer1 : {}'.format(x.size()))
        x = self.layer2(x)
        output["x2"] = x
        # print('layer2 : {}'.format(x.size()))
        x = self.layer3(x)
        output["x3"] = x
        # print(f'layer3 : {x.size()}')
        x = self.layer4(x)
        output["x4"] = x
        # print(f'layer4 : {x.size()}')
        # x = self.avgpool(x)
        # # print(x.size())
        # x = torch.flatten(x, 1)
        # print(x.size())
        if not self.remove_fc:
            x = self.avgpool(x)
            # print(x.size())
            x = torch.flatten(x, 1)
            x = self.fc(x)
        if self.bace_layer_features:
            return x, output
        else:
            return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False,   **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, remove_fc=False, root=os.path.expanduser("~/.cache/torch/checkpoints/"), deep_base=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=deep_base, remove_fc=remove_fc, **kwargs)
    if pretrained and deep_base is True:
        model.load_state_dict(load_url(model_urls['resnet50'], model_dir=root))
    elif pretrained and deep_base is False:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        target_state = OrderedDict()
        if remove_fc == True:
            for k, v in state_dict.items():
                if not 'fc' in k:
                    target_state[k] = v
        else:
            target_state = state_dict
        model.load_state_dict(target_state)
    return model


def resnet101(pretrained=False, root=os.path.expanduser("~/.cache/torch/checkpoints/"), **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101'], model_dir=root))
    return model


def resnet152(pretrained=False, root=os.path.expanduser("~/.cache/torch/checkpoints/"), **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet152'], model_dir=root))
    return model


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnext101_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress=progress)
        model.load_state_dict(state_dict)
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1].split('.')[0]
    cached_file = os.path.join(model_dir, filename+'.pth')
    if not os.path.exists(cached_file):
        cached_file = os.path.join(model_dir, filename+'.zip')
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
        zip_ref = zipfile.ZipFile(cached_file, 'r')
        zip_ref.extractall(model_dir)
        zip_ref.close()
        os.remove(cached_file)
        cached_file = os.path.join(model_dir, filename+'.pth')
    checkpoint = torch.load(cached_file, map_location=map_location)

    return checkpoint