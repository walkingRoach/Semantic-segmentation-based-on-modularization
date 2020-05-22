from core.base import BaseModel
import torch.nn as nn
from models.backbone import VGGNet
import torch.nn.functional as F
from itertools import chain


class FCN32s(BaseModel):
    def __init__(self, num_classes, backbone='vgg16', **_):
        super(FCN32s, self).__init__()
        self.n_class = num_classes
        self.pretrained_net = VGGNet(pretrained=True, model=backbone)

        self.decode1 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
            ])
        self.bn1 = nn.BatchNorm2d(512)

        self.decode2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn2 = nn.BatchNorm2d(256)

        self.decode3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn3 = nn.BatchNorm2d(128)

        self.decode4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn4 = nn.BatchNorm2d(64)

        self.decode5 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # forward
        output = self.pretrained_net(x)
        # print(x.size())
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        # backward
        score = self.bn1(self.decode1(x5))   # size=(N, 512, H/16, w/16)
        score = self.bn2(self.decode2(score))   # size=(N, 256, H/8, w/8)
        score = self.bn3(self.decode3(score))   # size=(N, 128, H/4, w/4)
        score = self.bn4(self.decode4(score))   # size=(N, 64, H/2, w/2)
        score = self.bn5(self.decode5(score))   # size=(N, 32, H/1, w/1)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def get_backbone_params(self):
        return self.pretrained_net.parameters()

    def get_decoder_params(self):
        return chain(self.decode1.parameters(), self.bn1.parameters(), self.decode2.parameters(), self.bn2.parameters(),
                     self.decode3.parameters(), self.bn3.parameters(), self.decode4.parameters(), self.bn4.parameters(),
                     self.decode5.parameters(), self.bn5.parameters(), self.classifier.parameters())


class FCN16s(BaseModel):
    def __init__(self, num_classes, backbone='vgg16', **_):
        super(FCN16s, self).__init__()
        self.n_class = num_classes
        self.pretrained_net = VGGNet(pretrained=True, model=backbone)

        self.decode1 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
            ])
        self.bn1 = nn.BatchNorm2d(512)

        self.decode2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn2 = nn.BatchNorm2d(256)

        self.decode3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn3 = nn.BatchNorm2d(128)

        self.decode4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn4 = nn.BatchNorm2d(64)

        self.decode5 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # forward
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        # backward
        score = self.decode1(x5)   # size=(N, 512, H/16, w/16)
        print(score.size(), x4.size())
        score = padding_tensor(x4, score)
        print(score.size(), x4.size())
        score = self.bn1(score + x4)
        score = self.bn2(self.decode2(score))   # size=(N, 256, H/8, w/8)
        score = padding_tensor(x3, score)
        score = self.bn3(self.decode3(score))   # size=(N, 128, H/4, w/4)
        score = padding_tensor(x2, score)
        score = self.bn4(self.decode4(score))   # size=(N, 64, H/2, w/2)
        print(score.size(), x1.size())
        score = padding_tensor(x1, score)
        print(score.size(), x1.size())
        score = self.bn5(self.decode5(score))   # size=(N, 32, H/1, w/1)
        score = padding_tensor(x, score)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def get_backbone_params(self):
        return self.pretrained_net.parameters()

    def get_decoder_params(self):
        return chain(self.decode1.parameters(), self.bn1.parameters(), self.decode2.parameters(), self.bn2.parameters(),
                     self.decode3.parameters(), self.bn3.parameters(), self.decode4.parameters(), self.bn4.parameters(),
                     self.decode5.parameters(), self.bn5.parameters(), self.classifier.parameters())


class FCN8s(BaseModel):
    def __init__(self, num_classes, backbone='vgg16', **_):
        super(FCN8s, self).__init__()
        self.n_class = num_classes
        self.pretrained_net = VGGNet(pretrained=True, model=backbone)

        self.decode1 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        ])
        self.bn1 = nn.BatchNorm2d(512)

        self.decode2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn2 = nn.BatchNorm2d(256)

        self.decode3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn3 = nn.BatchNorm2d(128)

        self.decode4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn4 = nn.BatchNorm2d(64)

        self.decode5 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # forward
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        # backward
        score = self.decode1(x5)  # size=(N, 512, H/16, w/16)
        score = padding_tensor(x4, score)
        score = self.bn1(score + x4)
        score = self.decode2(score)  # size=(N, 256, H/8, w/8)
        score = padding_tensor(x3, score)
        score = self.bn2(score + x3)
        score = self.bn3(self.decode3(score))  # size=(N, 128, H/4, w/4)
        score = padding_tensor(x2, score)
        score = self.bn4(self.decode4(score))  # size=(N, 64, H/2, w/2)
        score = padding_tensor(x1, score)
        score = self.bn5(self.decode5(score))  # size=(N, 32, H/1, w/1)
        score = padding_tensor(x, score)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def get_backbone_params(self):
        return self.pretrained_net.parameters()

    def get_decoder_params(self):
        return chain(self.decode1.parameters(), self.bn1.parameters(), self.decode2.parameters(), self.bn2.parameters(),
                     self.decode3.parameters(), self.bn3.parameters(), self.decode4.parameters(), self.bn4.parameters(),
                     self.decode5.parameters(), self.bn5.parameters(), self.classifier.parameters())


class FCNs(BaseModel):
    def __init__(self, num_classes, backbone='vgg16', **_):
        super(FCNs, self).__init__()
        self.n_class = num_classes
        self.pretrained_net = VGGNet(pretrained=True, model=backbone)

        self.decode1 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn1 = nn.BatchNorm2d(512)

        self.decode2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn2 = nn.BatchNorm2d(256)

        self.decode3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn3 = nn.BatchNorm2d(128)

        self.decode4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn4 = nn.BatchNorm2d(64)

        self.decode5 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        ])
        self.bn5 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # forward
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        # backward
        score = self.bn1(self.decode1(x5))  # size=(N, 512, H/16, w/16)
        score = padding_tensor(x4, score)
        score = score + x4
        score = self.bn2(self.decode2(score))  # size=(N, 256, H/8, w/8)
        score = padding_tensor(x3, score)
        score = score + x3
        score = self.bn3(self.decode3(score))  # size=(N, 128, H/4, w/4)
        score = padding_tensor(x2, score)
        score = score + x2
        score = self.bn4(self.decode4(score))  # size=(N, 64, H/2, w/2)
        score = padding_tensor(x1, score)
        score = score + x1
        score = self.bn5(self.decode5(score))  # size=(N, 32, H/1, w/1)
        score = padding_tensor(x, score)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def get_backbone_params(self):
        return self.pretrained_net.parameters()

    def get_decoder_params(self):
        return chain(self.decode1.parameters(), self.bn1.parameters(), self.decode2.parameters(), self.bn2.parameters(),
                     self.decode3.parameters(), self.bn3.parameters(), self.decode4.parameters(), self.bn4.parameters(),
                     self.decode5.parameters(), self.bn5.parameters(), self.classifier.parameters())


def padding_tensor(x_target, x):
    diff_Y = x_target.size()[2] - x.size()[2]
    diff_X = x_target.size()[3] - x.size()[3]
    x = F.pad(x, [diff_X // 2, diff_X - diff_X // 2, diff_Y // 2, diff_Y - diff_Y // 2],
              mode='reflect')
    return x
