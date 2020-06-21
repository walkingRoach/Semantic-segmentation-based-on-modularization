import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import BaseModel
from .neck import ConvLSTM
from utils.utils_model import initialize_weights, set_trainable
from torchvision import models


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


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_target, x):
        # print(x.size())
        x = self.up(x)
        # Padding in case the incomping volumes are of different sizes
        diffY = x_target.size()[2] - x.size()[2]
        diffX = x_target.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        # Concatenate
        x = torch.cat([x_target, x], dim=1)
        x = self.up_conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super(UpConv, self).__init__()
        self.up_sample = upsample
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_target, x):
        # print(x.size())
        # Concatenate
        x = self.up_conv(x)

        if self.up_sample:
            x = self.up(x)
            diffY = x_target.size()[2] - x.size()[2]
            diffX = x_target.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        return x


class UNet_ConvLSTM(BaseModel):
    def __init__(self, num_classes, freeze_bn=False, freeze_backbone=False):
        super(UNet_ConvLSTM, self).__init__()

        self.down1 = Encoder(3, 64)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512)

        self.convlstm = ConvLSTM(input_size=(20, 12),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

        self.up1 = Decoder(512, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        initialize_weights(self)
        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.down1, self.down2, self.down3, self.down4], False)


    def forward(self, x):
        inputs = torch.unbind(x, dim=1)
        data = []
        for item in inputs:
            x1, x = self.down1(item)
            x2, x = self.down2(x)
            x3, x = self.down3(x)
            x4, x = self.down4(x)
            data.append(x.unsqueeze(0))
        data = torch.cat(data, dim=0)
        # print(data.size())
        lstm, _ = self.convlstm(data)

        # print(f'lstm len {len(lstm)}')
        # print(f'lstm size {lstm[0].size()}')
        test = lstm[0][-1, :, :, :, :]
        # print(f'test size {test.size()}')

        x = self.up1(x4, test)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        # print(f'output size {x.size()}')
        return x, test

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

# 这个网络只有244层, 并不是很难训练
class SegNet_ConvLSTM(BaseModel):
    def __init__(self, num_classes, pretrained=False, freeze_bn=False, freeze_backbone=False):
        super(SegNet_ConvLSTM,self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=pretrained).features
        self.relu = nn.ReLU(inplace=True)
        self.index_MaxPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.index_UnPool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # net struct
        self.conv1_block = nn.Sequential(vgg16_bn[0],  # conv2d(3,64,(3,3))
                                         vgg16_bn[1],  # bn(64,eps=1e-05,momentum=0.1,affine=True)
                                         vgg16_bn[2],  # relu(in_place)
                                         vgg16_bn[3],  # conv2d(3,64,(3,3))
                                         vgg16_bn[4],  # bn(64,eps=1e-05,momentum=0.1,affine=True)
                                         vgg16_bn[5]   # relu(in_place)
                                         )
        self.conv2_block = nn.Sequential(vgg16_bn[7],
                                         vgg16_bn[8],
                                         vgg16_bn[9],
                                         vgg16_bn[10],
                                         vgg16_bn[11],
                                         vgg16_bn[12]
                                         )
        self.conv3_block = nn.Sequential(vgg16_bn[14],
                                         vgg16_bn[15],
                                         vgg16_bn[16],
                                         vgg16_bn[17],
                                         vgg16_bn[18],
                                         vgg16_bn[19],
                                         vgg16_bn[20],
                                         vgg16_bn[21],
                                         vgg16_bn[22]
                                         )
        self.conv4_block = nn.Sequential(vgg16_bn[24],
                                         vgg16_bn[25],
                                         vgg16_bn[26],
                                         vgg16_bn[27],
                                         vgg16_bn[28],
                                         vgg16_bn[29],
                                         vgg16_bn[30],
                                         vgg16_bn[31],
                                         vgg16_bn[32]
                                         )
        self.conv5_block = nn.Sequential(vgg16_bn[34],
                                         vgg16_bn[35],
                                         vgg16_bn[36],
                                         vgg16_bn[37],
                                         vgg16_bn[38],
                                         vgg16_bn[39],
                                         vgg16_bn[40],
                                         vgg16_bn[41],
                                         vgg16_bn[42]
                                         )

        self.up5 = UpConv(512, 512)
        self.up4 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up2 = nn.Sequential(
                                           nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu
                                           )
        self.up1 = nn.Sequential(
                                           nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1)),
                                           nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                                           self.relu,
                                           nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=(1,1)),
                                           )
        # self.upconv5_block = nn.Sequential(
        #                                    nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
        #                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    )
        # self.upconv4_block = nn.Sequential(
        #                                    nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),
        #                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    )
        # self.upconv3_block = nn.Sequential(
        #                                    nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
        #                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    )
        # self.upconv2_block = nn.Sequential(
        #                                    nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu
        #                                    )
        # self.upconv1_block = nn.Sequential(
        #                                    nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1)),
        #                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        #                                    self.relu,
        #                                    nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=(1,1)),
        #                                    )
        self.convlstm = ConvLSTM(input_size=(5, 10),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            f1, idx1 = self.index_MaxPool(self.conv1_block(item))
            f2, idx2 = self.index_MaxPool(self.conv2_block(f1))
            f3, idx3 = self.index_MaxPool(self.conv3_block(f2))
            f4, idx4 = self.index_MaxPool(self.conv4_block(f3))
            f5, idx5 = self.index_MaxPool(self.conv5_block(f4))
            data.append(f5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][-1,:,:,:,:]
        up6 = self.index_UnPool(test, idx5)
        # up5 = self.index_UnPool(self.upconv5_block(up6), idx4)
        # up4 = self.index_UnPool(self.upconv4_block(up5), idx3)
        # up3 = self.index_UnPool(self.upconv3_block(up4), idx2)
        # up2 = self.index_UnPool(self.upconv2_block(up3), idx1)
        # up1 = self.upconv1_block(up2)
        # print(idx4.size(), up6.size())
        up5 = self.up5(idx4, up6)
        up4 = self.up4(idx3, up5)
        up3 = self.up3(idx2, up4)
        # up2 = self.up2(idx1, up3)
        up2 = self.up2(up3)
        diffY = idx1.size()[2] - up2.size()[2]
        diffX = idx1.size()[3] - up2.size()[3]
        up2 = F.pad(up2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        up2 = self.index_UnPool(up2, indices=idx1)
        up1 = self.up1(up2)

        return up1
