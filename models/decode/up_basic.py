import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1):
        super(Conv2dBatchNormRelu, self).__init__()

        self.cnr = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outputs = self.cnr(x)
        return outputs


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=True, regularize=False, regularize_num=0.2,
                 block_num=2, upsampling_method='convTranspose'):
        super(UpConv2d, self).__init__()
        self.up_sample = up_sample
        self.regularize = regularize
        self.regularize_num = regularize_num

        if up_sample:
            if upsampling_method == 'convTranspose':
                self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            elif upsampling_method == 'bilinear':
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
                )

        layer = []
        for i in range(block_num):
            layer.append(Conv2dBatchNormRelu(in_channels=in_channels, out_channels=out_channels, 
                                             kernel_size=3, stride=1, padding=1))
            in_channels = out_channels
            
        self.up_conv = nn.Sequential(*layer)
        
        self.regularizer = nn.Dropout2d(0.1) if regularize else None
        
    def forward(self, x, target_size):
        if self.up_sample:
            x = self.up(x)
            # Padding in case the incomping volumes are of different sizes
            diffY = target_size.size()[2] - x.size()[2]
            diffX = target_size.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
            # Concatenate
        x = self.up_conv(x)

        if self.regularize:
            x = self.regularizer(x)

        return x


class UpConv2dWithCat(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None, up_sample=True,
                 regularize=False, regularize_num=0.2,
                 block_num=3, upsampling_method='convTranspose'):
        super(UpConv2dWithCat, self).__init__()
        self.up_sample = up_sample
        self.regularize = regularize
        self.regularize_num = regularize_num

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        layer = []
        for i in range(block_num):
            layer.append(Conv2dBatchNormRelu(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=3, stride=1, padding=1))
            in_channels = out_channels

        self.up_conv = nn.Sequential(*layer)

        if up_sample:
            if upsampling_method == 'convTranspose':
                self.up = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
            elif upsampling_method == 'bilinear':
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
                )

        if regularize:
            self.regularize = nn.Dropout2d(regularize_num)

    def forward(self, x, x_target):
        if self.up_sample:
            x = self.up(x)
            # Padding in case the incomping volumes are of different sizes
            diffY = x_target.size()[2] - x.size()[2]
            diffX = x_target.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
            # Concatenate
            x = torch.cat([x, x_target], dim=1)
        x = self.up_conv(x)

        if self.regularize:
            x = self.regularizer(x)

        return x