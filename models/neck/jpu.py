import torch
import torch.nn as nn
import torch.nn.functional as F


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()

        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )

    def forward(self, *input):
        feats = [self.conv5(input[-1]), self.conv4(input[-2]), self.conv3(input[-3])]
        h, w = feats[-1].size(2), feats[-1].size(3)

        feats[-2] = F.upsample(feats[-2], (h, w), mode='bilinear', align_corners=True)
        feats[-3] = F.upsample(feats[-3], (h, w), mode='bilinear', align_corners=True)

        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat
