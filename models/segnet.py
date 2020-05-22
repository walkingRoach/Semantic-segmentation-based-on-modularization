from core.base import BaseModel
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models


class Conv2dBatchNormRelu(nn.Module):
    def __init__(self, batch_name, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1):
        super(Conv2dBatchNormRelu, self).__init__()
        suffix = str(batch_name).split('_')[-1]

        self.cnr = nn.Sequential(OrderedDict([
            ('convd_'+suffix, nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation)),
            ('BatchNorm_'+suffix, nn.BatchNorm2d(out_channels)),
            ('Relu_'+suffix, nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        outputs = self.cnr(x)
        return outputs


class SegNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegNet, self).__init__()

        # self.vgg16 = models.vgg16_bn(pretrained=pretrained)

        self.stage1_encoder = nn.Sequential(*[
            Conv2dBatchNormRelu('encoder_01', in_channels=in_channels,
                                out_channels=64, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_02', in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage2_encoder = nn.Sequential(*[
            Conv2dBatchNormRelu('encoder_11', in_channels=64,
                                out_channels=128, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_12', in_channels=128, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage3_encoder = nn.Sequential(*[
            Conv2dBatchNormRelu('encoder_21', in_channels=128,
                                out_channels=256, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_22', in_channels=256, out_channels=256,
                                kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_23', in_channels=256, out_channels=256,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage4_encoder = nn.Sequential(*[
            Conv2dBatchNormRelu('encoder_31', in_channels=256,
                                out_channels=512, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_32', in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_33', in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage5_encoder = nn.Sequential(*[
            Conv2dBatchNormRelu('encoder_31', in_channels=512,
                                out_channels=512, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_32', in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('encoder_33', in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=1),
        ])

        self.init_vggs_weight()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.stage5_decoder = nn.Sequential(*[
            Conv2dBatchNormRelu('decoder_31', in_channels=512,
                                out_channels=512, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_32', in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_33', in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage4_decoder = nn.Sequential(*[
            Conv2dBatchNormRelu('decoder_21', in_channels=512,
                                out_channels=256, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_22', in_channels=256, out_channels=256,
                                kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_23', in_channels=256, out_channels=256,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage3_decoder = nn.Sequential(*[
            Conv2dBatchNormRelu('decoder_11', in_channels=256,
                                out_channels=128, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_12', in_channels=128, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_13', in_channels=128, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage2_decoder = nn.Sequential(*[
            Conv2dBatchNormRelu('decoder_11', in_channels=128,
                                out_channels=64, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_12', in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.stage1_decoder = nn.Sequential(*[
            Conv2dBatchNormRelu('decoder_01', in_channels=64,
                                out_channels=64, kernel_size=3, stride=1, padding=1),
            Conv2dBatchNormRelu('decoder_02', in_channels=64, out_channels=num_classes,
                                kernel_size=3, stride=1, padding=1),
        ])
        self.unpool = nn.MaxUnpool2d(2, 2)

        self.weight_init(self.stage5_decoder, self.stage4_decoder, self.stage3_decoder, self.stage2_decoder,
                         self.stage1_decoder)

    def forward(self, x):
        # encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage5_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage1_decoder(x)

        return x

    def weight_init(self, *stages):
        for modules in stages:
            for name, m in modules.named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0.)

    def init_vgg_weight(self, vgg16, vgg_init, *init_net):
        init_net = init_net[0]
        # vgg16 = models.vgg16_bn(pretrained=True)
        assert init_net.cnr[0].weight.size() != vgg16.features[vgg_init].weight.size(0)
        init_net.cnr[0].weight.data = vgg16.features[vgg_init].weight.data
        init_net.cnr[0].bias.data = vgg16.features[vgg_init].bias.data
        init_net.cnr[1].weight.data = vgg16.features[vgg_init+1].weight.data
        init_net.cnr[1].bias.data = vgg16.features[vgg_init+1].bias.data

    def init_vggs_weight(self):
        vgg16 = models.vgg16_bn(pretrained=True)
        self.init_vgg_weight(vgg16, 0, self.stage1_encoder[0])
        self.init_vgg_weight(vgg16, 3, self.stage1_encoder[1])

        self.init_vgg_weight(vgg16, 7, self.stage2_encoder[0])
        self.init_vgg_weight(vgg16, 10, self.stage2_encoder[1])

        self.init_vgg_weight(vgg16, 14, self.stage3_encoder[0])
        self.init_vgg_weight(vgg16, 17, self.stage3_encoder[1])
        self.init_vgg_weight(vgg16, 20, self.stage3_encoder[2])

        self.init_vgg_weight(vgg16, 24, self.stage4_encoder[0])
        self.init_vgg_weight(vgg16, 27, self.stage4_encoder[1])
        self.init_vgg_weight(vgg16, 30, self.stage4_encoder[2])

        self.init_vgg_weight(vgg16, 34, self.stage5_encoder[0])
        self.init_vgg_weight(vgg16, 37, self.stage5_encoder[1])
        self.init_vgg_weight(vgg16, 40, self.stage5_encoder[2])

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()
