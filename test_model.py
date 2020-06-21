from utils import summary
from models.backbone import vgg
from models.backbone import resnet
from models.backbone import xception
from models.backbone import xception_v2
from models import ENet
from models import UNet
from models import UNet_ConvLSTM, SegNet_ConvLSTM
from models.backbone import mobilenet
import models
from models.backbone import shuffleNet
from models.backbone import eespnet
from models import SeqLane, MutilSeqLane, SingleSeqLane
from models import LaneNet
from models import SeqLaneV2


if __name__ == '__main__':
    # test own model
    # model = getattr(vgg, 'vgg16_bn')(pretrained=False, remove_fc=True)  # 1.(14M, 320p)
    # model = getattr(mobilenet, 'mobilenet')(pretrained=False)
    # model = SegNet(num_classes=21, in_channels=3)
    # model = UNet(num_classes=21, backbone='vgg16_bn')
    # model = UNetResnet(backbone='resnet50', num_classes=21, pretrained=True)
    # model = ResNetDUC(num_classes=21, pretrained=True, dilated=False, multi_grid=False)
    # model = ResNetDUCHDC(num_classes=21, pretrained=True, dilated=False, multi_grid=True)
    # model = getattr(resnet, 'resnet18')(pretrained=False, remove_fc=True, deep_base=False) # 1.(11M, 320p)
    # model = getattr(xception_v2, 'xception')(pretrained=False)
    # model = getattr(shuffleNet, 'shuffleNet')(pretrained=False, remove_fc=False, scale=0.5)
    # model = getattr(mobilenet, 'mobileNetV2')(pretrained=False, remove_fc=True, width_mult=1.0)
    # model = getattr(eespnet, 'eespNet')(pretrained=False, remove_fc=True, s=1)
    # model = ENet(21, 3)
    # model = UNet_ConvLSTM(3, 2)
    # model = SeqLane(2, backbone='mobilenet', pretrained=False, input_size='640p')
    # model = MutilSeqLane(2, backbone='mobilenet', pretrained=False, input_size='320p')
    # model = LaneNet()
    # model = SegNet_ConvLSTM(2)  # (226M 320p)
    model = SeqLaneV2(2, 'vgg', input_size='320p')  # 1.(240M 794M 320p)
    model.cuda()
    # print(model)
    # visual_net = VisualNet("./results/viewer/")
    # visual_net.show_weight(model)
    summary(model, (4, 3, 180, 320))
    # print(model)

    # how to get layer parameter message
    # for index, layer in model.named_parameters():
    #     if select_layer in index:
    #         print(f"{index} : {layer.detach().cpu().numpy().shape}")

    # how to select specify layer
    # for index, layer in model.named_modules():
    #     print(f"{index}")
    #     if index == select_layer:
    #         sl = layer.state_dict()
    #         in_ch = sl['weight'].size(1)
    #         out_ch = sl['weight'].size(0)
    #         print(f"{in_ch} and {out_ch}")
    # # print(model.stage5_encoder[0].cnr[0].register_forward_hook())
