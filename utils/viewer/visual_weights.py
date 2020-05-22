import torchvision.utils
from tensorboardX import SummaryWriter
import os
from utils.dir import del_file_in_dir
import torch.nn as nn


class VisualNet:
    def __init__(self, root):
        self.root = root

    def show_weight(self, net):
        weight_path = os.path.join(self.root, "visual_weights")
        del_file_in_dir(path=weight_path)

        writer = SummaryWriter(log_dir=weight_path)

        for name, param in net.named_parameters():
            if 'BatchNorm' not in name or 'bn' not in name:
                print('show {} \'s weight'.format(name))
                writer.add_histogram(name, param, 0)
        writer.close()

    def show_feature(self, net, train_loader):
        feature_path = os.path.join(self.root, "visual_feature")
        del_file_in_dir(feature_path)
        writer = SummaryWriter(log_dir=feature_path)

        image, label = iter(train_loader).next()

        x = image[0]
        img_grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True, nrow=1)
        writer.add_image('raw_img', img_grid, global_step=666)

        net.eval()

        for name, layer in net.named_modules():
            print(x.size())
            x = layer(x)

            if type(layer) == nn.Conv2d:
                img_grid = torchvision.utils.make_grid(x, nrow=4, normalize=True, scale_each=True)
                writer.add_image("{}_feature_maps".format(name), img_grid, global_step=0)
        writer.close()
