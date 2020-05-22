import os
import numpy as np
import torch
from torch.optim import Adam
from utils.dir import del_file_in_dir
from .misc_functions import preprocess_image, recreate_image, save_image
import matplotlib.pyplot as plt


class CNNLayerVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, mean, std, save_dir):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter

        self.conv_output = 0
        self.features = 0
        self.save_dir = save_dir
        self.mean, self.std = mean, std
        self.total_filters_in_layer = 0

        del_file_in_dir(save_dir)

    def hook_layel(self):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.selected_filter]

        for index, layer in self.model.named_modules():
            if index == self.selected_layer:
                layer.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        self.hook_layel()
        random_image = np.uint8(np.random.uniform(150, 180, (380, 380, 3)))
        image = preprocess_image(random_image, self.mean, self.std)

        optimizer = Adam([image], lr=0.1, weight_decay=1e-6)  # 使用这个可以快速进行快速的优化image
        for i in range(1, 31):
            optimizer.zero_grad()
            x = image
            # print(x.size())
            self.model(x)

            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))

            loss.backward()
            optimizer.step()

            created_image = recreate_image(image, self.mean, self.std)

            if i % 5 == 0:
                img_path = os.path.join(self.save_dir,
                                        "l{}_f{}_iter{}.jpg".format(self.selected_layer, self.selected_filter, i))
                save_image(created_image, img_path)

    def save_features(self):
        def hook_fn(module, input, output):
            self.features = torch.tensor(output, requires_grad=True)

        # self.model.stage5_encoder[0].cnr[0].register_forward_hook(hook_fn)
        for index, layer in self.model.named_modules():
            if index == self.selected_layer:
                layer_state_dict = layer.state_dict()
                self.total_filters_in_layer = layer_state_dict['weight'].size(0)
                layer.register_forward_hook(hook_fn)
                break

    def cal_layer_mean_activation(self, image_path):
        self.save_features()

        image = preprocess_image(image_path, self.mean, self.std)

        x = self.model(image)

        mean_act = [-self.features[0, i].mean().item() for i in range(self.total_filters_in_layer)]
        print("max filter is {}".format(np.array(mean_act).argmax()))
        plt.figure(figsize=(7, 5))
        act = plt.plot(mean_act, linewidth=2.)
        extraticks = [filter]
        ax = act[0].axes
        ax.set_xlim(0, self.total_filters_in_layer)
        # plt.axvline(x=filter, color='grey', linestyle='--')
        ax.set_xlabel("feature map")
        ax.set_ylabel("mean activation")
        # ax.set_xticks([0, 200, 400] + extraticks)
        plt.show()
        plt.savefig(os.path.join(self.save_dir, 'mean_activation_layer_' + str(self.selected_layer) + '_filter_'
                                 + str(filter) + '.png'))