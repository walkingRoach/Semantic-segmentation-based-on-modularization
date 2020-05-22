"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU, Conv2d
import os
from PIL import Image

from .misc_functions import preprocess_image, convert_to_grayscale, save_gradient_images, get_positive_negative_saliency


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, save_dir, mean, std):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.save_dir = save_dir
        self.mean = mean
        self.std = std
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = None
        for index, layer in self.model.named_modules():
            if isinstance(layer, Conv2d):
                first_layer = layer
                break
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.named_modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        self.model.zero_grad()
        # Forward pass
        filename = str(input_image).split('/')[-1].split('.')[0]
        x = preprocess_image(Image.open(input_image), mean=self.mean, std=self.std)

        model_output = self.model(x)

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        print(one_hot_output)
        one_hot_output[0][target_class] = 1

        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]

        save_gradient_images(gradients_as_arr, os.path.join(self.save_dir, filename + "_Guided_BP_color"))
        # Convert to grayscale
        grayscale_gradients_as_arr = convert_to_grayscale(gradients_as_arr)
        # Save grayscale gradients
        save_gradient_images(grayscale_gradients_as_arr, os.path.join(self.save_dir, filename + "_Guided_BP_gray"))
        # Positive and negative saliency maps
        pos_sal, neg_sal = get_positive_negative_saliency(gradients_as_arr)
        save_gradient_images(pos_sal, os.path.join(self.save_dir, filename + "_pos_sal"))
        save_gradient_images(neg_sal, os.path.join(self.save_dir, filename + "_neg_sal"))
        print('Layer Guided backprop completed')
