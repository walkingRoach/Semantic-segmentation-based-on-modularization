import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from PIL import Image
import datasets
import models
import numpy as np
np.set_printoptions(threshold=np.inf)
from utils.utils_datasets import show_detect_image

available_gpus = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if len(available_gpus) > 0 else 'cpu')


def get_instance(module, base_name, args_name, config, *args):
    return getattr(module, config[base_name]['name'])(*args, **config[base_name][args_name])


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC', type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--mode', default=None, type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-w', '--weight', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--image', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    # dataset_type = config['dataset']['type']

    # data loader params
    loader = get_instance(datasets, 'dataset', 'val_args', config)
    # to_tensor = transforms.ToTensor()
    to_tensor = transforms.Compose([
        transforms.Resize((config['dataset']['train_args']['crop_size'], config['dataset']['train_args']['crop_size'])),
        transforms.ToTensor()
    ])
    restore_transform = transforms.Compose([
        DeNormalize(config['dataset']['mean'], config['dataset']['std']),
    ])
    palette = loader.dataset.palette

    model = get_instance(models, 'model', 'args', config)

    checkpoint = torch.load(args.weight)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(model)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with torch.no_grad():
        image_path = args.image
        if image_path is not None:
            image_name = str(image_path).split('/')[-1].split('.')[0]
            image = Image.open(image_path).convert('RGB')
            input = normalize(to_tensor(image)).unsqueeze(0)
            mask = inference(model, input, image, palette)
            mask_path = os.path.join(args.output, image_name + config['name'] + '.png')
            save_mask(mask, mask_path)
        else:
            dataiter = iter(loader)
            while True:
                batch = dataiter.next()
                images = batch['img']
                label = batch['label']
                if config['dataset']['name'] == "SeqLane":
                    images = torch.unbind(images, dim=0)
                    image = images[0][-1]
                    image_name = batch['images_name'][-1][0]
                else:
                    image = images[0]
                    image_name = batch['img_name'][0]
                # images = torch.unbind(images, dim=0)
                # print(image.size())
                mask = inference(model, batch['img'], restore_transform(image), label[0], palette)

                # image_name = batch['images_name'][-1]
                print(image_name)
                mask_path = os.path.join(args.output, image_name[-1])
                # save_mask(mask, mask_path)


def inference(model, input, image, label, palette):
    prediction = model(input.to(device))
    prediction = prediction.max(1)[-1].cpu().numpy()  # 求的是最大的索引值
    # print(prediction)
    # save mask
    # w, h = image.size
    mask = show_detect_image(image, prediction[0], label, palette)
    return mask


def save_mask(mask, save_path):
    mask.save(save_path)


if __name__ == '__main__':
    main()
