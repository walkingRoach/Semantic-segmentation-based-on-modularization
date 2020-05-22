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


def get_instance(module, base_name, args_name, config, *args):
    return getattr(module, config[base_name]['name'])(*args, **config[base_name][args_name])


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
    loader = get_instance(datasets, 'dataset', 'train_args', config)
    # to_tensor = transforms.ToTensor()
    to_tensor = transforms.Compose([
        transforms.Resize((config['dataset']['train_args']['crop_size'], config['dataset']['train_args']['crop_size'])),
        transforms.ToTensor()
    ])
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    palette = loader.dataset.palette

    model = get_instance(models, 'model', 'args', config)
    available_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(available_gpus) > 0 else 'cpu')

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

    image_path = args.image
    image_name = str(image_path).split('/')[-1].split('.')[0]
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        input = normalize(to_tensor(image)).unsqueeze(0)
        prediction = model(input.to(device))
        # prediction = prediction.squeeze(0).cpu().numpy()
        # print(prediction)
        # prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        prediction = prediction.max(1)[1].cpu().numpy()  # 求的是最大的索引值
        print(prediction)
        # save mask
        # w, h = image.size
        mask = show_detect_image(image, prediction[0], palette)
        mask_path = os.path.join(args.output, image_name+config['name']+'.png')
        # mask = mask.resize((w, h), Image.ANTIALIAS)
        mask.save(mask_path)


if __name__ == '__main__':
    main()
