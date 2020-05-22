import argparse
import os
import json
import torch
import models
from utils import GuidedBackprop


def get_instance(module, base_name, args_name, config, *args):
    return getattr(module, config[base_name]['name'])(*args, **config[base_name][args_name])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config/voc.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-w', '--weight', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--image', default='data/voc/dog.jpg', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='./results/viewer/visual_cnn', type=str,
                        help='Output Path')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    # dataset_type = config['dataset']['type']
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    model = get_instance(models, 'model', 'args', config)

    checkpoint = torch.load(args.weight)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    print(model)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_path = args.image
    # set you want to see layer
    selected_layer = 6

    layer_vis = GuidedBackprop(model,  mean=mean, std=std, save_dir=args.output)
    layer_vis.generate_gradients(image_path, selected_layer)


if __name__ == '__main__':
    main()
