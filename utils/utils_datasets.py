import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import random
import torchvision
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
from PIL import Image


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def show_random_image(dataset, show_num=6, if_mask=True):
    restore_transform = transforms.Compose([
        DeNormalize(dataset.mean, dataset.std),
        transforms.ToPILImage()
    ])
    viz_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()])

    vis_images = []
    for i in range(show_num):
        random_num = random.randint(0, len(dataset))
        sample = dataset[random_num]
        image = sample['img']
        label = sample['label']
        # print(sample['img_name'])

        image = restore_transform(image)
        label = label.data.cpu().numpy()
        # print(label)
        if if_mask:
            # print(label.shape)
            label = colorize_mask(label, dataset.palette)
            # print(label)
        else:
            # print(label.shape)
            label = Image.fromarray(np.uint8(label))
        image, label = image.convert('RGB'), label.convert('RGB')
        # print(np.array(label))
        [image, label] = [viz_transform(x) for x in [image, label]]
        vis_images.extend([image, label])
    vis_images = torch.stack(vis_images, dim=0)
    grid = torchvision.utils.make_grid(vis_images.cpu(), nrow=2, padding=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def show_batch_images(batch, mean=None, std=None, palette=None, if_mask=True):
    if mean is None or std is None:
        restore_transform = transforms.Compose(
            [transforms.ToPILImage()]
        )
    else:
        print('use mean')
        restore_transform = transforms.Compose([
            DeNormalize(mean, std),
            transforms.ToPILImage()
        ])
    viz_transform = transforms.Compose([
        transforms.ToTensor()])
    # image_size = batch['images'].size(0)
    vis_images = []
    images = batch['img']
    # print(images.size())
    images = torch.unbind(images, dim=0)
    label = batch['label'][0]
    print('label size {}'.format(label.size()))
    label = label.data.cpu().numpy()

    if if_mask:
        # print(label)
        label = colorize_mask(label, palette)
        # print(label)
    else:
        # print("sfs" + str(label.shape))
        label = Image.fromarray(np.uint8(label))
    # print(len(images))
    for image in images[0]:
        # image, label = batch[0][i], batch[1][i]

        print(image.size())
        image = restore_transform(image)
        # image.show()
        # print(type(image), type(label))
        new_image, new_label = image.convert('RGB'), label.convert('RGB')
        [new_image, new_label] = [viz_transform(x) for x in [new_image, new_label]]
        vis_images.extend([new_image, new_label])

    vis_images = torch.stack(vis_images, dim=0)
    grid = torchvision.utils.make_grid(vis_images.cpu(), nrow=2, padding=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def show_batch_image(batch, mean, std, palette):
    restore_transform = transforms.Compose([
        DeNormalize(mean, std),
        transforms.ToPILImage()
    ])
    viz_transform = transforms.Compose([
        transforms.ToTensor()])

    batch_size = batch['img'].size(0)

    vis_images = []

    for i in range(batch_size):
        # image, label = batch[0][i], batch[1][i]
        image = batch['img'][i]
        label = batch['label'][i]
        # print(label.size())
        image = restore_transform(image)

        label = label.data.cpu().numpy()
        label = colorize_mask(label, palette)

        image, label = image.convert('RGB'), label.convert('RGB')
        [image, label] = [viz_transform(x) for x in [image, label]]
        vis_images.extend([image, label])
    vis_images = torch.stack(vis_images, dim=0)
    grid = torchvision.utils.make_grid(vis_images.cpu(), nrow=2, padding=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def get_mean_std(dataloader):
    mean = 0.
    std = 0.
    for sample in dataloader:
        images = sample['img']
        print(sample['img_name'])
        batch_samples = images.size(0)
        # print(batch_samples)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return mean, std


def get_val_image(val_visual, mean, std, palette):
    restore_transform = transforms.Compose([
        DeNormalize(mean, std),
        transforms.ToPILImage()
    ])
    viz_transform = transforms.Compose([
        transforms.ToTensor()])

    val_img = []
    for d, t, o in val_visual:
        d = restore_transform(d)
        t, o = colorize_mask(t, palette), colorize_mask(o, palette)
        d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
        [d, t, o] = [viz_transform(x) for x in [d, t, o]]
        val_img.extend([d, t, o])
    val_img = torch.stack(val_img, 0)
    val_img = torchvision.utils.make_grid(val_img.cpu(), nrow=3, padding=5)

    return val_img


def show_detect_image(image, mask, palette):
    viz_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()])

    vis_img = []
    colorized_mask = colorize_mask(mask, palette)
    mask = colorized_mask.convert('RGB')
    [img, mask] = [viz_transform(x) for x in [image, mask]]
    vis_img.extend([img, mask])
    vis_img = torch.stack(vis_img, 0)
    grid = torchvision.utils.make_grid(vis_img.cpu(), nrow=2, padding=5)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    return colorized_mask


def get_filter_by_name(model, layer_name):
    for index, layer in model.named_parameters():
        if layer_name in index:
            print(layer.detach().cpu().numpy().shape[1])

