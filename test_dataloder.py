import json
import datasets
from utils.utils_datasets import show_batch_image, show_batch_images, get_mean_std
import time
from datasets.datasets_dali import VOC


def get_instance(module, base_name, args_name, config, *args):
    return getattr(module, config[base_name]['name'])(*args, **config[base_name][args_name])


def main(config):
    train_loader = get_instance(datasets, 'dataset', 'train_args', config)
    # mean, val = get_mean_std(train_loader)
    # print('mean is {} val is {}'.format(mean, val))  # [0.0046, 0.1073, 0.1376] [0.7319, 0.7800, 0.8414]
    # val_loader = train_loader.get_val_loader()
    # val_loader = get_instance(datasets, 'val', config)
    # test one batch
    start = time.time()
    dataiter = iter(train_loader)
    batch = dataiter.next()
    if config['dataset']['name'] == 'SeqLane' or config['dataset']['name'] == 'RobustLane':
        show_batch_images(batch, train_loader.MEAN, train_loader.STD,
                          palette=train_loader.dataset.palette, if_mask=True)
    else:
        show_batch_image(batch, train_loader.MEAN, train_loader.STD, palette=train_loader.dataset.palette)

    print(f'waste time is {(time.time() - start)/1000}')

    # test all batch
    # for batch_idx, batch in enumerate(tbar):
    #     print(batch[0].size())
    #     print(len(batch))

    # get mean and std
    # mean, std = get_mean_std(train_loader)
    # print(mean, std)
    # train_loader = VOC(root='/home/ouquanlin/datasets/VOCdevkit/VOC2007', split='train', batch_size=4,
    #                    num_workers=4, dali_cpu=False, augment=True, resize=True, resize_size=320).get_loader()
    #
    # print('start iterate')
    #
    # start = time.time()
    # for i, sample in enumerate(train_loader):
    #     images = sample['img'].cuda(non_blocking=True)
    #     labels = sample['label'].squeeze().long().cuda(non_blocking=True)
    # end = time.time()
    # print(f'dali iterate time {end - start}')


if __name__ == '__main__':
    config = json.load(open("./config/lanenet.json"))
    main(config)
