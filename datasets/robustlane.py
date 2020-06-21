from core.base import BaseDataset, BaseDataLoader
from utils import palette
from PIL import Image
import torch
import os
import numpy as np


class RobustDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.sequence_num = 4
        self.palette = palette.create_palette('lanesequence')
        super(RobustDataset, self).__init__(data_name='lanesequence', **kwargs)

    def _set_files(self):
        if self.split == 'val':
            self.split = 'test'
        if self.split == 'test':
            self.opp_split = 'train'
        else:
            self.opp_split = 'test'
        txt_path = os.path.join(self.root, self.split + ".txt")
        images_list, label_list = self._readTxt(txt_path)

        self.files = list(zip(images_list, label_list))

    def _load_data(self, item):
        img_path_list = self.files[item]
        # print(img_path_list)
        images = []
        images_name = []
        for i in range(self.sequence_num):
            images_name.append(str(img_path_list[0][i]).split(self.root)[-1])
            images.append(np.asarray(Image.open(img_path_list[0][i]).convert('RGB'), dtype=np.float32))
        label = np.asarray(Image.open(img_path_list[1]), dtype=np.int32)
        label[label == 1] = 255
        return images, label, images_name

    def __getitem__(self, item):
        images, label, images_name = self._load_data(item)

        data = []
        for i in range(len(images)):
            image = Image.fromarray(np.uint8(images[i]))
            data.append(torch.unsqueeze(self.normalize(self.to_tensor(image)), dim=0))
        data = torch.cat(data, dim=0)

        label = torch.from_numpy(label).long()
        # print(label.size())
        sample = {
            'img': data,  # 没有进行归一化, 之后进行了改进
            'label': label,
            'images_name': images_name
        }
        return sample

    def _analyze_data(self):
        return None

    def _readTxt(self, file_path):
        img_list = []
        label_list = []
        with open(file_path, 'r') as fs:
            while True:
                lines = fs.readline()
                if not lines:
                    break
                item = lines.strip().split()
                img_list.append(
                    [filename.replace(self.opp_split, self.split).replace('ouquanlin', 'ubuntu') for filename in
                     item[:-1]])
                label_list.append(item[-1].replace(self.opp_split, self.split).replace('ouquanlin', 'ubuntu'))
        fs.close()
        # print(img_list)
        return img_list, label_list

    @staticmethod
    def collate(batch):
        '''
        相当与将batch变成了一个sample
        :param batch:bath
        :return: sample
        '''
        if isinstance(batch[0]['images'], torch.Tensor):
            imgs = torch.stack([b['images'] for b in batch])
        else:
            imgs = [b['images'] for b in batch]

        if isinstance(batch[0]['label'], torch.Tensor):
            labels = torch.stack([b['label'] for b in batch])
        else:
            labels = [b['label'] for b in batch]

        samples = {
            'img': imgs,
            'label': labels,
            'img_name': [b['images_name'] for b in batch]
        }
        return samples


class RobustLane(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False, crop=True, shuffle=False, flip=False, rotate=False, blur=False, augment=False,
                 val_split=None):

        self.MEAN = [0.3419, 0.3648, 0.3597] # NONE
        self.STD = [0.1669, 0.1784, 0.1872]  # NONE

        if augment is True:
            augment = False

        kwargs = {
            'data_dir': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'crop': crop,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'val': val
        }

        self.dataset = RobustDataset(**kwargs)

        loader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'val_split': val_split
        }

        super(RobustLane, self).__init__(**loader_kwargs)
