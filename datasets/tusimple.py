from core.base import BaseDataset, BaseDataLoader
from utils import palette
from PIL import Image
import os
import numpy as np


class TuSimpleDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.create_palette('lanesequence')
        super(TuSimpleDataset, self).__init__(data_name='tusimple', **kwargs)

    def _set_files(self):
        if self.split == 'val':
            self.split = 'test'
        txt_path = os.path.join(self.root, self.split + ".txt")
        images_list, label_list = self._readTxt(txt_path)
        # print(images_list)
        # print(len(images_list), len(label_list))
        self.files = list(zip(images_list, label_list))

    def _load_data(self, item):
        images_list, label_path = self.files[item]
        # print(images_list)
        image_name = (str(images_list).split(self.root)[-1])
        image = (np.asarray(Image.open(images_list).convert('RGB'), dtype=np.float32))
        label = np.asarray(Image.open(label_path).convert('L'), dtype=np.int32)
        label[label == 255] = 1

        return image, label, image_name

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
                items = lines.strip().split()
                img_list.append(os.path.join(self.root, items[0]))
                label_list.append(os.path.join(self.root, items[-1]))
        fs.close()
        return img_list, label_list


class TuSimple(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False, crop=True, shuffle=False, flip=False, rotate=False, blur=False, augment=False,
                 val_split=None):

        self.MEAN = [0.3419, 0.3648, 0.3597]  # NONE
        self.STD = [0.1669, 0.1784, 0.1872]  # NONE

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

        self.dataset = TuSimpleDataset(**kwargs)

        loader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'val_split': val_split
        }

        super(TuSimple, self).__init__(**loader_kwargs)
