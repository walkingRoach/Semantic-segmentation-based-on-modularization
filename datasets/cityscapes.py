from core.base import BaseDataset, BaseDataLoader
from utils import palette
from utils import parse_cityscapes
from utils.parse import pretty_table
import torch
import numpy as np
from PIL import Image
import os
from utils import recursive_glob


ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


class CityScapesDataset(BaseDataset):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 19
        self.mode = mode
        self.palette = palette.create_palette('cityscapes')
        self.id_to_trainId = ID_TO_TRAINID
        super(CityScapesDataset, self).__init__(data_name='cityscapes', **kwargs)

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'val']) or \
        (self.mode == 'coarse' and self.split in ['train', 'train_extra', 'val'])

        img_list, label_list = None, None
        if self.mode == 'coarse':
            img_dir_name = 'leftImg8bit_trainextra' if self.split == 'train_extra' else 'leftImg8bit_trainvaltest'
            label_path = os.path.join(self.root, 'gtCoarse', 'gtCoarse', self.split)
        else:
            img_list = recursive_glob(os.path.join(self.root, 'leftImg8bit', self.split), suffix='.png')
            # label_list = os.path.join(self.root, 'gtFine', self.split, img_list)
            label_list = [os.path.join(self.root, 'gtFine', self.split, label_name.split('/')[-1].split('_')[0],
                                       label_name.split('/')[-1].replace('leftImg8bit', 'gtFine_labelIds')) for label_name in img_list]

        self.files = list(zip(img_list, label_list))

    def _load_data(self, item):
        image_path, label_path = self.files[item]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        for k, v in self.id_to_trainId.items():
            label[label == k] = v
        return image, label, image_name

    def _analyze_data(self):
        json_files = recursive_glob(os.path.join(self.root, 'gtFine', self.split), suffix='json')

        category = parse_cityscapes(json_files)
        self.category = list(category.keys())[:-2]
        table = pretty_table(category)
        return table


class CityScapes(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 mode='fine', val=False, crop=True, shuffle=False, flip=False, rotate=False, blur=False, augment=False,
                 val_split=None):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

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

        self.dataset = CityScapesDataset(mode=mode, **kwargs)

        loader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'val_split': val_split
        }

        super(CityScapes, self).__init__(**loader_kwargs)
