import os
import numpy as np
from PIL import Image
from core.base import BaseDataset, BaseDataLoader
from utils import palette
import torch


class CULaneDataset(BaseDataset):
    def __init__(self, save_exist=False, **kwargs):
        self.num_classes = 5
        self.save_exist = save_exist
        self.palette = palette.create_palette('culane')
        super(CULaneDataset, self).__init__(data_name='culane', **kwargs)

    def _set_files(self):
        if self.split in ['train', 'val']:
            ann_file = os.path.join(self.root, 'list', self.split + '_gt.txt')
        elif self.split == 'test':
            ann_file = os.path.join(self.root, 'list', self.split + 'txt')
        else:
            raise print('{} is not valid'.format(self.split))

        img_list = []
        label_list = []
        exist_list = []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if self.split in ['train', 'val']:
                    line = line.split(" ")
                    img_list.append(os.path.join(self.root, line[0][1:]))
                    label_list.append(os.path.join(self.root, line[1][1:]))
                    exist_list.append([int(x) for x in line[2:]])
        self.files = list(zip(img_list, label_list, exist_list))

    def _load_data(self, item):
        image = np.asarray(Image.open(self.files[item][0]), dtype=np.float32)
        label = np.asarray(Image.open(self.files[item][1]), dtype=np.int32)
        exist = np.array(self.files[item][2])

        return image, label, exist, self.files[item][0]

    def _analyze_data(self):
        category = {'lane': 0}
        for file in self.files:
            line_num = file[2].count(1)
            category['lane'] += line_num
            
        return category
    
    def __getitem__(self, item):
        image, label, exist, image_name = self._load_data(item)
        
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))

        if self.save_exist:
            sample = {
                'img': self.normalize(self.to_tensor(image)),
                'label': label,
                'exist': torch.from_numpy(exist),
                'img_name': image_name
            }
        else:
            sample = {
                'img': self.normalize(self.to_tensor(image)),
                'label': label,
                'img_name': image_name
            }

        return sample
    
    @staticmethod
    def collate(batch):
        print(len(batch))
        # print('overrode method')
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['label'] is None:
            label = None
            exist = None
        elif isinstance(batch[0]['label'], torch.Tensor):
            label = torch.stack([b['label'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            label = [b['label'] for b in batch]
            exist = [b['exist'] for b in batch]

        samples = {'img': img,
                   'label': label,
                   'exist': exist,
                   'img_name': [x['img_name'] for x in batch]}

        return samples


class CULane(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, num_workers=1, val=False,
                 scale=True, crop=True, shuffle=False, flip=False, rotate=False, blur=False, augment=False,
                 val_split=0.0):
        self.MEAN = [0.3598, 0.3653, 0.3662]
        self.STD = [0.2573, 0.2663, 0.2756]

        dataset_kwargs = {
            'data_dir': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'crop_size': crop_size,
            'base_size': base_size,
            'augment': augment,
            'val': val,
            'scale': scale,
            'crop': crop,
            'flip': flip,
            'rotate': rotate,
            'blur': blur
        }
        if split in ["train", 'val', 'test']:
            self.dataset = CULaneDataset(**dataset_kwargs)
        else:
            raise ValueError('please input a split')

        loader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'val_split': val_split
        }

        super(CULane, self).__init__(**loader_kwargs)
