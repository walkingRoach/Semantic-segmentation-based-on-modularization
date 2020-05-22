from core.base import BaseDataset, BaseDataLoader
from PIL import Image
import numpy as np
import scipy.io as sio
from utils import palette
import pathlib
from utils import ParseJson, ParseCoco
from utils import parse


class CocoStuff10K(BaseDataset):
    def __init__(self, **kwargs):
        self.num_classes = 182
        self.palette = palette.create_palette('coco')
        super(CocoStuff10K, self).__init__(**kwargs)

    def _set_files(self):
        self.root = pathlib.Path(self.root)
        self.image_dir = self.root/'images'
        self.label_dir = self.root/'annotations'
        self.category_dir = self.root/'label.txt'
        category_list = [str(line).split(':') for line in open(self.category_dir)]
        self.category_dict = {int(t[0]): t[1].strip('\n') for t in category_list}

        self.parse_json = ParseJson()
        if self.split in ['train', 'test', 'all']:
            file_list = str(self.root/'imageLists'/self.split) + '.txt'
            self.files = [name.rstrip() for name in tuple(open(file_list, "r"))]
        else:
            raise ValueError("Invalid split name {} choose one of [train, test, all]".format(self.split))

    def _load_data(self, item):
        image = np.asarray(Image.open(str(self.image_dir/self.files[item]) + '.jpg'), dtype=np.float32)
        label = sio.loadmat(str(self.label_dir/self.files[item]) + '.mat')['S']

        label -= 1
        label[label == -1] = 255

        return image, label, self.files[item]

    def _analyze_data(self):
        self.json_path = self.root/'cocostuff-10k-v1.1.json'
        imgIds = [int(str(file_id.split('_')[-1])) for file_id in self.files]
        category = self.parse_json(self.json_path, imgIds)
        category = {self.category_dict[key]: val for key, val in category.items()}
        category = parse.pretty_table(category)
        return category


class CocoStandard(BaseDataset):
    def __init__(self, number_classes, **kwargs):
        self.number_classes = number_classes
        self.palette = palette.create_palette(name=None, cls_num=self.number_classes)
        super(CocoStandard, self).__init__(**kwargs)

    def _set_files(self):
        self.root = pathlib.Path(self.root)
        self.category_dir = self.root/'label.txt'
        category_list = [str(line).split(':') for line in open(self.category_dir)]
        self.category_dict = {int(t[0]): t[1].strip('\n') for t in category_list}

        if self.split in ['train', 'test']:
            self.image_dir = self.root/self.split
            self.json_dir = str(self.root/'annotations') + '/instances_' + self.split + '.json'
            self.parse_coco = ParseCoco(self.json_dir)
            self.files = [str(file_path).split(str(self.image_dir))[-1] for file_path in self.image_dir.glob('*jpg')]
        else:
            raise ValueError("Invalid split name {} choose one of [train, test]".format(self.split))

    def _load_data(self, item):
        filename, label = self.parse_coco(item)
        image = np.asarray(Image.open(str(self.image_dir/filename)), dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)

        return image, label, filename

    def _analyze_data(self):
        parse_json = ParseJson()
        category = parse_json(self.json_dir)
        category = {self.category_dict[key]: val for key, val in category.items()}
        print(len(category.keys()))
        category = parse.pretty_table(category)
        return category


class COCO(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, num_workers=1, val=False,
                 scale=True, crop=True, shuffle=False, flip=False, rotate=False, blur=False, augment=False):
        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        dataset_kwargs = {
            'data_dir': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'crop_size': crop_size,
            'base_size': base_size,
            'augment': augment,
            'crop': crop,
            'val': val,
            'scale': scale,
            'flip': flip,
            'rotate': rotate,
            'blur': blur
        }

        if split in ['train', 'test', 'all']:
            self.dataset = CocoStuff10K(**dataset_kwargs)
        else:
            raise ValueError('please input train, test, all')

        loader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
        }

        super(COCO, self).__init__(**loader_kwargs)
