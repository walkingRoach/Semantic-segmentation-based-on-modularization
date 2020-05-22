from core.base import BaseDataset, BaseDataLoader
from utils import palette
from utils import ParseXml
from utils.parse import pretty_table
import numpy as np
import pathlib
from PIL import Image
import os


class VOCDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.palette = palette.create_palette('voc')
        self.num_classes = 21
        super(VOCDataset, self).__init__(data_name='voc', **kwargs)

    def _set_files(self):
        self.root = pathlib.Path(self.root)
        self.dataset_dir = self.root
        if not os.path.exists(self.dataset_dir):
            print('can not find dataset')

        self.image_dir = self.dataset_dir/'JPEGImages'
        self.label_dir = self.dataset_dir/'SegmentationClass'
        self.xml_dir = self.dataset_dir/'Annotations'

        self._test_file_dir()

        self.parse_xml = ParseXml()
        file_list = str(self.dataset_dir/'ImageSets/Segmentation'/self.split) + '.txt'
        self.files = [line.rstrip() for line in tuple(open(file_list, 'r'))]

    def _load_data(self, item):
        image_id = self.files[item]
        image_path = str(self.image_dir/str(image_id)) + '.jpg'
        lable_path = str(self.label_dir/str(image_id)) + '.png'

        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(lable_path), dtype=np.int32)
        # print('start label size : {}'.format(label.shape))
        # print(label)
        image_id = self.files[item].split("/")[-1].split(".")[0]

        return image, label, image_id

    def _analyze_data(self):
        xml_files = [str(self.xml_dir/str(image_id))+'.xml' for image_id in self.files]

        category = self.parse_xml(xml_files)
        self.category = list(category.keys())[:-2]
        table = pretty_table(category)
        return table

    def _test_file_dir(self):
        if not self.xml_dir.exists():
            raise FileNotFoundError("can't find xml dir {}".format(self.xml_dir))
        # if not self.label_dir.exists():
        #     raise FileNotFoundError(f"can't find label dir {self.label_dir}")
        if not self.image_dir.exists():
            raise FileNotFoundError("can't find image dir {}".format(self.image_dir))


class VOC(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, num_workers=1, val=False,
                 scale=True, crop=True, shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=0.0):
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

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
        if split in ["train", 'val']:
            self.dataset = VOCDataset(**dataset_kwargs)
        else:
            raise ValueError('please input a split')

        loader_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'val_split': val_split
        }

        super(VOC, self).__init__(**loader_kwargs)

