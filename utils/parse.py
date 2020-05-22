import xml.etree.ElementTree as ET
from prettytable import PrettyTable
from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
from PIL import Image
import json


class ParseXml:
    def __init__(self):
        self.category = {}

    def __call__(self, xml_paths):
        for xml_path in xml_paths:
            root = ET.parse(xml_path).getroot()

            for obj in root.iter('object'):
                obj_name = obj.find('name').text
                if obj_name not in self.category.keys():
                    self.category[obj_name] = 1
                else:
                    self.category[obj_name] += 1

        self.category['total'] = sum(self.category.values())
        return self.category


class ParseJson:
    def __init__(self):
        self.category = {}

    def __call__(self, json_path, ids=None):
        self.coco = COCO(json_path)
        categories = self.coco.loadCats(self.coco.getCatIds())
        if ids is not None:
            imgIds = self.coco.getImgIds(imgIds=ids)
        else:
            imgIds = self.coco.getImgIds()

        for imgId in imgIds:
            annIds = self.coco.getAnnIds(imgId, iscrowd=None)
            anns = self.coco.loadAnns(annIds)

            for i in range(len(anns)):
                if anns[i]['category_id'] not in self.category.keys():
                    self.category[anns[i]['category_id']] = 1
                else:
                    self.category[anns[i]['category_id']] += 1

        return self.category


class ParseCoco:
    def __init__(self, json_path):
        self.coco = COCO(json_path)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cate_ids = self.coco.getCatIds()

        self.imgIds = self.coco.getImgIds()
        self.coco_mask = mask

    def __call__(self, item):
        img_info = self.coco.loadImgs(self.imgIds[item])[0]
        filename = img_info['file_name']

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        ann_info = self.coco.loadAnns(ann_ids)

        img_mask = Image.fromarray(self.parse_ann_info(ann_info, img_info['height'], img_info['width']))

        return filename, img_mask

    def parse_ann_info(self, ann_info, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for i, ann in enumerate(ann_info):
            rle = coco_mask.frPyObjects(ann['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = ann['category_id']
            if cat in self.cate_ids:
                c = self.cate_ids.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * ((np.sum(m, axis=2) > 0) * c).astype(np.uint8)

        return mask


cityscapes_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']


def parse_cityscapes(json_list):
    category = {}
    for json_path in json_list:
        root = json.load(open(json_path))
        for obj in root['objects']:
            if obj['label'] in cityscapes_names:
                if obj['label'] not in category.keys():
                    category[obj['label']] = 1
                else:
                    category[obj['label']] += 1
    return category


def pretty_table(category):
    table = PrettyTable(['category', 'number'])

    for key, val in category.items():
        table.add_row([key, val])

    return table


if __name__ == '__main__':
    test_dict = {'ou':1, 'li':3, 'total':5, 'ss':4}

    print(pretty_table(test_dict))
