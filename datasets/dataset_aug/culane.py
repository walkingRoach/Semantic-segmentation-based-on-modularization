from datasets.aug_core import augment
from datasets.aug_core import BaseAug
from ..aug_core.config import cfg
import os
import cv2


class CULaneAug(BaseAug):
    def __init__(self, file_path, save_root):
        super(CULaneAug, self).__init__(file_path, save_root)

    def augment(self, index):
        img_path, label_path = self.files[index]

        for i in range(1, 20):
            image_path = os.path.join(os.path.dirname(img_path), str(i) + '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError()

            print(image_path)

            img = cv2.imread(image_path)

            resize_img, _ = augment.resize(img)
            filp_img = cv2.flip(resize_img, 1)

            resize_img_save_path = str(image_path).replace('CULane', 'resize_CULane')
            self.create_dir(resize_img_save_path)
            filp_img_save_path = str(image_path).replace('CULane', 'resize-f')
            self.create_dir(filp_img_save_path)
            print(filp_img_save_path)
            print(resize_img_save_path)

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        # 首先对图像进行大小的修改
        resize_img, resize_label = augment.resize(img, label)

        # 图像翻转
        filp_img = cv2.flip(resize_img, 1)
        filp_label = cv2.flip(resize_label, 1)

        resize_img_save_path = str(img_path).replace('clips', 'resize-clips')
        self.create_dir(resize_img_save_path)
        flip_img_save_path = str(img_path).replace('clips', 'resize-f')
        self.create_dir(flip_img_save_path)
        cv2.imwrite(resize_img_save_path, resize_img)
        cv2.imwrite(flip_img_save_path, filp_img)

        resize_label_save_path = str(label_path).replace('truth', 'truth-resize')
        self.create_dir(resize_label_save_path)
        cv2.imwrite(resize_label_save_path, resize_label)

        flip_label_save_path = str(filp_label).replace('truth', 'truth-f')
        self.create_dir(flip_label_save_path)
        cv2.imwrite(flip_label_save_path, filp_label)

    def create_dir(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)


if __name__ == '__main__':
    seqlane = CULaneAug('/home/ouquanlin/datasets/tusimple/train_set/', '..')

    for i in range(len(seqlane.files)):
        seqlane.augment(i)
    cv2.waitKey(0)
