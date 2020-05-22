from datasets.aug_core import augment
from datasets.aug_core import BaseAug
import os
import cv2
import numpy as np


class SeqLaneAug(BaseAug):
    def __init__(self, file_path, save_root, file_name):
        super(SeqLaneAug, self).__init__(file_path, save_root, file_name)
        
    def augment_by_index(self, index):
        img_path, label_path = self.files[index]

        for i in range(1, 20):
            image_path = os.path.join(os.path.dirname(img_path), str(i) + '.jpg')
            if not os.path.exists(image_path):
                raise FileNotFoundError()

            img = cv2.imread(image_path)

            resize_img, _ = augment.resize(img)
            flip_img = cv2.flip(resize_img, 1)

            resize_img_save_path = os.path.join(self.save_root,
                                                str(image_path).replace('clips', 'resize-clips').split(self.file_path)[
                                                    -1])
            self.save_img(resize_img, resize_img_save_path)

            flip_img_save_path = os.path.join(self.save_root,
                                              str(image_path).replace('clips', 'resize-f').split(self.file_path)[-1])
            self.save_img(flip_img, flip_img_save_path)

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        # 首先对图像进行大小的修改
        resize_img, resize_label = augment.resize(img, label)

        # 图像翻转
        flip_img = cv2.flip(resize_img, 1)
        flip_label = cv2.flip(resize_label, 1)

        resize_img_save_path = os.path.join(self.save_root,
                                            str(img_path).replace('clips', 'resize-clips').split(self.file_path)[-1])
        self.save_img(resize_img, resize_img_save_path)
        flip_img_save_path = os.path.join(self.save_root, str(img_path).replace('clips', 'resize-f').split(self.file_path)[-1])
        self.save_img(flip_img, flip_img_save_path)

        resize_label_save_path = os.path.join(self.save_root, str(label_path).replace('truth', 'truth-resize').split(self.file_path)[-1])
        self.save_img(resize_label, resize_label_save_path)

        flip_label_save_path = os.path.join(self.save_root, str(label_path).replace('truth', 'truth-f').split(self.file_path)[-1])
        self.save_img(flip_label, flip_label_save_path)
        self.update_file(resize_img_save_path, resize_label_save_path)
        self.update_file(flip_img_save_path, flip_label_save_path)

    def save_img(self, save_img, save_path):
        self.create_dir(save_path)
        print(save_path)
        cv2.imwrite(save_path, save_img)

    def change_label(self):
        for i in range(len(self.files)):
            img_path, label_path = self.files[i]
            label_img = cv2.imread(label_path)
            label_img[label_img == 1] = 255

    def create_train(self, step):
        for i in range(len(self.files)):
            img_path, label_path = self.files[i]
            for j in step:
                line = ''
                for k in reversed(range(4)):
                    save_path = os.path.join(os.path.dirname(img_path), '{}.jpg '.format(20-j*k))
                    line += save_path
                line += label_path
                line += '\n'
                self.train_f.write(line)

        self.train_f.close()


if __name__ == '__main__':
    seqlane = SeqLaneAug('/home/ouquanlin/datasets/tusimple/train_set/', '..')

    for i in range(len(seqlane.files)):
        seqlane.augment(i)
    cv2.waitKey(0)
