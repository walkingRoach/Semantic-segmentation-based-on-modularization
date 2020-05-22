import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, data_name, data_dir, split, mean, std, base_size=None, augment=True, val=False,
                 crop_size=416, crop=True, scale=True, flip=True, rotate=True, blur=False):
        self.data_name = data_name
        self.root = data_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.crop = crop
        self.augment = augment
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val

        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, item):
        raise NotImplementedError

    def _analyze_data(self):
        raise NotImplementedError

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
            int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        while True:  # The rotated crop must have some objects
            image_new, label_new = image, label
            h, w, _ = image_new.shape
            # Rotate the image with an angle between -10 and 10
            if self.rotate:
                # print('rotate')
                angle = random.randint(-10, 10)
                center = (w / 2, h / 2)
                rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image_new = cv2.warpAffine(image_new, rot_matrix, (w, h),
                                           flags=cv2.INTER_LINEAR)  # , borderMode=cv2.BORDER_REFLECT)
                label_new = cv2.warpAffine(label_new, rot_matrix, (w, h),
                                           flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)

            # Padding to return the correct crop size
            if self.crop:
                if self.crop_size:
                    pad_h = max(self.crop_size - h, 0)
                    pad_w = max(self.crop_size - w, 0)
                    pad_kwargs = {
                        "top": 0,
                        "bottom": pad_h,
                        "left": 0,
                        "right": pad_w,
                        "borderType": cv2.BORDER_CONSTANT, }
                    if pad_h > 0 or pad_w > 0:
                        image_new = cv2.copyMakeBorder(image_new, value=0, **pad_kwargs)
                        label_new = cv2.copyMakeBorder(label_new, value=0, **pad_kwargs)

                    # Cropping
                    h, w, _ = image_new.shape
                    start_h = random.randint(0, h - self.crop_size)
                    start_w = random.randint(0, w - self.crop_size)
                    end_h = start_h + self.crop_size
                    end_w = start_w + self.crop_size
                    image_new = image_new[start_h:end_h, start_w:end_w]
                    label_new = label_new[start_h:end_h, start_w:end_w]

            if label_new.sum() != 0:
                image = image_new
                label = label_new
                break

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def __getitem__(self, item):
        image, label, image_name = self._load_data(item)
        # print(label.shape)
        # print(image.shape)
        if self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        # print(label.size())
        image = Image.fromarray(np.uint8(image))

        sample = {
            'img': self.normalize(self.to_tensor(image)),
            'label': label,
            'img_name': image_name
        }
        return sample

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "    Dataset : " + self.__class__.__name__ + "\n"
        fmt_str += "  image-number : {} \n".format(self.__len__())
        fmt_str += str(self._analyze_data())
        return fmt_str

    @staticmethod
    def collate(batch):
        # print(batch.size())
        if isinstance(batch[0]['img'], torch.Tensor):
            imgs = torch.stack([b['img'] for b in batch])
        else:
            imgs = [b['img'] for b in batch]

        if isinstance(batch[0]['label'], torch.Tensor):
            labels = torch.stack([b['label'] for b in batch])
        else:
            labels = [b['label'] for b in batch]

        samples = {
            'img': imgs,
            'label': labels,
            'img_name': [b['img_name'] for b in batch]
        }
        return samples
