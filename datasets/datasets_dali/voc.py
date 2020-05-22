import os
import numpy as np
from PIL import Image
from random import shuffle
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator


class VOCDali(Pipeline):
    def __init__(self, root, split, batch_size, device_id, dali_cpu=False, local_rank=0, world_size=1, num_workers=2,
                 augment=False, resize=False, resize_size=314, crop=True, crop_size=314, shuffle=True, flip=True,
                 rotate=False, rotate_angle=10.0):
        super(VOCDali, self).__init__(batch_size, num_threads=num_workers, device_id=device_id, seed=12 + device_id)
        self.iterator = iter(VOCIter(batch_size, split, root, shuffle))
        self.split = split
        assert self.split in ['train', 'val']
        dali_device = "gpu" if not dali_cpu else "cpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        self.is_flip = flip
        self.is_rotate = rotate
        self.is_augment = augment
        self.is_crop = crop
        self.is_resize = resize

        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        # self.cast = ops.Cast(device='gpu', dtype=types.INT32)
        self.rng = ops.Uniform(range=(0., 1.))
        self.coin = ops.CoinFlip(probability=0.5)
        # 定义大小
        self.resize = ops.Resize(device="gpu", resize_x=resize_size, resize_y=resize_size, interp_type=types.INTERP_TRIANGULAR)
        # 定义旋转
        self.rotate = ops.Rotate(device="gpu", angle=rotate_angle)
        # 定义翻转
        self.flip = ops.Flip(device="gpu", vertical=1, horizontal=0)
        # 定义剪裁
        self.crop = ops.Crop(device=dali_device, crop_h=crop_size, crop_w=crop_size)
        # 定义正则化
        self.cmnp = ops.CropMirrorNormalize(
            device=dali_device,
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.45734706 * 255, 0.43338275 * 255, 0.40058118 * 255],
            std=[0.23965294 * 255, 0.23532275 * 255, 0.2398498 * 255]
        )

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()

        print()

        new_images = self.jpegs
        new_labels = self.labels
        if self.is_augment and self.split == 'train':
            new_images = self.decode(new_images)
            if self.is_resize:
                new_images = self.resize(new_images.gpu())
                # new_labels = self.resize(new_labels)

            if self.is_crop:
                new_images = self.crop(new_images, crop_pos_x=self.rng(), crop_pos_y=self.rng())
                new_labels = self.crop(new_labels, crop_pos_x=self.rng(), crop_pos_y=self.rng())

            if self.is_flip:
                new_images = self.flip(new_images)
                new_labels = self.flip(new_labels)

            if self.is_rotate:
                new_images = self.rotate(new_images)
                new_labels = self.rotate(new_labels)

        output = self.cmnp(new_images)

        return (output, new_labels)


class VOCIter:
    def __init__(self, batch_size, split, root, shuffle):
        self.root = root
        self.batch_size = batch_size
        self.split = split
        self.shuffle = shuffle
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')
        self._set_files()

    def _set_files(self):
        file_path = os.path.join(self.root, 'ImageSets', 'Segmentation', self.split + '.txt')
        self.files = [line.rstrip() for line in tuple(open(file_path, 'r'))]
        # print(self.files)
        if self.shuffle:
            shuffle(self.files)
        # print(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        for _ in range(self.batch_size):
            image_id = self.files[self.i]
            image_path = os.path.join(self.image_dir, str(image_id) + '.jpg')
            label_path = os.path.join(self.label_dir, str(image_id) + '.png')

            image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
            label = np.asarray(Image.open(label_path), dtype=np.int8)

            batch.append(image)
            labels.append(label)
            self.i = (self.i + 1) % self.n

        return (batch, labels)

    next = __next__


class VOC:
    def __init__(self, root, split, batch_size, dali_cpu=False, local_rank=0, world_size=1, num_workers=2,
                 augment=False, resize=False, resize_size=314, crop=True, crop_size=314, shuffle=True, flip=True,
                 rotate=False, rotate_angle=10.0):
        voc_pipe = VOCDali(root, split, batch_size, device_id=local_rank, dali_cpu=dali_cpu, local_rank=local_rank,
                           world_size=world_size, num_workers=num_workers, augment=augment, resize=resize,
                           resize_size=resize_size, crop=crop, crop_size=crop_size, shuffle=shuffle, flip=flip,
                           rotate=rotate, rotate_angle=rotate_angle)

        voc_pipe.build()

        self.loader = DALIGenericIterator([voc_pipe], ['img', 'label'], 256)

    def get_loader(self):
        return self.loader
