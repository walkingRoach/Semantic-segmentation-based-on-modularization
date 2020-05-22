## 针对参数的介绍

### dataset

```
"name": "CityScapes",
    "mean": [0.28689529, 0.32513294, 0.28389176],
    "std": [0.17613647, 0.18099176, 0.17772235],
    "train_args": {
      "data_dir": "/home/ouquanlin/datasets/cityscapes/",
      "batch_size": 4,
      "base_size": 416,
      "crop_size": 384,
      "augment": true,
      "crop": false,
      "scale": false,
      "shuffle": false,
      "flip": false,
      "rotate": false,
      "blur": false,
      "split": "train",
      "num_workers": 2
    }
```
* name : 数据名，当前支持的数据有：voc, coco, culane, cityScapes
* mean : 数据的均方差
* std : 数据的协方差
* data_dir: 数据存放的路径
* base_size: 当crop=False时数据的大小
* crop_size: 当对图像进行切割时候的图像大小
* scale : 当为True时，对图像进行缩放，缩放的大小进行按照左右大小0.5进行随机
* shuffle: 是否对图像的顺序继续你给打乱
* flip: 是否进行图像的翻转
* rotate: 是否进行图像的旋转
* blur: 是否进行图像模糊
* split: 图像数据集类型，包括train, test, val
