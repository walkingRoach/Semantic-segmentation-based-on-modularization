本身的目的是为了实现车道线检测网络，但进一步提升了其利用网络框架的能力。对其他的语义分割算法也进行了实现，本框架是对[该框架](https://github.com/yassouali/pytorch_segmentation#requirements)的扩展实现
----

## 框架介绍
针对网络模型实现的具体实现，为了更好的实现利用网络，通过将网络结构分成backbone, modules, neck, decode_modules，通过这４大模块组合出一个完整model网络用以训练。从而实现可以更好的实现语义分割网络的整合复用,本框架目前还在开发中。
## 内容列表
- [模块介绍](#模块介绍)
- [数据集](#数据集)
- [安装](#安装)
- [使用](#使用)
- [后记](#后记) 

## 模块介绍 
### 1. models网络包括: 
 
- DeeplabV3 
- Duc_hdc
- ENet
- FastFcn 
- PspNet 
- SegNet 
- UNet
- RobustLane
    
### 2. backbone模块包括: 

- EspNet
- EspNetV2
- MobileNetV2
- ResNet 
- ResNext 
- ShuffleNet 
- VggNet
- Xecption
- XecptionV2

### 3. neck模块包括: 

- Psp
- Convlstm
- Assp

### 4. modules模块包括:

- bottleneck 
- SeparbleConv
- EspNet_Moudles

### 5. decode_modules模块:
 
- up_basic (针对普通网络的Decode网络模块,该网络模块暂时没有形成较好的统一复用模块) 

## 数据集

本框架对原框架有个更好的数据集扩展，支持常规的VOC, COCO, CityScapes, Tusimple, CULane。同时利用百度的图像扩展方案，对图像进行扩展，图像的扩展方案可以查看aug_core，支持翻转、旋转、尺寸大小，亮度、饱和度，模糊等应用，可以通过修改config文件进行配置。除了事先对图像进行处理，也可以在训练时对图像进行增强，具体修改config.json文件。除了使用百度的图像增强手段。以后还会添加dali对图像进行增强

### 查看数据集数据
```bash
# 修改test_datasets.py中的选择的数据集。来测试选择的数据集，可以实现显示部分图像，同时支持部分数据库打印图像总类别
python3 test_datasets.py 

# 进一步,可以通过test_dataloder.py来测试dataloder加载的数据集是否符合标准，不过要修改config目录下的配置文件进行数据集的选择
python3 test_datalaoder.py
```

## 安装 
使用的基础环境为pytorch v1.1和python3.6，可以使用下列的指令安装依赖：
```bash
pip3 install -r requirements.text
``` 
为了扩展针对coco数据进行训练，需要额外安装pycocotools，安装方式如下:
```bash
git clone https://github.com/pdollar/coco.git

cd coco/PythonAPI
make
sudo make install
sudo python3 setup.py install
```

 
## 使用
### 设置配置文件
为了方便利用本框架的进行测试和训练, 通过config目录下的json文件对训练参数进行配置。如何配置可以浏览下预留的配置文件, 下面简单介绍下配置文件:
```javascript
{
    "name": "PSPNet",         // training session name
    "n_gpu": 1,               // number of GPUs to use for training.

    "model": {
        "name": "PSPNet", // model目录下的网络都支持
        "args": {
            "backbone": "resnet50",     // 骨架网络
            "num_classes": 1000.        // 训练
            "freeze_bn": false,         // 微调网络时可以使用
            "freeze_backbone": false    // 只训练decode网络
        }
    },

  "lr_scheduler": {
    "name": "Poly",  // 学习率修改策略,支持one_cycle, poly, exponential
    "args": {
      "end_lr": null
    }
  },

  "loss": {
    "name": "CrossEntropyLoss2d",  // 损失函数，支持ceDice, crossEntroppy, dice, focal, lovaszSoftmax
    "args": {
      "weight": [0.02, 1.02]
    }
  },

 "optimizer": {
    "name": "SGD",  // 支持torch自带
    "differential_lr": false,
    "args": {
      "lr": 0.01,
      "weight_decay": 1e-4,
      "momentum": 0.9
    }
  },

  "dataset": {
    "name": "voc",  // 支持voc, coco, cityscapes, tusimple, culane
    "mean": [0.3419, 0.3648, 0.3597],
    "std": [0.1669, 0.1784, 0.1872],
    "train_args": {
      "data_dir": "~/VOC/train_set/320p",
      "batch_size": 8,
      "base_size": [320, 180],
      "crop_size": 180,
      "augment": false,
      "crop": false,
      "scale": false,
      "shuffle": true,
      "flip": false,
      "rotate": false,
      "blur": false,
      "split": "train",
      "num_workers": 2
    },
    "have_val": true,
    "val_args": {
      "data_dir": "~/VOC/test_set/320p",
      "augment": true,
      "batch_size": 2,
      "crop_size": [320, 180],
      "split": "test",
      "num_workers": 2
    }
  },

  "train": {
    "epochs": 300,
    "writer_dir": "~",  //保存训练数据
    "save_dir": "!/checkpoint/",  // 保存权重地点
    "n_gpu": 1,
    "save_period": 20,
    "monitor": {
      "is_monitor": true,
      "type": "max",
      "metric": "Mean_IoU",
      "early_stop": 30
    },
    "tensorboard": true,
    "log_dir": "./results/viewer",  // 保存tensorboard数据
    "log_per_iter": 30
  },

  "val": {
    "is_val": true,
    "val_per_epochs": 5
  }
}
```
 
### 测试 
在训练前，可以测试model模型是否安全可行,同时获取模型的参数信息。当然还可以测试数据集，损失函数，学习率策略，输入下方指令测试：
```bash
# 测试model
python3 test_model.py

# 测试学习率策略
python3 test_lr.py

# 测试损失函数
python3 test_loss.py

# 测试数据集
python3 test_datasets.py

#　测试dataloder是否成功加载数据
python3 test_dataloder.py
```
 
### 训练
配置好文件后，修改train.py文件中json文件，输入下方指令便可训练:
```bash
python3 train.py
```
 
### 评估模型
使用下列指令进行模型的评估:
```bash
# 评估整个数据集
python3 eval.py -c config_file_path -w weight.pth

# 前向推到单一图片
python3 detect.py -c config_file_path -w weight.pth -i image_path -s save_result_img_path
```

### 后记
本框架是本人用于论文研究的基础实现，属于项目副本。不会优先更新这个该框架，但欢迎大家对bug进行指正。QQ:1145893246
