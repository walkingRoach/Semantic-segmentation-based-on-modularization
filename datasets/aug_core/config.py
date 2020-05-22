from .collect import SegConfig

cfg = SegConfig()

## 数据增强配置 ##
# 图像镜像左右翻转
cfg.AUG.MIRROR = True
# 图像上下翻转开关，True/False
cfg.AUG.FLIP = False
# 图像启动上下翻转的概率，0-1
cfg.AUG.FLIP_RATIO = 0.5

# 图像resize的固定尺寸（宽，高），非负
cfg.AUG.FIX_RESIZE_SIZE = (960, 540)
# 图像resize的方式有三种：
# unpadding（固定尺寸），stepscaling（按比例resize），rangescaling（长边对齐）
cfg.AUG.AUG_METHOD = 'unpadding'
# 图像resize方式为stepscaling，resize最小尺度，非负
cfg.AUG.MIN_SCALE_FACTOR = 0.5
# 图像resize方式为stepscaling，resize最大尺度，不小于MIN_SCALE_FACTOR
cfg.AUG.MAX_SCALE_FACTOR = 0.5
# 图像resize方式为stepscaling，resize尺度范围间隔，非负, 等于零直接返回最大最小值
cfg.AUG.SCALE_STEP_SIZE = 0
# 图像resize方式为rangescaling，训练时长边resize的范围最小值，非负
cfg.AUG.MIN_RESIZE_VALUE = 400
# 图像resize方式为rangescaling，训练时长边resize的范围最大值，
# 不小于MIN_RESIZE_VALUE
cfg.AUG.MAX_RESIZE_VALUE = 600
# 图像resize方式为rangescaling, 测试验证可视化模式下长边resize的长度，
# 在MIN_RESIZE_VALUE到MAX_RESIZE_VALUE范围内
cfg.AUG.INF_RESIZE_VALUE = 500

# RichCrop数据增广开关，用于提升模型鲁棒性
cfg.AUG.RICH_CROP.ENABLE = False
# 图像旋转最大角度，0-90
cfg.AUG.RICH_CROP.MAX_ROTATION = 15
# 裁取图像与原始图像面积比，0-1
cfg.AUG.RICH_CROP.MIN_AREA_RATIO = 0.5
# 裁取图像宽高比范围，非负
cfg.AUG.RICH_CROP.ASPECT_RATIO = 0.33
# 亮度调节范围，0-1
cfg.AUG.RICH_CROP.BRIGHTNESS_JITTER_RATIO = 0.5
# 饱和度调节范围，0-1
cfg.AUG.RICH_CROP.SATURATION_JITTER_RATIO = 0.5
# 对比度调节范围，0-1
cfg.AUG.RICH_CROP.CONTRAST_JITTER_RATIO = 0.5
# 图像模糊开关，True/False
cfg.AUG.RICH_CROP.BLUR = False
# 图像启动模糊百分比，0-1
cfg.AUG.RICH_CROP.BLUR_RATIO = 0.1