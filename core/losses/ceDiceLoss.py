# 就是普通的CE + DICE的组合,非常适合二分类不均的地方
import torch
import torch.nn as nn
from .diceLoss import DiceLoss


class CEDiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=250, weight=None):
        super(CEDiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        if weight is not None:
            weight = torch.tensor(list(weight))
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss
