import math
from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations
    """

    def __init__(self, optimizer, end_lr, num_epochs, iters_per_epoch=0, last_epoch=-1):
        self.end_lr = end_lr
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        # self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        r = T / self.N
        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
