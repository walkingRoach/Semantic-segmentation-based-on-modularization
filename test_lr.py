import torchvision
import torch
import matplotlib.pyplot as plt
from core.lr_scheduler import *


if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16_bn()
    params = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9
    }

    optimizer = torch.optim.SGD(params=vgg16.parameters(), **params)

    epochs = 3
    iter_per_epoch = 150
    lrs = []
    momentums = []

    lr_scheduler = Poly(optimizer, epochs, iter_per_epoch)

    for epoch in range(epochs):
        for i in range(iter_per_epoch):
            optimizer.step()
            lr_scheduler.step(epoch=epoch)
            lrs.append(optimizer.param_groups[0]['lr'])
            momentums.append(optimizer.param_groups[0]['momentum'])

    print(lrs)
    plt.ylabel("learning rate")
    plt.xlabel("iteration")
    plt.plot(lrs)
    plt.show()

    plt.ylabel("momentum")
    plt.xlabel("iteration")
    plt.plot(momentums)
    plt.show()

