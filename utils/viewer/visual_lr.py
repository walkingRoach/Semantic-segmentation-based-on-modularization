import matplotlib.pyplot as plt


def show_lr_epoch(lr_iter_dict):
    plt.plot(*zip(*sorted(lr_iter_dict.items())))
    plt.xlabel("iters")
    plt.ylabel("lr")
    plt.show()


def show_lr_loss(lr_loss_dict):
    plt.plot(*zip(*sorted(lr_loss_dict.items())))
    plt.xlabel("loss")
    plt.ylabel("lr")
    plt.show()
