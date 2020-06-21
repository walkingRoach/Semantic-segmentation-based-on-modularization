import json
import datasets
import models
from core import losses
from trainer import Trainer
from utils.logger import Logger


def get_instance(module, base_name, args_name, config, *args):
    return getattr(module, config[base_name]['name'])(*args, **config[base_name][args_name])


def main(config, resume=False):
    train_logger = Logger()

    train_loader = get_instance(datasets, 'dataset', 'train_args', config)
    if config['dataset']['have_val']:
        val_loader = get_instance(datasets, 'dataset', 'val_args', config)
    else:
        val_loader = train_loader.get_val_loader()

    model = get_instance(models, 'model', 'args', config)

    print(model)

    loss = get_instance(losses, 'loss', 'args', config)

    trainer = Trainer(model=model, loss=loss, resume=resume, config=config, train_loader=train_loader,
                      val_loader=val_loader, train_logger=train_logger, prefetch=True)

    trainer.train()


if __name__ == '__main__':
    config = json.load(open("./config/seqlane.json"))
    main(config)
