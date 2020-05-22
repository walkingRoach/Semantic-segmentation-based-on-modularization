import pathlib
import logging
import json
import math
import datetime
from torch.utils import tensorboard
import torch
from core import lr_scheduler
import os
from utils.dir import check_dir
from utils.logger import Logger


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['name'])(*args, **config[name]['args'])


class BaseTrainer:
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['val']['is_val']
        self.start_epoch = 1
        self.improved = False
        self.MEAN = config['dataset']['mean']
        self.STD = config['dataset']['std']

        # SETTING THE DEVICE
        self.device, available_gpus = self._get_available_devices(config['train']['n_gpu'])
        if len(available_gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
            self.loss = torch.nn.DataParallel(self.loss, device_ids=available_gpus)
        self.model.to(self.device)
        self.loss.to(self.device)

        # CONFIG
        self.epochs = self.config['train']['epochs']
        self.save_period = self.config['train']['save_period']

        writer_base_dir = config['train']['writer_dir']
        check_dir(writer_base_dir)
        writer_dir = os.path.join(writer_base_dir, config['name']+"_"+config['dataset']['name'])
        check_dir(writer_dir)
        self.writer = tensorboard.SummaryWriter(str(writer_dir))

        # OPTIMIZER
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()),
                                    'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()),
                                    'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())

        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        self.lr_scheduler = getattr(lr_scheduler, config['lr_scheduler']['name'])(self.optimizer, self.epochs, len(train_loader))

        # monitor
        self.monitor = self.config['train']['monitor']['is_monitor']
        if not self.monitor:
            self.mnt_mode = None
            self.mnt_best = 0
        else:
            self.mnt_mode = self.config['train']['monitor']['type']
            self.mnt_metric = self.config['train']['monitor']['metric']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = self.config['train']['monitor']['early_stop']
        self. not_improved_count = 0

        # CHECKPOINTS
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        checkpoint_base_dir = config['train']['save_dir']
        check_dir(checkpoint_base_dir)
        self.checkpoint_dir = os.path.join(checkpoint_base_dir, config['name']+"_"+config['dataset']['name'] + "_"+start_time)
        check_dir(self.checkpoint_dir)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning('Nbr of GPU requested is {} but only {} are available'.format(n_gpu, sys_gpu))
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info('Detected GPUs: {} Requested: {}'.format(sys_gpu, n_gpu))
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir) + "/checkpoint-epoch" + str(epoch) + ".pth"
        self.logger.info('\nSaving a checkpoint: {} ***'.format(filename))
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, "best_epoch.pth")
            self.logger.info("Saveing a best epoch")
            torch.save(state, str(filename))

    def _resume_checkpoint(self, resume_path):
        self.logger.info('Loading checkpoint : {}'.format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['name'] != self.config['optimizer']['name']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info('Checkpoint <{}> (epoch {}) was loaded'.format(resume_path, self.start_epoch))

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            results = self._train_epoch(epoch)

            if self.do_validation and epoch % self.config['val']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                self.logger.info('\n*** val {} epoch ***'.format(epoch))

                for k, v in results.items():
                    self.logger.info(' {}: {}'.format(k, v))

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)
            else:
                self.train_logger = Logger()
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            if self.mnt_mode is not None and epoch % self.config['val']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min':
                        self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning('The metrics being tracked ({}) has not been calculated. Training stops.'.format(self.mnt_metric))
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info('\nPerformance didn\'t improve for {} epochs'.format(self.early_stoping))
                    self.logger.warning('Training Stop')
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
