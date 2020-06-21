import torch
import time
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
from core import BaseTrainer, DataPrefetcher
from utils import AverageMeter, eval_metrics
from utils.utils_datasets import get_val_image


class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.prefetch = prefetch
        self.config = config
        self.log_step = config['train'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['train']['log_per_iter']:
            self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        if self.device == torch.device('cpu'):
            self.prefetch = False

        if self.prefetch:
            self.train_loader = DataPrefetcher(self.train_loader, device=self.device)
            if val_loader is not None:
                self.val_loader = DataPrefetcher(self.val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')

        self.model.train()

        if self.config['model']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()

        self.wrt_mode = 'train'

        tic = time.time()
        tbar = tqdm(self.train_loader, ncols=130)
        self._reset_metrics()
        for batch_idx, sample in enumerate(tbar):
            torch.cuda.synchronize(device=self.device)
            self.data_time.update(time.time() - tic)

            self.optimizer.zero_grad()

            if self.prefetch:
                data = sample['img']
                target = sample['label']
            else:
                data = sample['img'].to(self.device)
                target = sample['label'].to(self.device)

            if self.config['model']['name'] == 'UNet_ConvLSTM':
                output, _ = self.model(data)
            else:
                output = self.model(data)
            # print(output.size(), target.size())
            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == self.num_classes

            loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())
            self.lr_scheduler.step(epoch=epoch-2)

            # measure time
            torch.cuda.synchronize(device=self.device)
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('{}/loss'.format(self.wrt_mode), loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes, is_train=True)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {} | Acc: {:.4f} mIoU: {:.4f}'.format(
                epoch, self.total_loss.average,
                pixAcc, mIoU))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar('{}/{}'.format(self.wrt_mode, k), v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('{}/Learning_rate_{}'.format(self.wrt_mode, i), opt_group['lr'], self.wrt_step)
            # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
               **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, sample in enumerate(tbar):
                if self.prefetch:
                    data = sample['img']
                    target = sample['label']
                else:
                    data = sample['img'].to(self.device)
                    target = sample['label'].to(self.device)

                if self.config['model']['name'] == 'UNet_ConvLSTM':
                    output, _ = self.model(data)
                else:
                    output = self.model(data)

                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes

                loss = self.loss(output, target)

                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes, is_train=True)
                # seg_metrics = seg_metrics[:4]
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    if self.config['dataset']['name'] == "SeqLane":
                        val_visual.append([sample['img'][0][-1].data.cpu(), target_np[0], output_np[0]])
                    else:
                        val_visual.append([sample['img'][0].data.cpu(), target_np[0], output_np[0]])
                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = get_val_image(val_visual, self.MEAN, self.STD, self.val_loader.dataset.palette)
            self.writer.add_image('{}/inputs_targets_predictions'.format(self.wrt_mode), val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar('{}/loss'.format(self.wrt_mode), self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar('{}/{}'.format(self.wrt_mode, k), v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        # print(self.total_correct, self.total_label)
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        # print(pixAcc)
        # print(self.total_inter, self.total_union)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # print(IoU)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 4),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }
