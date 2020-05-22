import pathlib
import logging
import torch
from utils import AverageMeter, eval_metrics
import numpy as np
from tqdm import tqdm
import time


class BaseEval:
    def __init__(self, model, resume_path, config, val_loader=None):
        self.model = model
        self.config = config
        self.val_loader = val_loader
        self.logger = logging.getLogger(self.__class__.__name__)

        self.num_classes = self.val_loader.dataset.num_classes

        # SETTING THE DEVICE
        self.device, available_gpus = self._get_available_devices(config['train']['n_gpu'])
        self.model.to(self.device)

        self._resume_checkpoint(resume_path)
        # self._eval_metrics()

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

    def _resume_checkpoint(self, resume_path):
        self.logger.info('Loading checkpoint : {}'.format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info('Checkpoint <{}> (epoch {}) was loaded'.format(resume_path, self.start_epoch))

    def evaluate(self):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._eval_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, sample in enumerate(tbar):
                torch.cuda.synchronize(device=self.device)
                tic = time.time()

                data, target = sample['img'].to(self.device), sample['label'].to(self.device)
                # LOSS
                if self.config['model']['name'] == 'UNet_ConvLSTM':
                    output, _ = self.model(data)
                else:
                    output = self.model(data.to(self.device))
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes

                # update time
                torch.cuda.synchronize(self.device)
                self.batch_time.update(time.time() - tic)

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('PixelAcc: {:.2f}, Mean IoU: {:.2f}, Inference Time: {:.4f} |'
                                     .format(pixAcc, mIoU, self.batch_time.average))

            # METRICS TO TENSORBOARD
            seg_metrics = self._get_seg_metrics()

            log = {
                'batch_time': self.batch_time.average,
                **seg_metrics
            }

        return log

    def _eval_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        self.total_precision, self.total_recall, self.total_f1 = 0, 0, 0
        self.error = 0

    def _update_seg_metrics(self, correct, labeled, inter, union, precision, recall, f1, error):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.total_precision += precision
        self.total_recall += recall
        self.total_f1 += f1
        self.total_error = error

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        precision = self.total_precision / (len(self.val_loader.dataset) - self.error)
        recall = self.total_recall / (len(self.val_loader.dataset) - self.error)
        f1 = self.total_f1 / (len(self.val_loader.dataset) - self.error)
        print("have iou is number {}".format(len(IoU)))
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(self.val_loader.dataset.category, np.round(IoU, 3))),
            "Precision": np.round(precision, 3),
            "Recall": np.round(recall, 3),
            "F1_Measure": np.round(f1, 3)
        }
