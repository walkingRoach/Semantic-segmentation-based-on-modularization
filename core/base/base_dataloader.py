import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from prefetch_generator import BackgroundGenerator


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split=0.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_images = len(dataset)
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else:
            self.train_sampler, self.val_sampler = None, None
        self.kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            # 'collate_fn': self.dataset.collate
        }
        # print(num_workers)
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_index = int(self.n_images * split)
        np.random.seed(0)

        indexes = np.arange(self.n_images)

        np.random.shuffle(indexes)
        train_indexes = indexes[split_index:]
        val_indexes = indexes[:split_index]

        self.n_images = len(train_indexes)
        train_sampler = SubsetRandomSampler(train_indexes)
        val_sampler = SubsetRandomSampler(val_indexes)

        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None

        return DataLoader(sampler=self.val_sampler, **self.kwargs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            sample = next(self.loaditer)
            self.next_input = sample['img']
            self.next_target = sample['label']
            # self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            sample = {
                'img': input,
                'label': target
            }
            yield sample
            if type(self.stop_after) is int and (count > self.stop_after):
                print('i want to stop iter')
                break

