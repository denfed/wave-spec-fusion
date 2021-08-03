import os

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from logger import get_logger
from utils import msg_box
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Split one dataset into train data_loader and valid data_loader
    """
    logger = get_logger('data_loader')

    def __init__(self, dataset, validation_split=0.0,
                 DataLoader_kwargs=None, do_transform=False):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.split = validation_split
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}

        if Cross_Valid.k_fold > 1:
            fold_msg = msg_box(f"Fold {Cross_Valid.fold_idx}")
            self.logger.info(fold_msg)
            split_idx = dataset.get_split_idx(Cross_Valid.fold_idx - 1)
            train_sampler, valid_sampler = self._get_sampler(*split_idx)
            if do_transform:
                dataset.transform(split_idx)
            super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
            self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
        else:
            if validation_split > 0.0:
                split_idx = self._split_sampler()
                train_sampler, valid_sampler = self._get_sampler(*split_idx)
                if do_transform:
                    dataset.transform(split_idx)
                super().__init__(dataset, sampler=train_sampler, **self.init_kwargs)
                self.valid_loader = DataLoader(dataset, sampler=valid_sampler, **self.init_kwargs)
            else:
                super().__init__(self.dataset, **self.init_kwargs)
                self.valid_loader = None

    def _get_sampler(self, train_idx, valid_idx):
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.init_kwargs['shuffle'] = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _split_sampler(self):
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(self.split, int):
            assert self.split > 0
            assert self.split < self.n_samples, \
                "validation set size is configured to be larger than entire dataset."
            len_valid = self.split
        else:
            len_valid = int(self.n_samples * self.split)

        train_idx, valid_idx = idx_full[len_valid:], idx_full[:len_valid]

        return (train_idx, valid_idx)


class Cross_Valid:
    @classmethod
    def create_CV(cls, k_fold=1, fold_idx=0):
        cls.k_fold = k_fold
        cls.fold_idx = 1 if fold_idx == 0 else fold_idx
        return cls()

    @classmethod
    def next_fold(cls):
        cls.fold_idx += 1

        
class MultiDatasetDataLoader(DataLoader):
    
    def __init__(self, datasets, dataset_batches=None, DataLoader_kwargs=None):
        dss = [i for i in datasets.values()]
        batches = [i for i in dataset_batches.values()]
        self.init_kwargs = DataLoader_kwargs if DataLoader_kwargs is not None else {}
        
        dataset = torch.utils.data.ConcatDataset(dss)
        
        samplers = [torch.utils.data.RandomSampler(i) for i in dss]
        
        batch_sampler = ConcatDatasetBatchSampler(samplers, batches)
        
        super().__init__(dataset, batch_sampler=batch_sampler, collate_fn= lambda b: dict_split_collate(b, dataset_batches), **self.init_kwargs)
        

def dict_split_collate(batch, dataset_batches):
    final_data = {}
    final_target = {}
    offset=0
    
    for key, value in dataset_batches.items():
        collated = default_collate(batch[offset:offset+value])
        
        if isinstance(collated, list):
            final_data[key] = collated[0]
            final_target[key] = collated[1]
        else:
            final_data[key] = collated
        offset += value
        
    return final_data, final_target
        
from torch.utils.data import Sampler
import numpy as np


class ConcatDatasetBatchSampler(Sampler):
    """This sampler is built to work with a standard Pytorch ConcatDataset.
    From SpeechBrain dataio see https://github.com/speechbrain/
    It is used to retrieve elements from the different concatenated datasets placing them in the same batch
    with proportion specified by batch_sizes, e.g 8, 16 means each batch will
    be of 24 elements with the first 8 belonging to the first dataset in ConcatDataset
    object and the last 16 to the second.
    More than two datasets are supported, in that case you need to provide 3 batch
    sizes.
    Note
    ----
    Batched are drawn from the datasets till the one with smallest length is exhausted.
    Thus number of examples in your training epoch is dictated by the dataset
    whose length is the smallest.
    Arguments
    ---------
    samplers : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    batch_sizes: list
        Batch sizes.
    epoch : int
        The epoch to start at.
    """

    def __init__(self, samplers, batch_sizes: (tuple, list), epoch=0) -> None:

        if not isinstance(samplers, (list, tuple)):
            raise ValueError(
                "samplers should be a list or tuple of Pytorch Samplers, "
                "but got samplers={}".format(batch_sizes)
            )

        if not isinstance(batch_sizes, (list, tuple)):
            raise ValueError(
                "batch_sizes should be a list or tuple of integers, "
                "but got batch_sizes={}".format(batch_sizes)
            )

        if not len(batch_sizes) == len(samplers):
            raise ValueError("batch_sizes and samplers should be have same length")

        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):

        iterators = [iter(i) for i in self.samplers]
        tot_batch = []

        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):

        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]

            min_len = min(c_len, min_len)
        return min_len