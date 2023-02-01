import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
import os

from datamodules.abstractdm import AbstractDataModule

mapping = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


class CIFAR10DataModule(AbstractDataModule):

    def __init__(self, batch_size=64, **kwargs):
        super().__init__()

    def _setup(self, positive_class=-1):

        self.train_dataset = CIFAR10(
            root=os.path.join(
                self.hparams.root,
                'datasets'),
            train=True,
            download=True,
            transform=self.hparams.train_transform)
        self.val_dataset = CIFAR10(
            root=os.path.join(
                self.hparams.root,
                'datasets'),
            train=True,
            download=True,
            transform=self.hparams.test_transform)
        self.test_dataset = CIFAR10(
            root=os.path.join(
                self.hparams.root,
                'datasets'),
            train=False,
            download=True,
            transform=self.hparams.test_transform)

        self.idx_to_class = {
            v: k for k,
            v in self.train_dataset.class_to_idx.items()}

        targets_ = self.train_dataset.targets
        train_idx, val_idx = train_test_split(np.arange(len(
            targets_)), test_size=0.1, random_state=self.hparams.seed, shuffle=True, stratify=targets_)

        if positive_class >= 0:
            self.train_dataset.targets.apply_(
                lambda x: 1 if x == positive_class else 0)
            self.val_dataset.targets.apply_(
                lambda x: 1 if x == positive_class else 0)
            self.test_dataset.targets.apply_(
                lambda x: 1 if x == positive_class else 0)
            # Make the training dataset balanced - or use weighted crossentropy
            train_idx, _ = RandomUnderSampler(random_state=self.hparams.seed).fit_resample(
                train_idx.reshape(-1, 1), targets_[train_idx])

        self.train_sampler = SubsetRandomSampler(train_idx.reshape(-1))
        self.val_sampler = SubsetRandomSampler(val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            shuffle=False)

    def map_idx_to_class(self, number):
        return self.idx_to_class[number]
