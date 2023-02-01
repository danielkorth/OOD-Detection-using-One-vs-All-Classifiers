import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchvision.datasets import Places365
from torchvision import transforms as T
import os

from datamodules.abstractdm import AbstractDataModule

# {0: 'airplane',
# 1: 'automobile',
# 2: 'bird',
# 3: 'cat',
# 4: 'deer',
# 5: 'dog',
# 6: 'frog',
# 7: 'horse',
# 8: 'ship',
# 9: 'truck'}


class Places365DataModule(AbstractDataModule):

    def __init__(self, batch_size=64, **kwargs):
        super().__init__()

    def _setup(self, positive_class=-1):

        self.train_dataset = Places365(
            root=os.path.join(
                self.hparams.root,
                'datasets'),
            split='val',
            download=False,
            small=True,
            transform=self.hparams.train_transform)

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            shuffle=True)

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            shuffle=False)
