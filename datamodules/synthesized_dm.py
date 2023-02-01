import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl


class SynthesizedDataModule(pl.LightningDataModule):
    def __init__(
            self,
            clazz,
            in_dataset,
            out_dataset,
            batch_size=256,
            seed=42,
            dataloader_num_workers=4,
            **kwargs):
        """
        :*_dataset: a tensordataset that contains features on the first and classes on the second
        """
        super().__init__()
        self.save_hyperparameters()
        self.clazz = clazz
        self.in_dataset = in_dataset
        self.out_dataset = out_dataset

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MNISTDataModule")
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--dataloader_num_workers', default=4, type=int)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)

    def prepare_data(self):
        # EMPTY FOR NOW
        return

    def _setup(self, stage=None):
        # if len(in_dataset) != len(out_dataset):
        #     raise Exception('Datasets are not balanced')
        self.dataset = ConcatDataset((self.in_dataset, self.out_dataset))

        train_idx, val_idx = train_test_split(
            np.arange(len(self.dataset)),
            test_size=0.2,
            shuffle=True,
            stratify=torch.concat(
                (self.in_dataset[:][1], self.out_dataset[:][1]))  # the targets 0/1
        )
        # train_idx, val_idx = train_test_split(
        #     np.arange(len(train_idx)),
        #     test_size=0.25,
        #     shuffle=True,
        #     stratify=self.dataset[train_idx][1]
        # )
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)

    def setup(self, stage=None):
        self._setup()

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=min(self.hparams.batch_size,
                                         len(self.train_sampler)),
                          num_workers=self.hparams.dataloader_num_workers,
                          sampler=self.train_sampler,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=min(self.hparams.batch_size,
                                         len(self.val_sampler)),
                          num_workers=self.hparams.dataloader_num_workers,
                          sampler=self.val_sampler,
                          drop_last=True)

    def get_weight(self):
        return len(self.out_dataset) / len(self.in_dataset)

    # def test_dataloader(self):
    #     return DataLoader(self.dataset, batch_size=self.hparams.batch_size,
    # num_workers=self.hparams.dataloader_num_workers, shuffle=False,
    # sampler=self.test_sampler)

    # def predict_dataloader(self):
    #     return self.test_dataloader()
