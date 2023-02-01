# Trains the Energy method (https://arxiv.org/abs/2010.03759 & https://github.com/wetliu/energy_ood)

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# datamodules
from datamodules.mnist import MNISTDataModule
from datamodules.cifar10 import CIFAR10DataModule
from datamodules.cifar100 import CIFAR100DataModule
from datamodules.fashionmnist import FashionMNISTDataModule

from lightningmodules.energy import EnergyFineTuneModule

# default transforms for each dataset
from utils.transforms import *

CHECKPOINT_PATH = 'model_checkpoints/'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--in_data', default='CIFAR10', type=str)
parser.add_argument('--ood_data', default='CIFAR100', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser = pl.Trainer.add_argparse_args(parser)
parser = EnergyFineTuneModule.add_model_specific_args(parser)

args = parser.parse_args()

# model checkpoint to save best performing model


def setup_callbacks(args, fname):
    model_checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename=fname,
        save_last=True
    )
    return [model_checkpoint]


# main training loop
def main():
    pl.seed_everything(args.seed, workers=True)

    transforms = eval(f'get_{args.in_data.lower()}_transform()')

    # Use our datamodules since they already download data and handle everything for us
    in_dm = eval(f'{args.in_data}DataModule')(**transforms)
    ood_dm = eval(f'{args.ood_data}DataModule')(**transforms)

    # we denote the ood data with target = -1
    ood_dm.train_dataset.targets = torch.tensor(
        -1).repeat(len(ood_dm.train_dataset))
    in_dm.train_dataset.targets = torch.tensor(in_dm.train_dataset.targets)

    train = torch.utils.data.ConcatDataset(
        (in_dm.train_dataset, ood_dm.train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = EnergyFineTuneModule.from_argparse_args(args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=setup_callbacks(args, f'{args.model}')
    )

    trainer.fit(model, train_loader)
    # trainer.test(model, in_dm.test_dataloader())

    # Load best model and get test accuracy
    model = EnergyFineTuneModule.load_from_checkpoint(
        os.path.join(CHECKPOINT_PATH, f'last.ckpt'))
    trainer.test(model, in_dm.test_dataloader())


if __name__ == '__main__':
    main()
