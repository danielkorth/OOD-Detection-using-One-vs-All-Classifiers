# Train One-vs-All classifiers from scratch

import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import argparse

# Lightning stuff
from datamodules.mnist import MNISTDataModule
from lightningmodules.mlp import MLPModule

from utils.transforms import get_mnist_transform

CHECKPOINT_PATH = 'model_checkpoints/'

# Parse CLI input
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data', type=str, default='MNIST')
parser.add_argument('--sched_monitor', type=str, default='val_loss')
parser.add_argument('--sched_mode', type=str, default='min')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--weighted_ce', default=False, type=bool)
parser = pl.Trainer.add_argparse_args(parser)

parser = MLPModule.add_model_specific_args(parser)

args = parser.parse_args()

#####


def setup_callbacks(args, fname):
    model_checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename=fname,
        monitor=args.sched_monitor,
        mode=args.sched_mode
    )
    return [model_checkpoint]

#####


def main():
    pl.seed_everything(args.seed, workers=True)

    dm = eval(f'{args.data}DataModule').from_argparse_args(
        args, **get_mnist_transform())
    for i in range(10):
        dm.make_binary(i)
        model = MLPModule.from_argparse_args(args, binary=True, weight=torch.Tensor(
            [dm.get_weight()]) if args.weighted_ce else None)
        trainer = pl.Trainer.from_argparse_args(
            args,
            gpus=1 if torch.cuda.is_available() else 0,
            callbacks=setup_callbacks(args, f'class_{i}'),
        )

        trainer.fit(model, dm)


if __name__ == '__main__':
    main()
