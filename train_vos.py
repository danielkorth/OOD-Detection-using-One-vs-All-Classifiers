# Trains the VOS method (https://arxiv.org/abs/2202.01197 & https://github.com/deeplearning-wisc/vos)

import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import argparse

# Lightning stuff
from datamodules.mnist import MNISTDataModule
from datamodules.cifar10 import CIFAR10DataModule
from datamodules.cifar100 import CIFAR100DataModule
from lightningmodules.vos import VOSModule

from utils.transforms import *

CHECKPOINT_PATH = 'model_checkpoints/'

# Parse CLI input
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data', type=str, default='MNIST')
parser.add_argument('--sched_monitor', type=str, default='val_loss')
parser.add_argument('--sched_mode', type=str, default='min')
parser.add_argument('--batch_size', type=int, default=256)
parser = pl.Trainer.add_argparse_args(parser)
parser = VOSModule.add_model_specific_args(parser)

args = parser.parse_args()

#####


def setup_callbacks(args, fname):
    model_checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename=fname,
        monitor=args.sched_monitor,
        mode=args.sched_mode,
        save_last=True
    )
    return [model_checkpoint]

#####


def main():
    pl.seed_everything(args.seed, workers=True)

    # Train the backbone feature extractor using softmax
    dm = eval(f'{args.data}DataModule').from_argparse_args(
        args, **eval(f'get_{args.data.lower()}_transform()'))
    model = VOSModule.from_argparse_args(
        args, len_loader=len(dm.train_dataloader()))
    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=1 if torch.cuda.is_available() else 0,
                                            callbacks=setup_callbacks(
                                                args, f'model'),
                                            )

    trainer.fit(model, dm)

    # Load best model
    model = VOSModule.load_from_checkpoint(
        os.path.join(CHECKPOINT_PATH, f'model.ckpt'))
    trainer.test(model, dm)


if __name__ == '__main__':
    main()
