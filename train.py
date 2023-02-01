# Trains the classifiers (used as basis for further fine-tuning)

import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# datamodules
from datamodules.mnist import MNISTDataModule
from datamodules.cifar10 import CIFAR10DataModule
from datamodules.cifar100 import CIFAR100DataModule

from lightningmodules.energy import EnergyFineTuneModule

# default transforms for each dataset
from utils.transforms import *

CHECKPOINT_PATH = 'model_checkpoints/'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model', default='densenet', type=str)
parser.add_argument('--data', default='CIFAR10', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--sched_monitor', default='val_loss', type=str)
parser.add_argument('--sched_mode', default='min', type=str)
parser = pl.Trainer.add_argparse_args(parser)

# parse the specific args for a model
if 'mlp' in sys.argv:
    from lightningmodules.mlp import MLPModule
    MLPModule.add_model_specific_args(parser)
    model_class = MLPModule
elif 'densenet' in sys.argv:
    from lightningmodules.densenet import DenseNetModule
    DenseNetModule.add_model_specific_args(parser)
    model_class = DenseNetModule
elif 'wideresnet' in sys.argv:
    from lightningmodules.wideresnet import WideResNetModule
    WideResNetModule.add_model_specific_args(parser)
    model_class = WideResNetModule
else:
    print('no valid model provided')
    sys.exit()

args = parser.parse_args()

def setup_callbacks(args, fname):
    model_checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename=fname,
        monitor=args.sched_monitor,
        mode=args.sched_mode
    )
    return [model_checkpoint]


def main():
    pl.seed_everything(args.seed, workers=True)

    transforms = eval(f'get_{args.data.lower()}_transform()')
    dm = eval(f'{args.data}DataModule').from_argparse_args(args, **transforms)

    model = model_class.from_argparse_args(args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=setup_callbacks(args, f'{args.model}')
    )

    trainer.fit(model, dm)

    # Load best model and get test accuracy
    model = model_class.load_from_checkpoint(
        os.path.join(CHECKPOINT_PATH, f'{args.model}.ckpt'))
    trainer.test(model, dm)


if __name__ == '__main__':
    main()
