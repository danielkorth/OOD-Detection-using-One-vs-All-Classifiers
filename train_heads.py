# Trains the Filtering Heads for the OVAF method that is presented in the thesis

import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import TensorDataset
import argparse
import gc

# Lightning stuff
from datamodules.cifar10 import CIFAR10DataModule
from datamodules.cifar100 import CIFAR100DataModule
# from datamodules.svhn import SVHNDataModule
from datamodules.mnist import MNISTDataModule
from datamodules.fashionmnist import FashionMNISTDataModule
from datamodules.places365 import Places365DataModule
from datamodules.synthesized_dm import SynthesizedDataModule
from lightningmodules.ovaf import FilteringHeadModel, FilteringHeadWrapper

# utils
from utils.multivariate_gaussian import synthesize_features, get_mean_cov_features, get_ood_features, MultivariateNormalUtil
from utils.sampling import sample_negatives, get_all_negatives
from utils.experiments import Experiment
from utils.transforms import *

CHECKPOINT_PATH = 'model_checkpoints/'

# Parse CLI input
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--backbone_id', type=str)
parser.add_argument('--sched_monitor', type=str, default='val_loss')
parser.add_argument('--sched_mode', type=str, default='min')

# dataset splits
parser.add_argument('--real_ood_data', type=bool, default=False)
parser.add_argument('--pct_negative_in_distribution', type=float, default=0.5)
parser.add_argument('--pct_negative_synthesized_outliers',
                    type=float, default=0.5)
parser.add_argument('--pct_negative_real_outliers', type=float, default=0.0)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_classes', default=10, type=int)
# -> ratio of sampled to used: 1/sample_from
parser.add_argument('--sample_from', default=1000, type=int)

# modules
parser.add_argument('--model', default='densenet', type=str)
parser.add_argument('--in_data', default='CIFAR10', type=str)
# description='The DataModule used as "real" OOD training data'
parser.add_argument('--ood_data', default='CIFAR100', type=str)

parser = FilteringHeadModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
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
    ex = Experiment()

    # check whether a model path or guild run id was provided
    if args.backbone_id.endswith('.ckpt'):
        checkpoint = args.backbone_id
    else:
        checkpoint = ex.get_all_checkpoints(args.backbone_id)

    if args.model == 'mlp':
        from lightningmodules.mlp import MLPModule
        backbone = MLPModule.load_from_checkpoint(checkpoint)
    elif args.model == 'densenet':
        from lightningmodules.densenet import DenseNetModule
        backbone = DenseNetModule.load_from_checkpoint(checkpoint)
    else:
        from lightningmodules.wideresnet import WideResNetModule
        backbone = WideResNetModule.load_from_checkpoint(checkpoint)

    backbone = backbone.cuda()

    # Load the data
    transforms = eval(f'get_{args.in_data.lower()}_transform(training=False)')
    dm = eval(f'{args.in_data}DataModule').from_argparse_args(
        args, **transforms)
    if args.ood_data == 'MNIST':
        print(f'get_mnist_to_{args.in_data.lower()}_transform(training=False)')
        ood_dm = MNISTDataModule.from_argparse_args(
            args, **eval(f'get_mnist_to_{args.in_data.lower()}_transform(training=False)'))
    elif args.ood_data == 'Places365':
        print(
            f'get_places365_to_{args.in_data.lower()}_transform(training=False)')
        ood_dm = Places365DataModule.from_argparse_args(
            args, **eval(f'get_places365_to_{args.in_data.lower()}_transform(training=False)'))
    else:
        ood_dm = eval(f'{args.ood_data}DataModule').from_argparse_args(
            args, **transforms)

    # extract data about current features
    # TODO: change dm to train_dataloader() again
    class_mean, class_covariance, list_features, num_sample_per_class = get_mean_cov_features(
        backbone, args.num_classes, dm.train_dataloader())

    # print(num_sample_per_other_class)

    # Sampling in distribution data
    # in_syn = synthesize_features(num_classes=args.num_classes, num_sample_per_class=num_sample_per_class, sample_mean=class_mean,
    # cov=class_covariance, sample_pct=3.0, sample_from=3, in_dist=True)

    # num_sample_per_class = [(x*3 + x) for x in num_sample_per_class]

    # backbone = backbone.cpu()
    if args.pct_negative_real_outliers > 0:
        ood_real = get_ood_features(backbone, ood_dm.val_dataloader())

    del backbone
    gc.collect()

    # collect syntesized outliers
    if args.pct_negative_synthesized_outliers > 0:
        ood_syn = synthesize_features(num_classes=args.num_classes, num_sample_per_class=num_sample_per_class, sample_mean=class_mean,
                                      cov=class_covariance, sample_pct=args.pct_negative_synthesized_outliers, sample_from=args.sample_from)

    # Train the args.n_classes different Sigmoid Heads
    for i, n_samples in enumerate(num_sample_per_class):
        # positive_data = torch.cat((list_features[i].detach().cpu(), in_syn[i].detach().cpu()))
        # positive_dataset = TensorDataset(positive_data, torch.ones(len(positive_data)))

        positive_dataset = TensorDataset(
            list_features[i].detach().cpu(), torch.ones(len(list_features[i])))

        negative_class = None

        # Sample negatives examples from other in_distribution classes
        if args.pct_negative_in_distribution > 0:
            negative_class = sample_negatives(list_features, int(
                n_samples*args.pct_negative_in_distribution), args.num_classes, i)

        # Sample negatives from synthesized outliers
        if args.pct_negative_synthesized_outliers > 0:
            if negative_class is not None:
                negative_class = torch.cat((negative_class, ood_syn[i]))
            else:
                negative_class = ood_syn[i]

        # Sample negatives from real outliers
        if args.pct_negative_real_outliers > 0:
            if negative_class is not None:
                negative_class = torch.cat(
                    (negative_class, ood_real[:int(n_samples*args.pct_negative_real_outliers)]))
            else:
                negative_class = ood_real[:int(
                    n_samples*args.pct_negative_real_outliers)]

        negative_dataset = TensorDataset(
            negative_class.detach().cpu(), torch.zeros(len(negative_class)))

        # print(len(positive_dataset))
        # print(len(negative_dataset))

        dm = SynthesizedDataModule(
            i, positive_dataset, negative_dataset, batch_size=args.batch_size)
        sigmoid_head = FilteringHeadModel.from_argparse_args(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            gpus=1 if torch.cuda.is_available() else 0,
            callbacks=setup_callbacks(args, f'sigmoid-{i}')
        )
        trainer.fit(sigmoid_head, dm)


if __name__ == '__main__':
    main()
