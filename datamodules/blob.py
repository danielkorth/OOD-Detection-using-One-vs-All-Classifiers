import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class BlobDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        n_samples=1000,
        n_features=2,
        n_compoments=10,
        cluster_std=1,
        batch_size=128,
        seed=42,
        structured=True,
        radius=20,
        N=100,
        ood_data=False,
        omit_quadrant=False
    ):
        super().__init__()
        self.save_hyperparameters()

        np.random.seed(seed)

    def prepare_data(self):
        if self.hparams.structured:
            self.__make_structured_blobs()
        else:
            self.X, self.y = make_blobs(
                n_samples=self.hparams.n_samples,
                n_features=self.hparams.n_features,
                centers=self.hparams.n_compoments,
                cluster_std=self.hparams.cluster_std,
                random_state=self.hparams.seed
            )
        if self.hparams.ood_data:
            self.__make_ood_circle()

    def setup(self, stage=None):
        self.__setup()

    def __setup(self, ova_num=-1):
        # 80/20 split
        # train_X, test_X, train_y, test_y = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=self.hparams.seed)

        if self.hparams.ood_data:

            if self.hparams.omit_quadrant:
                ff = np.logical_or(self.X_ood[:, 0] > 0, self.X_ood[:, 1] > 0)
                self.X_ood = self.X_ood[ff]
                self.y_ood = self.y_ood[ff]

            self.X = np.concatenate((self.X, self.X_ood), axis=0)
            self.y = np.concatenate((self.y, self.y_ood), axis=0)

        train_X, train_y = self.X, self.y

        if ova_num >= 0:
            # OVA relabeling
            func = np.vectorize(lambda x: 1 if x == ova_num else 0)
            train_y = func(train_y)
            # test_y = func(test_y)

            # OVA undersampling
            n_minority_samples = train_y.sum()
            idx_negative_samples = np.where(train_y == 0)[0]
            idx_positive_samples = np.where(train_y == 1)[0]
            idx_chosen_samples = np.random.choice(
                idx_negative_samples, n_minority_samples)
            idx_samples = np.concatenate(
                (idx_chosen_samples, idx_positive_samples))
            train_X, train_y = train_X[idx_samples], train_y[idx_samples]

        # Create datasets:
        self.train_dataset = TensorDataset(
            torch.FloatTensor(train_X),
            torch.LongTensor(train_y))
        # self.test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))

    def add_ova_filter(self, ova_num):
        self.__setup(ova_num)

    def filter_out(self, ova_num):
        # Aims to filter out all images which do not belong to the positive
        # class of the current OVA model/ova_num
        train_X, train_y = self.X, self.y
        func = np.vectorize(lambda x: x if x == ova_num else -1)
        train_y = func(train_y)
        idx_positive_samples = np.where(train_y == ova_num)[0]
        train_X, train_y = train_X[idx_positive_samples], train_y[idx_positive_samples]

        self.train_dataset = TensorDataset(
            torch.FloatTensor(train_X),
            torch.LongTensor(train_y))

    def get_data(self):
        return (self.X, self.y)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=shuffle)

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False)

    def __make_structured_blobs(self):
        r = self.hparams.radius
        N = self.hparams.N
        clusters = self.hparams.n_compoments
        step = N // clusters
        samples = self.hparams.n_samples

        theta = np.linspace(0, 2 * np.pi, N)
        a = r * np.cos(theta)
        b = r * np.sin(theta)

        X = []
        y = []
        i = 0
        while i * step < N:
            means = (a[i * step], b[i * step])
            cov = [[1, 0],
                   [0, 1]]

            X.append(
                np.random.multivariate_normal(
                    means, cov, samples).squeeze())
            y.append(np.repeat(i, samples))
            i += 1

        self.X = np.stack(X).reshape(-1, 2)
        self.y = np.stack(y).reshape(-1)

    def __make_ood_circle(self):
        r = self.hparams.radius * 2.5
        N = 100
        samples = 10

        theta = np.linspace(0, 2 * np.pi, N)
        a = r * np.cos(theta)
        b = r * np.sin(theta)

        X_ood = []
        y_ood = []

        for _a, _b in zip(a, b):
            means = (_a, _b)
            cov = [[10, 0],
                   [0, 10]]

            X_ood.append(
                np.random.multivariate_normal(
                    means, cov, samples).squeeze())
            y_ood.append(np.repeat(10, samples))

        self.X_ood = np.stack(X_ood).reshape(-1, 2)
        self.y_ood = np.stack(y_ood).reshape(-1)
