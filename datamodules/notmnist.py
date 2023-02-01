import os
from datamodules.abstractdm import AbstractDataModule
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# code adapted from https://www.kaggle.com/code/eranartzi/notmnist-cnn-3-conv2d-3-pooling-2-fc
def convert_to_pandas(dir_, letters):
    # Retrieve pictures files names
    pictures_files = {}
    for letter in letters:
        images = [name for name in os.listdir(str(dir_) + '/' + letter) if name[-4:] == '.png']
        pictures_files[letter] = images

    # print(pictures_files)
    # Get the actual pictures
    data = {}
    for letter in letters:
        images = []
        for name in pictures_files[letter]:
            try:
                images.append(plt.imread(str(dir_)+'/{}/{}'.format(letter, name)))
            except Exception as e:
                print(e)
                print(str(dir_)+'/{}/{}'.format(letter, name))
        data[letter] = images
        break

    # Merge all data to one list
    X = []
    Y = []
    X_nd = np.zeros(shape=(18724, 784))
    Y_nd = np.zeros(shape=(18724))
    for key, list_ in data.items():
        for img in list_:
            X.append(img.reshape(-1))
            Y.append(key)
    X = np.array(X)
    Y = np.array([mapping[x] for x in Y])

    return pd.DataFrame(X, Y).reset_index()

mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9
}

class NotMNISTDataset(Dataset):

    def __init__(self, path='/u/home/korth/notMNIST_small/', transform=None):
        letters = os.listdir(path)
        self.df = convert_to_pandas(path, letters)

        self.images = self.df.iloc[:, 1:].valuues.reshape((-1, 1, 28, 28))
        self.targets = self.df.iloc[:, 0].values

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.images[idx].squeeze()
        if self.transform:
            x = self.transform(x)
        y = self.targets[idx]
        return x, y


idx_to_class = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J'
}


class NotMNISTDataModule(AbstractDataModule):

    def __init__(self, **kwargs):
        super().__init__()

    def _setup(self, positive_class=-1):
        self.test_dataset = NotMNISTDataset(
            transform=self.hparams.test_transform)

    def train_dataloader(self):
        raise Exception('NotMNIST only supports a test_dataloader()')

    def val_dataloader(self):
        raise Exception('NotMNIST only supports a test_dataloader()')

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            shuffle=False)

    def map_idx_to_class(self, number):
        return idx_to_class[number]
