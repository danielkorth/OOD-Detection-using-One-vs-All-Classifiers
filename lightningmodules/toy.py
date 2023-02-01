# from models.base_model import BaseModel
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModule(pl.LightningModule):
    '''
    Model for a 2D toy dataset used to visualize different loss functions and there resulting confidence
    '''

    def __init__(
        self,
        ova=False,
        n_classes=10,
        n_hidden_features=16,
        **kwargs
    ):
        super().__init__()

        if ova:
            n_classes = 1
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        self._build_model()

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     BaseModel.add_model_specific_args(parent_parser)
    #     parser = parent_parser.add_argument_group('ToyModel')
    #     parser.add_argument('--n_hidden_features', default=16, type=int)
    #     return parent_parser

    # @classmethod
    # def from_argparse_args(cls, args, **kwargs):
    #     return pl.utilities.argparse.from_argparse_args(cls, args, **kwargs)

    def _build_model(self):
        # Based on paper
        self.model = nn.Sequential(
            nn.Linear(in_features=2,
                      out_features=self.hparams.n_hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=self.hparams.n_hidden_features,
                      out_features=self.hparams.n_hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=self.hparams.n_hidden_features,
                      out_features=self.hparams.n_hidden_features),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(
            in_features=self.hparams.n_hidden_features, out_features=self.hparams.n_classes)

    def forward(self, x):
        out = self.model(x)
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'val_acc': acc}

    def _share_step(self, batch):
        img, y = batch
        y_hat = self(img)
        loss = self.criterion(y_hat.squeeze(), y.float()
                              if self.hparams.ova else y)
        acc = self._get_accuracy(y_hat, y)
        return loss, acc

    def _get_accuracy(self, y_hat, y):
        if self.hparams.ova:
            pred = F.sigmoid(y_hat)
            pred = pred.round().squeeze()  # default threshold at 0.5
        else:
            pred = y_hat.max(1)[1]  # get index of max logit

        correct = pred.eq(y).cpu().sum()
        all_s = len(y)

        return correct/all_s

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, eta_min=1e-6, T_max=100)
        return {'optimizer': optim, 'scheduler': scheduler}
