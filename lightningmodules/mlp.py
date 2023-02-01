import torch
import torch.nn as nn
from lightningmodules.abstractmodule import AbstractModule
from models.mlp import MLP


class MLPModule(AbstractModule):

    def __init__(
        self,
        learning_rate=1e-3,
        optim='torch.optim.Adam',
        scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
        t_max=20,
        eta_min=1e-6,
        binary=False,
        weight=None,
        **kwargs
    ):
        super().__init__()

        if binary:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self._build_model()

    # Default arguments as described in the paper for CIFAR10
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('MLP')
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--optim', default='torch.optim.Adam', type=str)
        parser.add_argument(
            '--scheduler', default='torch.optim.lr_scheduler.CosineAnnealingLR', type=str)
        parser.add_argument('--t_max', default=20, type=int)
        parser.add_argument('--eta_min', default=1e-6, type=float)
        return parent_parser

    def _build_model(self):
        self.model = MLP(self.hparams.binary)

    def forward(self, x):
        return self.model(x)

    def penultimate_forward(self, x):
        return self.model.penultimate_forward(x)

    def intermediate_forward(self, x, layer_index):
        return self.model.intermediate_forward(x, layer_index)

    def feature_list(self, x):
        return self.model.feature_list(x)

    def _share_step(self, batch):
        img, y = batch
        y_hat = self(img)
        loss = self.criterion(y_hat.squeeze(), y.float()
                              if self.hparams.binary else y)
        acc = self._get_accuracy(y_hat, y)
        return loss, acc

    def _get_accuracy(self, y_hat, y):
        if self.hparams.binary:
            pred = torch.sigmoid(y_hat)
            pred = pred.round().squeeze()
            correct = pred.eq(y).cpu().sum()
        else:
            pred = y_hat.max(1)[1]  # get the index of the max logit
            correct = pred.eq(y).cpu().sum()
        return correct / len(y)

    def configure_optimizers(self):
        optim = eval(self.hparams.optim)(
            self.parameters(), lr=self.hparams.learning_rate)
        scheduler = eval(self.hparams.scheduler)(
            optim, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        return {"optimizer": optim, "lr_scheduler": scheduler}
