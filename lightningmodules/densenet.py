import torch
import torch.nn as nn
import timm
from lightningmodules.abstractmodule import AbstractModule
from models.densenet import densenet121


class DenseNetModule(AbstractModule):

    def __init__(
        self,
        num_classes=10,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
        optim='torch.optim.SGD',
        scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
        t_max=100,
        eta_min=1e-6,
        **kwargs
    ):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        self._build_model()

    # Default arguments as described in the paper for CIFAR10
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DenseNet')
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--learning_rate', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--nesterov', default=True, type=bool)
        parser.add_argument('--optim', default='torch.optim.SGD', type=str)
        parser.add_argument(
            '--scheduler', default='torch.optim.lr_scheduler.CosineAnnealingLR', type=str)
        parser.add_argument('--t_max', default=100, type=int)
        parser.add_argument('--eta_min', default=1e-6, type=float)
        return parent_parser

    def _build_model(self):
        self.densenet = densenet121(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.densenet(x)

    def penultimate_forward(self, x):
        return self.densenet.penultimate_forward(x)

    def intermediate_forward(self, x, layer_index):
        return self.densenet.intermediate_forward(x, layer_index)

    def feature_list(self, x):
        return self.densenet.feature_list(x)

    def _get_accuracy(self, y_hat, y):
        pred = y_hat.max(1)[1]
        correct = pred.eq(y).cpu().sum()
        return correct / len(y)

    def configure_optimizers(self):
        optim = eval(self.hparams.optim)(self.parameters(), lr=self.hparams.learning_rate,
                                         weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        scheduler = eval(self.hparams.scheduler)(
            optim, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        return {"optimizer": optim, "lr_scheduler": scheduler}
