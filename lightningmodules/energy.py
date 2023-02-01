import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.experiments import Experiment


class EnergyFineTuneModule(pl.LightningModule):
    """
    Module that fine-tunes models according to Energy Score: implementation based on https://github.com/wetliu/energy_ood
    """

    def __init__(self,
                 model='densenet',
                 backbone_id='',
                 num_classes=10,
                 learning_rate=1e-4,
                 momentum=0.9,
                 weight_decay=5e-4,
                 optim='torch.optim.SGD',
                 scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
                 t_max=10,
                 eta_min=1e-6,
                 m_in=-25.,
                 m_out=-7.,
                 **kwargs
                 ):
        super().__init__()

        self.save_hyperparameters()
        self._build_model()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('VOS')
        parser.add_argument('--model', default='densenet', type=str)
        parser.add_argument('--backbone_id', default='', type=str)
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--optim', default='torch.optim.SGD', type=str)
        parser.add_argument(
            '--scheduler', default='torch.optim.lr_scheduler.CosineAnnealingLR', type=str)
        parser.add_argument('--t_max', default=10, type=int)
        parser.add_argument('--eta_min', default=1e-6, type=float)
        parser.add_argument('--m_in', default=-25., type=float)
        parser.add_argument('--m_out', default=-7., type=float)
        return parent_parser

    def _build_model(self):
        ex = Experiment()
        # check whether a model path or guild run id was provided
        if self.hparams.backbone_id.endswith('.ckpt'):
            checkpoint = self.hparams.backbone_id
        else:
            checkpoint = ex.get_all_checkpoints(self.hparams.backbone_id)

        # using modules for convenience since the models were saved that way
        if self.hparams.model == 'mlp':
            from lightningmodules.mlp import MLPModule
            self.model = MLPModule.load_from_checkpoint(checkpoint).model
        elif self.hparams.model == 'densenet':
            from lightningmodules.densenet import DenseNetModule
            self.model = DenseNetModule.load_from_checkpoint(
                checkpoint).densenet
        else:
            from lightningmodules.wideresnet import WideResNetModule
            self.model = WideResNetModule.load_from_checkpoint(checkpoint).wrn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, y = batch

        x = self(img)

        # filter out OOD data
        mask = y != -1

        # calculate loss with all positive examples
        loss = F.cross_entropy(x[mask], y[mask])

        # calculate energies and loss
        Ec_in = -torch.logsumexp(x[mask], dim=1)
        Ec_out = -torch.logsumexp(x[~mask], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-self.hparams.m_in), 2).mean() +
                     torch.pow(F.relu(self.hparams.m_out-Ec_out), 2).mean())

        acc = self._get_accuracy(x[mask], y[mask])
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        _, acc = self._share_step(batch)
        self.log('test_acc', acc, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        img, y = batch
        logit_all = self(img)
        logit_max, y_hat = logit_all.max(1)
        prob_all = F.softmax(logit_all, dim=1)
        prob_max = prob_all.max(1)[0]
        return {'y': y, 'y_hat': y_hat, 'prob_max': prob_max, 'prob_all': prob_all, 'logit_max': logit_max, 'logit_all': logit_all}

    def _share_step(self, batch):
        img, y = batch
        y_hat = self(img)
        loss = F.cross_entropy(y_hat.squeeze(), y)
        acc = self._get_accuracy(y_hat, y)
        return loss, acc

    def _get_accuracy(self, y_hat, y):
        pred = y_hat.max(1)[1]
        correct = pred.eq(y).cpu().sum()
        return correct / len(y)

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return cls(**vars(args), **kwargs)

    def configure_optimizers(self):
        if self.hparams.model == 'mlp':
            optim = eval(self.hparams.optim)(
                self.parameters(), lr=self.hparams.learning_rate)
        else:
            optim = eval(self.hparams.optim)(self.parameters(), lr=self.hparams.learning_rate,
                                             weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=True)
        scheduler = eval(self.hparams.scheduler)(
            optim, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        return {"optimizer": optim, "lr_scheduler": scheduler}
