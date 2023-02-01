import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class FilteringHeadModel(pl.LightningModule):
    '''
    Base model that adds the essential functionality to models regardless of specific model architecture
    '''

    def __init__(
        self,
        feature_size='[256, 128]',
        learning_rate=1e-3,
        optim='torch.optim.Adam',
        dropout=0.1,
        scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
        t_max=20,
        eta_min=1e-6,
        weight=None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

        self._build_model()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('FilteringHeadModel')
        parser.add_argument('--feature_size', default='[256, 128]', type=str)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--optim', default='torch.optim.Adam', type=str)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument(
            '--scheduler', default='torch.optim.lr_scheduler.CosineAnnealingLR', type=str)
        parser.add_argument('--t_max', default=20, type=int)
        parser.add_argument('--eta_min', default=1e-6, type=float)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return cls(**vars(args), **kwargs)

    def _build_model(self):
        self.head = nn.Sequential()
        for out in eval(self.hparams.feature_size):
            self.head.append(nn.Dropout(p=self.hparams.dropout))
            self.head.append(nn.LazyLinear(out_features=out))
            self.head.append(nn.ReLU())

        self.head.append(nn.LazyLinear(out_features=1))

    def forward(self, x):
        self.head.to(x.device)
        x = self.head(x.squeeze()).squeeze()
        return x

    def penultimate_forward(self, x):
        self.head.to(x.device)
        x = self.head[0:5](x)
        return x

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

    def test_step(self, batch, batch_idx):
        pass

    def _share_step(self, batch):
        img, y = batch
        y_hat = self(img)
        loss = self.criterion(y_hat.squeeze(), y.float())
        acc = self._get_accuracy(y_hat, y)
        return loss, acc

    def _get_accuracy(self, y_hat, y):
        pred = torch.sigmoid(y_hat)
        pred = pred.round().squeeze()
        correct = pred.eq(y).cpu().sum()
        size_all = len(y)
        return correct/size_all

    def configure_optimizers(self):
        optim = eval(self.hparams.optim)(
            self.parameters(), lr=self.hparams.learning_rate)
        scheduler = eval(self.hparams.scheduler)(
            optim, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        return {"optimizer": optim, "lr_scheduler": scheduler}


class FilteringHeadWrapper(pl.LightningModule):
    """
    the og
    """

    def __init__(
        self,
        backbone,
        heads,
        thresholds=-1
    ):
        super().__init__()

        backbone.eval()
        self.backbone = backbone

        for head in heads:
            head = head.to(backbone.device)
            head.eval()
        self.heads = heads

    def forward(self, x):
        features = self.backbone.penultimate_forward(x)
        logits = [model(features).squeeze() for model in self.heads]
        # This brings it into a "softmax-like" format (i.e. [num_samples, class_logits])
        logits = torch.stack(logits).T
        return logits

    def test_step(self, batch, batch_idx):
        img, y = batch
        y_hats = self(img)
        acc = self.__get_accuracy(y_hats, y)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return {'acc': acc}

    def predict_step(self, batch, batch_idx):
        img, y = batch
        logit_backbone = self.backbone(img)
        logit_max, y_hat = logit_backbone.max(1)
        logit_all = self(img)
        # gather the logits only from the head where softmax had maximum probability
        logit = logit_all.gather(dim=1, index=y_hat.reshape(-1, 1))
        logit = logit.squeeze()
        return {'y': y, 'y_hat': y_hat, 'logit': logit, 'backbone_logit_max': logit_max}

    def penultimate_forward(self, batch):
        img, y = batch
        features = self.backbone.penultimate_forward(img.cuda())
        features = [model.penultimate_forward(
            features) for model in self.heads]
        # This brings it into a "softmax-like" format (i.e. [num_samples, class_logits])
        features = torch.stack(features)
        return features

    def penultimate_forward_2(self, batch):
        img, y = batch
        logit_backbone = self.backbone(img.cuda())
        _, y_hat = logit_backbone.max(1)
        features = self.backbone.penultimate_forward(img.cuda())
        features = [model.penultimate_forward(
            features) for model in self.heads]
        features = torch.stack(features)
        return features, y_hat

    def __get_accuracy(self, y_hats, y):
        y_hat = y_hats.max(1)[1]
        correct = y_hat.eq(y).cpu().sum()

        size_all = len(y)
        return correct/size_all
