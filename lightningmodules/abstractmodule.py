from abc import ABC, abstractmethod
import torch.nn as nn
import pytorch_lightning as pl


class AbstractModule(pl.LightningModule, ABC):
    '''
    Abstract Module that defines functionality required and used for all specific models/modules
    '''

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, batch_idx):
        _, acc = self._share_step(batch)
        self.log('test_acc', acc, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        img, y = batch
        logit_all = self(img)
        logit_max, y_hat = logit_all.max(1)
        prob_all = nn.Softmax(dim=1)(logit_all)
        prob_max = prob_all.max(1)[0]
        return {'y': y, 'y_hat': y_hat, 'prob_max': prob_max, 'prob_all': prob_all, 'logit_max': logit_max, 'logit_all': logit_all}

    def _share_step(self, batch):
        img, y = batch
        y_hat = self(img)
        loss = self.criterion(y_hat.squeeze(), y)
        acc = self._get_accuracy(y_hat, y)
        return loss, acc

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return cls(**vars(args), **kwargs)

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _get_accuracy(self, y_hat, y):
        pass

    # Functions used for Mahalanobis, VOS, etc.
    @abstractmethod
    def penultimate_forward(self, x):
        '''
        obtain the penultimate layer activations
        '''
        pass

    @abstractmethod
    def feature_list(self, x):
        '''
        obtain the layer activations of all layers
        '''
        pass

    @abstractmethod
    def intermediate_forward(self, x, layer_index):
        '''
        obtain the layer activation from layer_index
        '''
        pass

    @abstractmethod
    def configure_optimizers(self):
        '''
        optimizer stuff
        '''
        pass
