import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class OVNNIWrapper(pl.LightningModule):
    """
    Implementation based on the paper:
    One Versus All for Deep Neural Network for Uncertainty (OVNNI) Quantification
    https://arxiv.org/abs/2006.00954
    """

    def __init__(
        self,
        ava,
        ovas,
    ):
        super().__init__()

        self.ava = ava

        for ova in ovas:
            ova.to('cuda')
        self.ovas = ovas

    def forward(self, x):
        logits_ava = self.ava(x)
        probs_ava = F.softmax(logits_ava, dim=1)
        logits = [ova(x) for ova in self.ovas]
        # This brings it into a "softmax-like" format (i.e. [num_samples, class_logits])
        logits = torch.stack(logits).squeeze().T
        probs_ova = F.sigmoid(logits)
        return probs_ava * probs_ova

    def test_step(self, batch, batch_idx):
        img, y = batch
        y_hats = self(img)
        acc = self.__get_accuracy(y_hats, y)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return {'acc': acc}

    def predict_step(self, batch, batch_idx):
        img, y = batch
        prob_all = self(img)
        prob, y_hat = prob_all.max(1)
        return {'y': y, 'y_hat': y_hat, 'prob_max': prob, 'prob_all': prob_all}

    def __get_accuracy(self, y_hats, y):
        y_hat = y_hats.max(1)[1]
        correct = y_hat.eq(y).cpu().sum()

        size_all = len(y)
        return correct/size_all
