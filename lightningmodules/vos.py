# implementation for VOS: Learning What You Don't Know by Virtual Outlier Synthesis
# https://arxiv.org/abs/2202.01197
# original implementation: https://github.com/deeplearning-wisc/vos/blob/25eec0dd1af06bcabccc4b20e8f338bf5de909bb/classification/CIFAR/train_virtual.py#L134

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.densenet import densenet121
from models.wideresnet import wideresnet
from models.mlp import MLP


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


class VOSModule(pl.LightningModule):

    def __init__(self,
                 model='densenet',
                 num_classes=10,
                 learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=5e-4,
                 optim='torch.optim.SGD',
                 scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
                 t_max=200,
                 eta_min=1e-6,
                 nesterov=True,
                 start_epoch=40,
                 sample_number=1000,
                 sample_from=10000,
                 num_select=1,
                 loss_weight=0.1,
                 **kwargs
                 ):
        super().__init__()

        self.save_hyperparameters()

        self._build_model()
        self._setup_queue()

        self.eye_matrix = torch.eye(self.feature_size).cuda()

        self.curr_epoch = 0

        # self.automatic_optimization = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('VOS')
        parser.add_argument('--model', default='densenet', type=str)
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--learning_rate', default=0.1, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--optim', default='torch.optim.SGD', type=str)
        parser.add_argument(
            '--scheduler', default='torch.optim.lr_scheduler.CosineAnnealingLR', type=str)
        parser.add_argument('--t_max', default=100, type=int)
        parser.add_argument('--eta_min', default=1e-6, type=float)
        parser.add_argument('--nesterov', default=True, type=bool)
        parser.add_argument('--start_epoch', default=40, type=int)
        parser.add_argument('--sample_number', default=1000, type=int)
        parser.add_argument('--sample_from', default=10000, type=int)
        parser.add_argument('--num_select', default=1, type=int)
        parser.add_argument('--loss_weight', default=0.1, type=float)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return cls(**vars(args), **kwargs)

    def _build_model(self):
        if self.hparams.model == 'mlp':
            self.model = MLP()
        elif self.hparams.model == 'densenet':
            self.model = densenet121(num_classes=self.hparams.num_classes)
        else:
            self.model = wideresnet(num_classes=self.hparams.num_classes)

        self.feature_size = self.model.get_feature_size()

        self.weight_energy = torch.nn.Linear(self.hparams.num_classes, 1)
        torch.nn.init.uniform_(self.weight_energy.weight)

        self.logistic_regression = torch.nn.Linear(1, 2)

    def _setup_queue(self):
        # keeps track of how many samples we have put forward already
        self.number_dict = {}
        for i in range(self.hparams.num_classes):
            self.number_dict[i] = 0

        # stores the data in the queue
        self.data_dict = torch.zeros(
            self.hparams.num_classes, self.hparams.sample_number, self.feature_size).cuda()

    def training_step(self, batch, batch_idx):
        data, target = batch

        features = self.model.penultimate_forward(data)
        x = self.model.linear_forward(features)

        # Check how many samples have already been added into the queue
        sum_temp = 0
        for index in range(self.hparams.num_classes):
            sum_temp += self.number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]

        # Enqueue features into the queue
        if sum_temp == self.hparams.num_classes * self.hparams.sample_number and self.curr_epoch < self.hparams.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                self.data_dict[dict_key] = torch.cat(
                    (self.data_dict[dict_key][1:], features[index].detach().view(1, -1)), 0)

        # Create Virtual outliers and train logistic regression head
        elif sum_temp == self.hparams.num_classes * self.hparams.sample_number and self.curr_epoch >= self.hparams.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                      features[index].detach().view(1, -1)), 0)
            # center the data
            for index in range(self.hparams.num_classes):
                if index == 0:
                    X = self.data_dict[index] - self.data_dict[index].mean(0)
                    mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat(
                        (X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               self.data_dict[index].mean(0).view(1, -1)), 0)

            # add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            # for stable training
            temp_precision += 0.0001 * self.eye_matrix

            for index in range(self.hparams.num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((self.hparams.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(
                    - prob_density, self.hparams.num_select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat(
                        (ood_samples, negative_samples[index_prob]), 0)
            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = self.log_sum_exp(x, 1)
                predictions_ood = self.model.linear_forward(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = self.log_sum_exp(predictions_ood, 1)

                input_for_lr = torch.cat(
                    (energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(features)).cuda(),
                                           torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = self.logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                self.log('lr_loss', lr_reg_loss)

                # if self.curr_epoch % 5 == 0:
                #     print(lr_reg_loss)

        # Fill queue until it is full
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if self.number_dict[dict_key] < self.hparams.sample_number:
                    self.data_dict[dict_key][self.number_dict[dict_key]
                                             ] = features[index].detach()
                    self.number_dict[dict_key] += 1

        # opt = self.optimizers()
        # sch = self.lr_schedulers()

        # opt.zero_grad()
        loss = F.cross_entropy(x, target)
        loss += self.hparams.loss_weight * lr_reg_loss
        # self.manual_backward(loss)

        # opt.step()
        # sch.step()

        acc = self._get_accuracy(x, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return {'loss': loss, 'acc': acc}

    def on_train_epoch_end(self):
        self.curr_epoch += 1

    def forward(self, x):
        return self.model(x)

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

        energy_score = self.log_sum_exp(logit_all, 1)
        uncertainty = self.logistic_regression(energy_score.view(-1, 1))

        logit_max, y_hat = logit_all.max(1)
        prob_all = F.softmax(logit_all, dim=1)
        prob_max = prob_all.max(1)[0]
        return {'y': y, 'y_hat': y_hat, 'prob_max': prob_max, 'prob_all': prob_all, 'logit_max': logit_max, 'logit_all': logit_all, 'vos_uncertainty': uncertainty}

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

    def configure_optimizers(self):
        parameters = list(self.model.parameters(
        )) + list(self.weight_energy.parameters()) + list(self.logistic_regression.parameters())
        if self.hparams.model == 'mlp':
            optim = eval(self.hparams.optim)(
                parameters, lr=self.hparams.learning_rate)
        else:
            optim = eval(self.hparams.optim)(parameters, lr=self.hparams.learning_rate,
                                             weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        scheduler = eval(self.hparams.scheduler)(
            optim, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step:cosine_annealing(step, self.hparams.max_epochs * self.hparams.len_loader, 1, 1e-6 / self.hparams.learning_rate))
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)
