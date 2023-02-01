# adapted from https://github.com/pokaxpoka/deep_Mahalanobis_detector

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis
from tqdm import tqdm
import math


def get_feature_list(model, x):
    """
    get a list of the dimensions of all layers in the model
    :return: the feature_list
    """
    feature_list = list()
    _, features = model.feature_list(x)
    for feature in features:
        size = feature.squeeze().shape[0]
        feature_list.append(size)
    return feature_list


def get_ood_features(model, loader):
    """
    compute the embeddings of penultimate layer and store its features
    :return: list of features separated by class
    """
    model.eval()
    list_features = list()

    # Collect the features
    for data, _ in tqdm(loader, ascii=True, desc='get_ood_features'):
        data = data.to(model.device)
        with torch.no_grad():
            out_features = model.penultimate_forward(data)
        if len(list_features) == 0:
            list_features = out_features.squeeze().cpu()
        else:
            list_features = torch.cat(
                (list_features, out_features.squeeze().cpu()), 0)

    torch.cuda.empty_cache()

    return list_features


def get_mean_cov_features(model, num_classes, loader, only_correct=False):
    """
    *only_correct*: whether we only add features to the list that were correctly classified or not
    compute the mean and covariance of the penultimate layer (last embedding layer) and also store the features
    :return: mean, cov and features
    """
    model.eval()
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)

    list_features = [0 for _ in range(num_classes)]
    num_features = 0

    # only correct stuff
    if only_correct:
        num_sample_per_other_class = np.empty(num_classes)
        num_sample_per_other_class.fill(0)
        list_other_features = [0 for _ in range(num_classes)]

    # Collect the features
    for data, target in tqdm(loader, ascii=True, desc='get_mean_cov_features'):
        data = data.to(model.device)
        with torch.no_grad():
            out_features = model.penultimate_forward(data)
            if only_correct:
                pred = model(data)
                pred = pred.max(1)[1]
        num_features = out_features.shape[1]

        for i in range(data.size(0)):
            label = target[i]

            # correct stuff - experimental
            if only_correct and pred[i] != target[i]:
                if num_sample_per_other_class[label] == 0:
                    list_other_features[label] = out_features[i].view(
                        1, -1).cpu()
                else:
                    list_other_features[label] = torch.cat(
                        (list_other_features[label], out_features[i].view(1, -1).cpu()), 0)
                num_sample_per_other_class[label] += 1
            else:
                if num_sample_per_class[label] == 0:
                    list_features[label] = out_features[i].view(1, -1).cpu()
                else:
                    list_features[label] = torch.cat(
                        (list_features[label], out_features[i].view(1, -1).cpu()), 0)
                num_sample_per_class[label] += 1

    # Calculate mean and covariance
    class_mean = torch.Tensor(num_classes, num_features)
    class_covariance = torch.Tensor(num_classes, num_features, num_features)
    for i in range(num_classes):
        class_mean[i] = torch.mean(list_features[i], 0)
        class_covariance[i] = torch.cov(
            list_features[i].reshape(num_features, -1))

    torch.cuda.empty_cache()

    return class_mean, class_covariance, list_features, num_sample_per_class


def synthesize_features(num_classes, num_sample_per_class, sample_mean, cov, sample_pct, sample_from, in_dist=False):
    """
    synthesize outliers from a specific feature space
    refer to https://github.com/deeplearning-wisc/vos for more details
    :return: list of synthesized outliers
    """
    # Create MultivariateNormalUtil(s)
    print(f'The number of samples per class: {num_sample_per_class}')
    multivariates = list()
    for mean, covv in zip(sample_mean, cov):
        multivariates.append(MultivariateNormalUtil(
            mean, covariance_matrix=covv))

    ood_samples = list()

    for idx, n_samples in tqdm(enumerate(num_sample_per_class), ascii=True, desc='synthesize_features'):

        # Because of memory constraints, we need to loop and create outlier multiple times as it cannot hold enough generated samples in memory at once
        xxx = 20
        temp = list()
        for _ in range(xxx):
            negative_samples = multivariates[idx].rsample(
                (int(n_samples * sample_pct * sample_from / xxx), ))
            prob_density = multivariates[idx].log_prob(negative_samples)
            _, index_prob = torch.topk(prob_density, math.ceil(
                n_samples * sample_pct / xxx), largest=in_dist)

            # temp.append(negative_samples[index_prob])

            if len(temp) == 0:
                temp = negative_samples[index_prob]
            else:
                temp = torch.cat((temp, negative_samples[index_prob]), 0)

            del negative_samples
            del index_prob
            torch.cuda.empty_cache()

        # get the exact number needed
        prob_density = multivariates[idx].log_prob(temp)
        _, index_prob = torch.topk(prob_density, int(
            n_samples * sample_pct), largest=in_dist)

        ood_samples.append(temp[index_prob])

    return ood_samples


class MultivariateNormalUtil(MultivariateNormal):
    """
    Extended version of MultivariateNormal that allows direct computation of the mahalanobis distance
    """
    
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        # for stable training
        covariance_matrix += 0.0001 * \
            torch.eye(len(covariance_matrix), device=covariance_matrix.device)

        super().__init__(loc, covariance_matrix=covariance_matrix)

    def mahalanobis(self, value):
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        return M
