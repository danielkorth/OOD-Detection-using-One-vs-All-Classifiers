# to calculate metrics for the different methods

import torch
import pandas as pd
import numpy as np

from utils.metric_utils import get_metrics
from utils.aggregation import aggregate_predictions
from utils import mahalanobis


def get_measures_from_predict(trainer, model, in_dm, out_dm, in_measures, out_measures):
    in_outputs_OVAF = trainer.predict(model, in_dm)
    ood_outputs_OVAF = trainer.predict(model, out_dm)
    in_measures = aggregate_predictions(in_outputs_OVAF, in_measures)
    out_measures = aggregate_predictions(ood_outputs_OVAF, out_measures)

    return in_measures, out_measures


def get_msp(trainer, model, in_dm, out_dm, recall_level):
    in_, out_ = get_measures_from_predict(trainer, model, in_dm, out_dm, [
                                          'y', 'y_hat', 'prob_max'], ['prob_max'])
    auroc, aupr, fpr = get_metrics(in_[2], out_, recall_level=recall_level)
    acc = in_[0].eq(in_[1]).sum() / len(in_[0])

    return np.array([acc, auroc, aupr, fpr])


def get_energy_score(trainer, model, in_dm, out_dm, recall_level, temperature=1):
    in_, out_ = get_measures_from_predict(trainer, model, in_dm, out_dm, [
                                          'y', 'y_hat', 'logit_all'], ['logit_all'])
    in_energies = - \
        (temperature *
         torch.logsumexp(in_[2] / temperature, dim=1)).cpu().numpy()
    out_energies = -(temperature * torch.logsumexp(out_ /
                     temperature, dim=1)).cpu().numpy()

    auroc, aupr, fpr = get_metrics(-in_energies, -
                                   out_energies, recall_level=recall_level)

    acc = in_[0].eq(in_[1]).sum() / len(in_[0])

    return np.array([acc, auroc, aupr, fpr])


def get_mahalanobis(trainer, model, in_dm, out_dm, recall_level, noise, data):
    if data == 'mnist':
        temp_x = torch.rand(2, 1, 28, 28)
    else:
        temp_x = torch.rand(2, 3, 32, 32)

    if data == 'cifar100':
        n_classes = 100
    else:
        n_classes = 10

    model = model.cuda()
    temp_x = temp_x.cuda()
    temp_list = model.feature_list(temp_x)[1]

    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    sample_mean, precision = mahalanobis.sample_estimator(
        model, n_classes, feature_list, in_dm.train_dataloader())
    in_score = mahalanobis.get_Mahalanobis_score(model, in_dm.test_dataloader(
    ), n_classes, sample_mean, precision, count-1, noise, data)
    out_score = mahalanobis.get_Mahalanobis_score(model, out_dm.test_dataloader(
    ), n_classes, sample_mean, precision, count-1, noise, data)

    auroc, aupr, fpr = get_metrics(-in_score, -
                                   out_score, recall_level=recall_level)

    return np.array([-1, auroc, aupr, fpr])


def get_ovnni(trainer, model, in_dm, out_dm, recall_level):
    """
    :model: should be the OVNNI Model with correct predict_step logic
    """
    in_, out_ = get_measures_from_predict(trainer, model, in_dm, out_dm, [
                                          'y', 'y_hat', 'prob_max'], ['prob_max'])
    auroc, aupr, fpr = get_metrics(in_[2], out_, recall_level=recall_level)
    acc = in_[0].eq(in_[1]).sum() / len(in_[0])

    return np.array([acc, auroc, aupr, fpr])


def get_OVAF(trainer, model, in_dm, out_dm, recall_level=0.95, debug=False, tnr=False):
    """
    Calculate the metrics for each head individually and take the average
    """
    in_, out_ = get_measures_from_predict(trainer, model, in_dm, out_dm, [
                                          'y', 'y_hat', 'logit'], ['y_hat', 'logit'])
    print(in_[1])

    aurocs, auprs, fprs, threshs = [], [], [], []

    for i in range(len(model.heads)):

        # get the logits for each corresponding head
        filter_in = (in_[1] == i)
        filter_out = (out_[0] == i)

        # obtain the logits for those heads
        logit_in = in_[2][filter_in]
        logit_out = out_[1][filter_out]

        #
        pos_ = logit_in
        neg_ = logit_out

        if len(neg_) == 0:
            auroc, aupr, fpr = 1, 1, 0
        elif debug:
            auroc, aupr, fpr, thresh = get_metrics(
                pos_, neg_, recall_level=recall_level, get_conf_thresh=True)
            threshs.append(thresh)
            # TODO add that information to a pandas thing or something else
        else:
            auroc, aupr, fpr = get_metrics(
                pos_, neg_, recall_level=recall_level)

        fprs.append(fpr)
        aurocs.append(auroc)
        auprs.append(aupr)

    if debug:
        return aurocs, auprs, fprs, threshs
    else:
        auroc, aupr, fpr = np.mean(aurocs), np.mean(auprs), np.mean(fprs)
        return np.array([-1, auroc, aupr, fpr])


def get_everything(trainer, in_dm, out_dm, base, heads=None, heads_w_ood=None, ovnni=None, energy=None, vos=None, recall_level=0.95, print_results=True,
                   energy_t=1,
                   mahalanobis_noise=0.0001,
                   data='mnist',
                   ):
    base = base.cuda()

    methods, values = [], []
    metrics = ['ACC', 'AUROC', 'AUPR', f'FPR@{int(recall_level*100)}']

    # MSP
    methods.append('MSP')
    values.append(get_msp(trainer, base, in_dm, out_dm, recall_level))

    # Energy Score - no finetune
    methods.append('Energy')
    values.append(get_energy_score(trainer, base, in_dm,
                  out_dm, recall_level, temperature=energy_t))

    # Mahalanobis Distance
    methods.append('Mahalanobis')
    values.append(get_mahalanobis(trainer, base, in_dm, out_dm,
                  recall_level, noise=mahalanobis_noise, data=data))

    # OVNNI
    if ovnni != None:
        methods.append('OVNNI')
        values.append(get_ovnni(trainer, ovnni, in_dm, out_dm, recall_level))

    # VOS
    if vos != None:
        methods.append('VOS')
        values.append(get_energy_score(
            trainer, vos, in_dm, out_dm, recall_level))

    # OVAF
    if heads != None:
        methods.append('OVAF')
        values.append(get_OVAF(trainer, heads, in_dm, out_dm, recall_level))

    # finetuned Energy Score
    if energy != None:
        methods.append('Energy (finetune)')
        values.append(get_energy_score(trainer, energy, in_dm,
                      out_dm, recall_level, temperature=energy_t))

    # OVAF with OOD data during training
    if heads_w_ood != None:
        methods.append('OVAF outlier tuning')
        values.append(get_OVAF(trainer, heads_w_ood,
                      in_dm, out_dm, recall_level))

    print(
        f'Amount of data used for calculation:\nIn-Distribution: {len(in_dm.test_dataset)}\nOut-of-Distribution: {len(out_dm.test_dataset)}')
    results = pd.DataFrame(values, index=methods, columns=metrics)
    results = results.apply(lambda x: round(x*100, 3))

    if print_results:
        print(results)

    return results
