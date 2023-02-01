import torch
import math


def sample_negatives(features, num_negatives, num_classes, current_class):
    """
    sample negative classes from the in-distribution dataset and return it
    """
    num_negatives_per_class = math.ceil(num_negatives / (num_classes-1))
    for j in range(num_classes):
        if current_class != j:
            if j == 0 or (j == 1 and current_class == 0):
                out_data = features[j][:num_negatives_per_class]
            else:
                out_data = torch.cat(
                    (out_data, features[j][:num_negatives_per_class]))

    out_data = out_data.cpu()
    # can happen that we sample too many, therefore randomly take values from here
    indices = torch.randperm(len(out_data))[:num_negatives]
    out_data = out_data[indices]
    return out_data


def get_all_negatives(features, num_classes, current_class):
    """
    get all negative class instances from in-distribution dataset and return it
    """
    for j in range(num_classes):
        if current_class != j:
            if j == 0 or (j == 1 and current_class == 0):
                out_data = features[j]
            else:
                out_data = torch.cat((out_data, features[j]))
    out_data = out_data.cpu()
    return out_data
