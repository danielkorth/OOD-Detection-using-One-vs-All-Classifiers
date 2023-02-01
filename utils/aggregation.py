import torch


def aggregate_predictions(outputs, labels):
    """
    aggregates all predictions from predict_step() and returns the aggregated tensor
    """
    concatenated = [[] for _ in range(len(labels))]
    for output in outputs:
        for idx, label in enumerate(labels):
            concatenated[idx].append(output[label])
    concatenated = [torch.cat(conc) for conc in concatenated]
    if len(concatenated) == 1:
        return concatenated[0]
    return concatenated
