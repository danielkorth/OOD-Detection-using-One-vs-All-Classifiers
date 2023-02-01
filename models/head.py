import torch.nn as nn


class Head(nn.Module):
    def __init__(self,
                 feature_size,
                 **kwargs
                 ):

        super(Head, self).__init__()

        self.head = nn.Sequential()
        for out in feature_size:
            self.head.append(nn.Dropout(p=0.3))
            self.head.append(nn.LazyLinear(out_features=out))
            self.head.append(nn.ReLU())

        self.head.append(nn.LazyLinear(out_features=1))

    def forward(self, x):
        return self.head(x)
