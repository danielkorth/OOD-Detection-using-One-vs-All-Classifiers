import torch.nn as nn


class MLP(nn.Module):
    """
    A simple MLP for MNIST Classification
    """

    def __init__(self, binary=False):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1 if binary else 10)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.reshape((-1, 784))
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

    def feature_list(self, x):
        x = x.reshape((-1, 784))
        out_list = []
        out = self.activation(self.fc1(x))
        out_list.append(out)
        out = self.activation(self.fc2(out))
        out_list.append(out)
        out = self.fc3(out)
        return out, out_list

    def intermediate_forward(self, x, layer_index):
        x = x.reshape((-1, 784))
        out = self.activation(self.fc1(x))
        if layer_index == 1:
            out = self.activation(self.fc2(out))
        return out

    def penultimate_forward(self, x):
        x = x.reshape((-1, 784))
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

    def forward_virtual(self, x):
        virt = self.penultimate_forward(x)
        pred = self.fc3(virt)
        return pred, virt

    def linear_forward(self, x):
        return self.fc3(x)

    def get_feature_size(self):
        return self.fc3.in_features
