# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/densenet.py

import torch
import torch.nn as nn
from models.abstractmodel import AbstractModel


"""dense net in pytorch
[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""
# """Bottleneck layers. Although each layer only produces k
# output feature-maps, it typically has many more inputs. It
# has been noted in [37, 11] that a 1×1 convolution can be in-
# troduced as bottleneck layer before each 3×3 convolution
# to reduce the number of input feature-maps, and thus to
# improve computational efficiency."""


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # """In  our experiments, we let each 1×1 convolution
        # produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        # """We find this design especially effective for DenseNet and
        # we refer to our network with such a bottleneck layer, i.e.,
        # to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        # as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                in_channels,
                inner_channel,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                inner_channel,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False))

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

# """We refer to layers between blocks as transition
# layers, which do convolution and pooling."""


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # """The transition layers used in our experiments
        # consist of a batch normalization layer and an 1×1
        # convolutional layer followed by a 2×2 average pooling
        # layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

# DesneNet-BC
# B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
# C stands for compression factor(0<=theta<=1)


class DenseNet(AbstractModel):
    def __init__(
            self,
            block,
            nblocks,
            growth_rate=12,
            reduction=0.5,
            num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate

        # """Before entering the first dense block, a convolution
        # with 16 (or twice the growth rate for DenseNet-BC)
        # output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        # For convolutional layers with kernel size 3×3, each
        # side of the inputs is zero-padded by one pixel to keep
        # the feature-map size fixed.
        self.conv1 = nn.Conv2d(
            3,
            inner_channels,
            kernel_size=3,
            padding=1,
            bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module(
                "dense_block_layer_{}".format(index),
                self._make_dense_layers(
                    block,
                    inner_channels,
                    nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            # """If a dense block contains m feature-maps, we let the
            # following transition layer generate θm output feature-
            # maps, where 0 < θ ≤ 1 is referred to as the compression
            # fac-tor.
            # int() will automatic floor the value
            out_channels = int(reduction * inner_channels)
            self.features.add_module(
                "transition_layer_{}".format(index), Transition(
                    inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(
            nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('activation', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def feature_list(self, x):
        out_list = []
        x = self.conv1(x)
        out_list.append(x)
        x = self.features.transition_layer_0(
            self.features.dense_block_layer_0(x))
        out_list.append(x)
        x = self.features.transition_layer_1(
            self.features.dense_block_layer_1(x))
        out_list.append(x)
        x = self.features.transition_layer_2(
            self.features.dense_block_layer_2(x))
        out_list.append(x)
        x = self.avgpool(self.features.dense_block3(x))
        out_list.append(x)
        x = x.view(x.size()[0], -1)
        return self.linear(x), out_list

    def intermediate_forward(self, x, layer_index):
        x = self.conv1(x)
        if layer_index == 0:
            return x
        x = self.features.transition_layer_0(
            self.features.dense_block_layer_0(x))
        if layer_index == 1:
            return x
        x = self.features.transition_layer_1(
            self.features.dense_block_layer_1(x))
        if layer_index == 2:
            return x
        x = self.features.transition_layer_2(
            self.features.dense_block_layer_2(x))
        if layer_index == 3:
            return x
        x = self.avgpool(self.features.dense_block3(x))
        return x

    def penultimate_forward(self, x):
        x = self.conv1(x)
        x = self.features.transition_layer_0(
            self.features.dense_block_layer_0(x))
        x = self.features.transition_layer_1(
            self.features.dense_block_layer_1(x))
        x = self.features.transition_layer_2(
            self.features.dense_block_layer_2(x))
        x = self.avgpool(self.features.dense_block3(x))
        x = x.view(x.size()[0], -1)
        return x

    def linear_forward(self, x):
        return self.linear(x)

    def forward_virtual(self, x):
        virt = self.penultimate_forward(x)
        pred = self.linear(virt)
        return pred, virt

    def get_feature_size(self):
        return self.linear.in_features

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module(
                'bottle_neck_layer_{}'.format(index), block(
                    in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


def densenet121(num_classes=10):
    return DenseNet(Bottleneck, [6, 12, 24, 16],
                    growth_rate=32, num_classes=num_classes)

# def densenet169(activation = 'relu'):
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

# def densenet201(activation = 'relu'):
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

# def densenet161(activation = 'relu'):
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
