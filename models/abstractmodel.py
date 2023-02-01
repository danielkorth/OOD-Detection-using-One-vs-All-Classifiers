from abc import ABC, abstractmethod
import torch.nn as nn


class AbstractModel(nn.Module, ABC):
    """
    The abstract model structure each model should have
    """

    def __init__(self, **kwargs):
        super(AbstractModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        regular forward pass of the model
        """
        pass

    @abstractmethod
    def feature_list(self, x):
        """
        utility function that allows to extract feature sizes
        """
        pass

    @abstractmethod
    def intermediate_forward(self, x, layer_index):
        """
        function that allows an intermediate forward until layer_index
        """
        pass

    @abstractmethod
    def penultimate_forward(self, x):
        """
        forward until the last layer before liner layer
        """
        pass

    @abstractmethod
    def linear_forward(self, x):
        """
        last layer of the model; should work together with penultimate_forward to create a full forward pass
        """
        pass

    @abstractmethod
    def get_feature_size(self):
        """
        return the penultimate layers feature size
        """
        pass
