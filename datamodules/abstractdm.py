from abc import ABC, abstractmethod
import pytorch_lightning as pl


class AbstractDataModule(pl.LightningDataModule, ABC):

    def __init__(
            self,
            train_transform=None,
            test_transform=None,
            root='/u/home/korth/ood-detection-using-one-vs-all-classifiers/',
            seed=42,
            batch_size=256,
            dataloader_num_workers=4,
            **kwargs):

        super().__init__()

        self.save_hyperparameters()

        self.prepare_data()
        self.setup()

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DataModule')
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--dataloader_num_workers', default=4, type=int)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return cls(**vars(args), **kwargs)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self._setup()

    @abstractmethod
    def _setup(self, positive_class=-1):
        pass

    def make_binary(self, positive_class):
        """
        Takes a class of the dataset and labels it as positive (1). 
        All other classes in the dataset will be negatives(0)
        """
        return self._setup(positive_class)

    def reset_binary(self):
        """
        Assigns original classes again after the classes have beeen made binary.
        """
        return self._setup()

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def val_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        return self.test_dataloader()
