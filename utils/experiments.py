import os
import glob
import re

class Experiment:
    """
    Convenience Class for easy model loading of guild runs
    """
    
    def __init__(
        self,
        id=str,
        root='/u/home/korth',
        logdir='ood-detection-using-one-vs-all-classifiers/env3.8/.guild/runs'
    ):
        self.id = id
        self.root = root
        self.logdir = logdir

    def load_model(self, id, model_name):
        self.model = Model(**self.hparams)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()

    def get_all_checkpoints(self, id, ftype='ckpt'):
        checkpoints = os.path.join(
            self.root, self.logdir, id, 'model_checkpoints')

        checkpoints = [file for file in glob.glob(
            os.path.join(checkpoints, f'**/*.{ftype}'), recursive=True)]
        
        if len(checkpoints) == 1:
            return checkpoints[0]
        # to sort the ova model in the right order
        checkpoints = sorted(
            checkpoints, key=lambda x: int(re.findall(r"\d+", x)[-1]))
        return checkpoints
