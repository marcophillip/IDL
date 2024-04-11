
from lightning.pytorch import Trainer
import os

class CustomTrainer(Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        if self.is_global_zero:
            dirpath = os.path.split(filepath)[0]
            lightningmodel = self.model.model
            lightningmodel.save_pretrained(dirpath)