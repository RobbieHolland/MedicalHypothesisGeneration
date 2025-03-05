import torch.nn as nn
from Model.trainable_save import TrainableSave

class ConceptModel(TrainableSave):
    def __init__(self):
        super().__init__()

    def latent(self, x):
        """
        Must return the latent representation of x.
        Every derived model should override this method.
        """
        raise NotImplementedError("Subclasses must implement 'latent' method.")

    def configure(self, dataset):
        pass