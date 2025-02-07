import torch.nn as nn
from Model.trainable_save import TrainableSave

class ConceptModel(TrainableSave):
    def __init__(self, config):
        super().__init__(config)

    def latent(self, x):
        """
        Must return the latent representation of x.
        Every derived model should override this method.
        """
        raise NotImplementedError("Subclasses must implement 'latent' method.")
