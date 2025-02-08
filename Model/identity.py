import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from Model.concept_model import ConceptModel

class IdentityModel(ConceptModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, x):
        z = torch.stack([x[l] for l in self.config.data.inputs], axis=1)
        return z

    def latent(self, x):
        return x

