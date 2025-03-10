import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from Model.concept_model import ConceptModel

class IdentityModel(ConceptModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        z = [torch.Tensor(l).unsqueeze(0) if l.ndim == 1 else l for l in x]
        z = torch.cat(z, axis=1)
        return z
    
    def latent(self, x):
        return self(x)
