import math
import sys

import torch
import torch.nn as nn
import torchvision
import copy

# from contrastive_3d.models.inflated_convnets_pytorch.src import inflate
# from contrastive_3d.utils import window_level

# torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

class Mock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.linear = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.latent(x)

    def latent(self, x):
        if isinstance(x, dict):
            x = x['image']
            
        return x.view((x.shape[0], x.shape[1], -1))[:,:,:2048].squeeze(1)

def load_mock(config):
    return Mock(config)

# import hydra
# @hydra.main(config_path="../config", config_name="default", version_base=None)
# def main(config):
#     # from Data.get_data import get_data
#     # datasets, dataloaders = get_data(config)

#     device = torch.device('cuda:0')

#     x = torch.randn(2, 1, 224, 224, 224)
#     x = x.to(device)  # Input tensor
#     model = load_merlin_image_to_image(config)
#     model = model.to(device)

#     with torch.inference_mode():
#         y = model(x)
#         output = model.latent(x)
#     print(output.shape)

# if __name__ == "__main__":
#     main()