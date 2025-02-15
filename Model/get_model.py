import os
from Model.model import load_model
from Model.identity import IdentityModel
import torch
import torch
import torch.nn as nn
from Model.multimodal_model import MultimodalModel
from Analysis.run.linear_evaluation import LinearEvaluation

# Image Encoder Model
class ImageEncoder(nn.Module):
    def __init__(self, compression_model):
        super().__init__()
        self.model = compression_model

    def forward(self, image_list):
        return self.model.latent(torch.cat(image_list, dim=0))

# Text Encoder Model
class TextEncoder(nn.Module):
    def __init__(self, compression_model):
        super().__init__()
        self.model = compression_model

    def forward(self, text_embeddings):
        return self.model.model.encode_text(text_embeddings)

# OST Model for Prediction
class OSTModel(nn.Module):
    def __init__(self, linear_model):
        super().__init__()
        self.model = linear_model

    def forward(self, inputs):
        return self.model(inputs)

def get_multimodal_merlin(config, compress=True):
    device = torch.device('cuda:0')

    # Load Compression Model
    compression_model = load_model(config).to(device)

    # Load Linear Model
    # linear_model_path = '/dataNAS/people/rholland/MedicalHypothesisGeneration/pretrained_models/MedicalHypothesisGeneration-Analysis_run/LinearEvaluation/t3qho6x5/abdominal_ct_text_embeddings/ost/dark-sweep-32/epoch=23-step=504.ckpt'
    # linear_model = LinearEvaluation.load_from_checkpoint(linear_model_path).to(device)

    # Define inference map with proper nn.Modules
    inference_map = {
        # 'image': {
        #     'output_field': 'merlin/image',
        #     'compress': compress,
        #     'forward_model': ImageEncoder(compression_model)
        # },
        'image': {
            'output_field': 'merlin/image',
            'compress': compress,
            'forward_model': ImageEncoder(compression_model)
        },
        'findings': {
            'output_field': 'merlin/findings',
            'compress': compress,
            'forward_model': TextEncoder(compression_model)
        },
        ('merlin/image', 'merlin/findings'): {
            'output_field': 'multimodal_embedding',
            'compress': False,
            'forward_model': IdentityModel(config),
        },
        # 'merlin/image': {
        #     'output_field': 'multimodal_embedding',
        #     'compress': False,
        #     'forward_model': IdentityModel(config),
        # },
        # ('multimodal_embedding'): {
        #     'output_field': 'ost_prediction',
        #     'compress': False,
        #     'forward_model': OSTModel(linear_model)
        # }
    }

    # inference_map = {
    #     'phecodes': {
    #         'output_field': 'multimodal_embedding',
    #         'compress': False,
    #         'forward_model': IdentityModel(config),
    #     },
    # }

    # Instantiate Multimodal Model
    model = MultimodalModel(config, inference_map)
    
    return model

def get_model(config, specific_model_name=None, device=None):
    model_name = config.model_name if not specific_model_name else specific_model_name

    if model_name == 'merlin':
        model = load_model(config)
        
    elif model_name == 'merlin_sae':
        model = _load_merlin_sae(config)

    elif model_name == 'identity':
        return IdentityModel(config)
    
    elif model_name == 'multimodal_merlin':
        model = get_multimodal_merlin(config, compress=False)

    elif model_name == 'compressed_multimodal_merlin':
        model = get_multimodal_merlin(config, compress=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if device:
        model = model.to(device)

    return model

def _load_merlin_sae(config):
    """Loads the Merlin model and integrates it with the SAE component."""
    encoder = load_model(config)
    encoder.eval()  # Ensures proper BatchNorm and Dropout behavior

    # Load SAE checkpoint
    checkpoint_path = os.path.join(config.pretrained_model_dir, config.task.sae_checkpoint)
    
    from Analysis.run.fit_sae import TrainSparseAutoencoder
    sae = TrainSparseAutoencoder.load_from_checkpoint(checkpoint_path, strict=False).sae

    # Combine encoder with SAE
    from Analysis.model.combined_model import CombinedModel
    model = CombinedModel(encoder, sae).eval()
    
    return model
