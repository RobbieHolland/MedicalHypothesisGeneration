import os
from Model.model import load_model
from Model.identity import IdentityModel
import torch

def get_multimodal_merlin(config, compress=True):
    device = torch.device('cuda:0')
    compression_model = load_model(config)
    compression_model = compression_model.to(device)

    from Analysis.run.linear_evaluation import LinearEvaluation
    linear_model_path = '/dataNAS/people/rholland/MedicalHypothesisGeneration/pretrained_models/MedicalHypothesisGeneration-Analysis_run/LinearEvaluation/t3qho6x5/abdominal_ct_text_embeddings/ost/dark-sweep-32/epoch=23-step=504.ckpt'
    linear_model = LinearEvaluation.load_from_checkpoint(linear_model_path)

    # Define inference map
    inference_map = {
        'image': {'output_field': 'merlin/image', 'compress': compress, 'forward_model': lambda x: compression_model.latent(torch.stack(x))},
        'findings': {'output_field': 'merlin/findings', 'compress': compress, 'forward_model': lambda x: compression_model.model.encode_text(x)},
        ('merlin/image', 'merlin/findings'): {'output_field': 'ost_prediction', 'compress': False, 'forward_model': lambda x: linear_model(x)}
    }

    from Model.multimodal_model import MultimodalModel
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
