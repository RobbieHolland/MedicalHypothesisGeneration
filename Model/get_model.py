import os
from Model.model import load_model
from Model.identity import IdentityModel

def get_model(config, specific_model_name=None, device=None):
    model_name = config.model_name if not specific_model_name else specific_model_name

    if model_name == 'merlin':
        model = load_model(config)
    elif model_name == 'merlin_sae':
        model = _load_merlin_sae(config)
    elif model_name == 'identity':
        return IdentityModel(config)
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
