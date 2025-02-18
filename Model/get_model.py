import os
from Model.model import load_model
from Model.identity import IdentityModel
from Model.merlin_wrapper import ImageEncoder, TextEncoder
import torch
import torch
import torch.nn as nn
from Model.multimodal_model import MultimodalModel
from Analysis.run.linear_evaluation import LinearEvaluation

class ModelBuilder():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0')
        self.compression_model = load_model(config).to(self.device)

    def get_multimodal_merlin(self):
        # Load Linear Model
        # linear_model_path = '/dataNAS/people/rholland/MedicalHypothesisGeneration/pretrained_models/MedicalHypothesisGeneration-Analysis_run/LinearEvaluation/t3qho6x5/abdominal_ct_text_embeddings/ost/dark-sweep-32/epoch=23-step=504.ckpt'
        # linear_model = LinearEvaluation.load_from_checkpoint(linear_model_path).to(device)

        # Define inference map with proper nn.Modules
        inference_map = {}
        for input_field, forward_configuration in self.config.data.inference_map.items():
            inference_map[input_field] = {
                'forward_model': self.get_model(forward_configuration.forward_model),
                'compress': forward_configuration.compress,
                'output_field': forward_configuration.output_field,
            }

        # Instantiate Multimodal Model
        model = MultimodalModel(self.config, inference_map)
        model = model.to(self.device)
        
        return model

    def get_model(self, specific_model_name=None, device=None):
        model_name = self.config.model_name if not specific_model_name else specific_model_name

        if model_name == 'merlin':
            model = load_model(self.config)

        elif model_name == 'merlin_image_encoder':
            return ImageEncoder(self.compression_model)
            
        elif model_name == 'merlin_text_encoder':
            return TextEncoder(self.compression_model)

        elif model_name == 'merlin_sae':
            model = self._load_merlin_sae(self.config)

        elif model_name == 'identity':
            return IdentityModel(self.config)
        
        elif model_name == 'multimodal_merlin':
            model = self.get_multimodal_merlin()

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
