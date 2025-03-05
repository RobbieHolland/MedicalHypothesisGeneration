import torch
import os
from Model.model import load_model
from Model.identity import IdentityModel
from Model.merlin_wrapper import ImageEncoder, TextEncoder
from Model.multimodal_model import MultimodalModel
from Model.ct_fm_segresnet import load_ct_fm
from Model.merlin_image_to_image import load_merlin_image_to_image
from Model.mock_model import load_mock
from Model.tabular import TabularProcessor

class ModelBuilder():
    def __init__(self, config, device=None):
        self.config = config
        self.compression_model = load_model(config)
        self.device = device
        
        if device:
            self.compression_model = self.compression_model.to(self.device)

    def get_configured_model(self):
        # Define inference map with proper nn.Modules
        inference_map = {}
        for input_field, forward_configuration in self.config.model.inference_map.items():
            inference_map[input_field] = {
                'forward_model_name': forward_configuration.forward_model_name,
                'forward_model': self.get_model(forward_configuration.forward_model_name).to(self.device),
                'compress': forward_configuration.compress,
                'output_field': forward_configuration.output_field,
            }

        # Instantiate Multimodal Model
        model = MultimodalModel(self.config, inference_map)
        model = model.to(self.device)
        
        return model

    def get_model(self, specific_model_name=None):
        model_name = self.config.model_name if not specific_model_name else specific_model_name

        if model_name == 'ct_fm_image_encoder':
            model = load_ct_fm(self.config)

        elif model_name == 'merlin':
            model = load_model(self.config)

        elif model_name == 'tabular_processor':
            model = TabularProcessor(self.config)

        elif model_name == 'merlin_image_encoder':
            return ImageEncoder(self.compression_model)
            
        elif model_name == 'merlin_image_to_image_ssl_encoder':
            model = load_merlin_image_to_image(self.config)

        elif model_name == 'merlin_text_encoder':
            return TextEncoder(self.compression_model)

        elif model_name == 'merlin_sae':
            model = self._load_merlin_sae(self.config)

        elif model_name == 'identity':
            return IdentityModel(self.config)
        
        elif model_name == 'configured_model':
            model = self.get_configured_model()

        elif model_name == 'mock':
            model = load_mock(self.config)

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if self.device:
            model = model.to(self.device)

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
