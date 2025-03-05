import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from Model.concept_model import ConceptModel

# Image Encoder Model
class ImageEncoder(ConceptModel):
    def __init__(self, compression_model):
        super().__init__()
        self.model = compression_model

    def forward(self, image_list):
        return self.model.latent(torch.cat(image_list, dim=0))

# Text Encoder Model
class TextEncoder(ConceptModel):
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

class MerlinWrapper(ConceptModel):
    def __init__(self, config):
        super().__init__()
        from Model import clip_model_3d
        self.model = clip_model_3d.Clip3D(
            config={
                "architecture": "i3_resnet_clinical_longformer",
                "text_encoder": "clinical_longformer",
                "use_ehr": True,
            }
        )
        checkpoint = torch.load("/dataNAS/people/akkumar/contrastive-3d/pretrained_models/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt")
        model_state_dict = self.model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
        self.model.load_state_dict(filtered_checkpoint, strict=False)

    # def forward(self, x):
    #     return self.model.encode_image(x)

    def forward(self, x):
        return self.model.encode_image.i3_resnet(x)[1]  # Normal inference

    def multimodal_forward(self, x):
        image, text = x

        z_text = self.model.encode_text(text)
        z_image = self.latent(image)

        z = torch.cat((z_image, z_text), dim=1)
        return z

    def latent(self, x):
        if isinstance(x, dict):
            x = x['image']
            
        feature_maps = []

        def hook(module, _, output):
            feature_maps.append(output)

        # hook_handle = self.model.encode_image.i3_resnet.avgpool.register_forward_hook(hook)
        hook_handle = self.model.encode_image.i3_resnet.avgpool.register_forward_hook(hook)

        # Forward pass through the model
        output = self.model.encode_image(x)
        
        # Remove the hook after the forward pass to avoid redundancy
        hook_handle.remove()
        
        return feature_maps[0].reshape(feature_maps[0].size(0), -1)

# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('/dataNAS/people/akkumar/contrastive-3d')

    x = torch.randn(2, 1, 224, 224, 224)  # Input tensor
    model = MerlinWrapper(n_classes=1692, truncate_at=2)  # Adjust truncate_at as needed
    output = model.latent(x)
    print(output.shape)
