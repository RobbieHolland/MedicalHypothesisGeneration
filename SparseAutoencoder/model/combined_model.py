import torch.nn as nn
import torch
import torch.nn.functional as F

class CombinedModel(nn.Module):
    def __init__(self, encoder, sae):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.sae = sae

    def concept_forward(self, x):
        latent_enc = self.encoder.latent(x)  # Encoder latent representation
        latent_sae = self.sae.latent(latent_enc)  # SAE latent representation
        return latent_sae

    def supervised_forward(self, x):
        prediction = self.encoder(x)  # Encoder latent representation
        return prediction

    def clip_forward(self, x, phecode):
        texts = [f'No {phecode}', f'There is {phecode}']
        z_image = self.encoder.model.encode_image(x)[0]

        zs_text = self.encoder.model.encode_text(texts)

        similarity = F.cosine_similarity(z_image.unsqueeze(1), zs_text.unsqueeze(0), dim=-1)

        # Get the prediction (index of maximum similarity)
        # prediction = torch.argmax(similarity, dim=-1)
        return similarity