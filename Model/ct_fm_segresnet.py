# Imports
import torch
from lighter_zoo import SegResNet
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground, Invert,
    Activations, AsDiscrete, KeepLargestConnectedComponent,
    SaveImage, Spacing, DivisiblePad, Lambda, ResizeWithPadOrCrop
)
from monai.inferers import SlidingWindowInferer
from Model.concept_model import ConceptModel
import os

class CT_FM_SegResNet(ConceptModel):
    def __init__(self, config):
        super().__init__()

        model_dir = os.path.join(config.pretrained_model_dir, 'pretrained_models')
        os.makedirs(model_dir, exist_ok=True)
        self.model = SegResNet.from_pretrained(
            "project-lighter/ct_fm_segresnet",
            cache_dir=model_dir,
        )

        self.MAX_SIZE = (400, 512, 512)

        # Define conditional resize function
        def conditional_resize(image):
            if any(s > m for s, m in zip(image.shape[-3:], self.MAX_SIZE)):  # Check if any dimension is too large
                return ResizeWithPadOrCrop(spatial_size=self.MAX_SIZE)(image)
            return image  # Return unchanged if already within MAX_SIZE

        self.preprocess = Compose([
            LoadImage(ensure_channel_first=True),
            EnsureType(),
            Orientation(axcodes="SPL"),
            ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
            CropForeground(),
            Spacing(pixdim=(3.0, 1.0, 1.0), mode="bilinear"),
            DivisiblePad(k=16),
            Lambda(conditional_resize),  # Only resizes if needed
        ])

    def forward(self, x):
        return self.latent(x)

    def latent(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)

        # x = torch.stack(x)

        zs = []
        for image in x:
            print(image.shape)
            if not self.model.is_valid_shape(image):
                raise ValueError(f"Input spatial dims {image.shape} must be divisible by {self.model.shape_factor()}")

            image = torch.Tensor(image.to(self.model.encoder.conv_init.weight.device)).unsqueeze(1)
            output = self.model.encoder(image)
            z = output[-1].mean(dim=(-3, -2, -1))
            zs.append(z)

        zs = torch.stack(zs).squeeze(1)
        return zs

def load_ct_fm(config):
    model = CT_FM_SegResNet(config)
    return model

import hydra
@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(config):
    from Data.get_data import get_data
    datasets, dataloaders = get_data(config)

    test_image = datasets['validation'].dataset.dataset['image'].iloc[0]
    test_image2 = datasets['validation'].dataset.dataset['image'].iloc[1]

    x = [test_image2]
    device = torch.device('cuda:0')
    model = CT_FM_SegResNet(config).to(device)  # Adjust truncate_at as needed
    
    with torch.inference_mode():
        output = model.latent(x)
    print(output.shape)

if __name__ == "__main__":
    main()