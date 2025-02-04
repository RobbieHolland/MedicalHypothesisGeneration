import torch
import hydra
import torch.utils.checkpoint
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from MultimodalPretraining.data.raw_database.dataset import create_dataloaders
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import math

# Override checkpoint function globally to disable it
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

import torch

def gradcam(model, data, effect, target, target_layers):
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    sign = math.copysign(1, effect)
    class ValentTarget(ClassifierOutputTarget):
        def __call__(self, model_output):
            return sign * super().__call__(model_output)  # Inverts gradient influence

    # Generate GradCAM heatmap
    grayscale_cam = cam(input_tensor=data, targets=[ValentTarget(target)])
    attribution_map = grayscale_cam[0]  # Remove batch dimension

    return attribution_map
