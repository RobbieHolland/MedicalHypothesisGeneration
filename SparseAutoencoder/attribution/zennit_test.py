import torch
from torchvision.models import vgg16_bn
import hydra
import torch.utils.checkpoint
from MultimodalPretraining.data.raw_database.dataset import create_dataloaders

from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat, EpsilonPlus
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm

# Override checkpoint function globally to disable it
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    # Load dataset
    datasets, dataloaders = create_dataloaders(config, sets=['test'])
    batch = next(iter(dataloaders['test']))
    
    # Prepare input data
    data = batch['image'][0].unsqueeze(0).to(torch.float32)
    data.requires_grad = True  # Ensure gradients are tracked
    
    from MultimodalPretraining.model.model import load_model

    # Load model and set evaluation mode
    model = load_model(config)
    model.eval()  # Ensures proper BatchNorm and Dropout behavior

    # Ensure model outputs are finite before attribution
    with torch.no_grad():
        test_output = model(data)
        test_output = torch.nan_to_num(test_output, nan=0.0, posinf=1e4, neginf=-1e4)
        print("Test output (no NaNs):", test_output.isnan().any().item())

    # Select composite method
    # composite = EpsilonGammaBox()  # Using a stable LRP variant
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonGammaBox(low=-300., high=300., canonizers=canonizers)
    # composite = EpsilonGammaBox(low=-3., high=3.)

    # Attribution
    with Gradient(model=model, composite=composite) as attributor:
        out, relevance = attributor(data, torch.eye(1692)[[1517]][0])

    # Normalize relevance to prevent NaNs
    print(relevance[~relevance.isnan()])
    print((~relevance.isnan()).sum())

    # Debug: Print if relevance still has NaNs
    print("Relevance has NaNs:", relevance.isnan().any().item())

if __name__ == "__main__":
    main()
