import torch
import hydra
from Model.get_model import get_model
from Data.get_data import get_data
from Analysis.run.linear_evaluation import LinearEvaluation
from types import MethodType
from torch.cuda.amp import autocast

class IntegratedGradients:
    def __init__(self, model, baseline=None, steps=50, device=None):
        """
        Initializes the IG explainer.
        
        Args:
            model (torch.nn.Module): The PyTorch model.
            baseline (torch.Tensor or None): The reference input. Defaults to zeros if None.
            steps (int): Number of interpolation steps.
            device (str or None): Device to run computations on (auto-detect if None).
        """
        self.model = model.to(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.baseline = baseline  # If None, it will be set dynamically.

    def _generate_interpolated_inputs(self, x):
        """Creates interpolated inputs between the baseline and actual input."""
        baseline = self.baseline if self.baseline is not None else torch.zeros_like(x)
        alphas = torch.linspace(0, 1, self.steps).view(-1, *[1] * (x.dim() - 1)).to(self.device)
        return baseline + alphas * (x - baseline)  # Shape: (steps, *x.shape)

    def _compute_gradients(self, x, output_index):
        """Computes gradients for each interpolated input."""
        x.requires_grad_(True)
        outputs = self.model(x)[:, output_index]  # Extract output neuron of interest
        grads = torch.autograd.grad(outputs.sum(), x)[0]  # Compute gradients
        return grads  # Shape: (steps, *x.shape)

    def explain(self, x, output_index):
        """
        Computes Integrated Gradients attributions for the given input.

        Args:
            x (torch.Tensor): The input tensor (batch supported).
            output_index (int): Index of the output neuron to explain.

        Returns:
            torch.Tensor: Attribution map of the same shape as `x`.
        """
        x = x.to(self.device)
        interpolated_inputs = self._generate_interpolated_inputs(x)
        grads = self._compute_gradients(interpolated_inputs, output_index)
        avg_grads = grads.mean(dim=0)  # Average gradients over the interpolation steps
        attributions = (x - self.baseline) * avg_grads if self.baseline is not None else x * avg_grads
        return attributions

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = get_model(config)

    datasets, dataloaders = get_data(config, specific_data='abdominal_ct', splits=['validation'])

    sample = datasets['validation'].__getitem__(0)
    x = (sample['image'].unsqueeze(0).to(device), [sample['findings']])
    model = get_model(config)
    model = model.eval()
    model = model.to(device)

    with autocast(dtype=torch.float16):
        linear_model_path = '/dataNAS/people/rholland/MedicalHypothesisGeneration/pretrained_models/MedicalHypothesisGeneration-Analysis_run/LinearEvaluation/mu057prc/abdominal_ct_text_embeddings/ihd/clean-sweep-4/epoch=223-step=4704.ckpt'
        linear_model = LinearEvaluation.load_from_checkpoint(linear_model_path)
        linear_model = linear_model.to(device)

        def full_forward(self, x):
            z = self.multimodal_forward(x)
            y = linear_model(z.squeeze(1).unsqueeze(0))
            return y

        model.forward = MethodType(full_forward, model)

        y = model.forward(x)

        # Define the attribution method
        from captum.attr import InputXGradient

        input_x_gradient = InputXGradient(model)

        # Target class index
        target_class = torch.tensor([0])

        # Compute attributions
        attributions = input_x_gradient.attribute(x, target_class)

        # Detach to avoid computation graph issues
        attributions = attributions.detach()

        # # Initialize IG and compute attributions for output neuron 0
        # ig = IntegratedGradients(model, steps=100)
        # attributions = ig.explain(x, output_index=0)

        # print("Attributions:", attributions)

# Example Usage:
if __name__ == "__main__":
    main()