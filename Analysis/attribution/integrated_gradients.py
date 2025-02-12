import torch

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

# Example Usage:
if __name__ == "__main__":
    # Sample model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),  # Input: 10, Output: 5
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)  # Output: 1 neuron
    )

    # Random input
    x = torch.rand((1, 10))  # Single batch, 10 features

    # Initialize IG and compute attributions for output neuron 0
    ig = IntegratedGradients(model, steps=100)
    attributions = ig.explain(x, output_index=0)

    print("Attributions:", attributions)
