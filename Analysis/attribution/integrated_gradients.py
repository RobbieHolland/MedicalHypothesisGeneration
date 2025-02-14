import torch
import hydra
from Model.get_model import get_model
from Data.get_data import get_data
from Analysis.run.linear_evaluation import LinearEvaluation
from types import MethodType
from torch.cuda.amp import autocast

torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

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
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    datasets, dataloaders = get_data(config, splits=['validation'])

    inputs, outputs = datasets['validation'].__getitem__(0)

    compression_model = get_model(config, specific_model_name='merlin')
    compression_model = compression_model.to(device)
    tokenizer = compression_model.model.encode_text.tokenizer
    
    model = get_model(config)
    model.inference_map = datasets['validation'].inference_map
    model = model.eval()
    model = model.to(device)

    import numpy as np
    inputs['image'] = torch.Tensor(inputs['image']).to(device)
    input_ids = tokenizer(inputs['findings'].lower(), return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
    token_embeddings = compression_model.model.encode_text.text_encoder.embeddings.word_embeddings(input_ids)

    with autocast(dtype=torch.float16):
        # outputs = model.forward(inputs)
        # y = outputs['ost_prediction']

        # Move tokenizer output to the correct device
        input_ids = tokenizer(inputs['findings'].lower(), return_tensors="pt", padding=True, truncation=True)["input_ids"]
        input_ids = input_ids.to(device)  # Move input_ids to the model's device

        # Generate token embeddings and enable gradients
        token_embeddings = compression_model.model.encode_text.text_encoder.embeddings.word_embeddings(input_ids)
        token_embeddings = token_embeddings.clone().detach().requires_grad_(True)

        # Ensure image input is on the correct device and dtype
        image_input = inputs["image"].to(device, dtype=torch.float32).requires_grad_(True)

        # Forward pass
        inputs_dict = {
            "image": image_input.unsqueeze(1),
            "findings": token_embeddings
        }
        output = model(inputs_dict)['ost_prediction']

        # # Compute gradients w.r.t. inputs
        # target_class = 0  # Adjust if needed
        # loss = output[:, target_class].sum()
        # loss.backward()

        # # Get gradients
        # image_gradients = image_input.grad
        # text_gradients = token_embeddings.grad

        # Define the attribution method
        from captum.attr import InputXGradient

        # Define forward function compatible with Captum
        def forward_func(image_tensor, token_embeddings):
            # Prepare input dict for model
            inputs_dict = {
                "image": image_tensor.requires_grad_(True),
                "findings": token_embeddings.requires_grad_(True)  # Now using extracted token embeddings
            }

            outputs = model(inputs_dict)  # Forward pass
            return outputs['ost_prediction']  # Extract 'ost_prediction'

        # Wrap model in attribution method
        input_x_gradient = InputXGradient(forward_func)

        # Target class index
        target_class = torch.tensor([0])

        # Compute attributions
        attributions = input_x_gradient.attribute((inputs_dict['image'], inputs_dict['findings']), target=target_class)

        # Detach attributions
        image_attributions = attributions[0].detach()
        findings_attributions = attributions[1].detach()
        findings_attributions = findings_attributions.sum(2)

        print(image_attributions.shape)
        print(findings_attributions.shape)

        # 1) Convert IDs back to actual tokens
        encoded = tokenizer(inputs['findings'].lower(), return_tensors="pt", padding=True, truncation=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        import html

        attr_vals = findings_attributions[0].cpu().numpy()

        max_abs = max(abs(attr_vals.min()), abs(attr_vals.max()), 1e-9)

        alpha_min = 0.2
        alpha_max = 0.8

        html_tokens = []
        for token, val in zip(tokens, attr_vals):
            # Replace "Ġ" with a space for readability
            token_text = token.replace("Ġ", "")

            # Escape any HTML-like characters to avoid strikethrough or other unwanted rendering
            token_text = html.escape(token_text)

            scaled = max(-1, min(1, val / max_abs))
            alpha = alpha_min + (alpha_max - alpha_min) * abs(scaled)

            if scaled < 0:
                color = f"rgba(255, 0, 0, {alpha:.2f})"  # red
            else:
                color = f"rgba(0, 255, 0, {alpha:.2f})"   # green

            html_tokens.append(f"<span style='background-color:{color}; color:black;'>{token_text}</span>")

        html_text = " ".join(html_tokens)
        with open("colored_text.html", "w") as f:
            f.write(html_text)

        x = 3

        # # Initialize IG and compute attributions for output neuron 0
        # ig = IntegratedGradients(model, steps=100)
        # attributions = ig.explain(x, output_index=0)

        # print("Attributions:", attributions)

# Example Usage:
if __name__ == "__main__":
    main()