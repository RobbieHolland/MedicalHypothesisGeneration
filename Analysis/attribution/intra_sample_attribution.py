import torch
import hydra
from Data.get_data import get_data
from Analysis.run.linear_evaluation import LinearEvaluation
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
import os
import random
from Analysis.run.linear_evaluation import ClassifierModel
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import os
from torch.cuda.amp import autocast
from tqdm import tqdm
from itertools import product
import glob

torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

class AttributionVisualizer:
    def __init__(self, config, model, device, threshold, folder_name='feature_attribution'):
        self.config = config
        self.model = model.eval().to(device)
        self.device = device
        self.model = model
        self.tokenizer = self.model.inference_map['findings'].model.model.encode_text.tokenizer

        analysis_dir = os.path.join(config.base_dir, 'Analysis/output')
        self.output_dir = os.path.join(analysis_dir, folder_name, config.task.outputs[0], f">={threshold}")
        print(f"Saving to ------------> {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_occlusion_slices(self, index, shape, occlusion_size):
        """Returns slices for occlusion, starting at 'index' and extending forward."""
        return (slice(None),) + tuple(
            slice(i, min(i + occlusion_size, s))  # Start at index, extend occlusion_size
            for i, s in zip(index, shape[1:])
        )
    
    def occlude(self, data, occlusion_size, step_size):
        """Generic occlusion function for both 1D (text) and 3D (image) inputs."""
        shape = data.shape
        occlusions = []
        
        # Generate step indices for all occludable dimensions
        step_ranges = [range(0, s, step_size) for s in shape[1:]]  # Skip batch dim

        for index in product(*step_ranges):  # Cartesian product to iterate correctly
            occluded = data.clone()
            
            # Build slices dynamically
            slices = self._get_occlusion_slices(index, shape, occlusion_size)

            occluded[slices] = 0 if data.ndimension() > 2 else self.tokenizer.pad_token_id
            occlusions.append((occluded, index))

        return occlusions

    def run_occlusion(self, inputs, outputs, sample_ix):
        sample_output_dir = os.path.join(self.output_dir, str(sample_ix))
        os.makedirs(sample_output_dir, exist_ok=True)
        
        inputs['image'] = torch.Tensor(inputs['image']).unsqueeze(0).to(self.device)
        encoded = self.tokenizer(inputs['findings'].lower(), return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        occlusion_size_image = 48  # Can be adapted dynamically
        occlusion_step_image = 24
        occlusion_size_text = 4
        occlussion_step_text = 2
        
        image_occlusions = self.occlude(inputs['image'], occlusion_size_image, step_size=occlusion_step_image)
        text_occlusions = self.occlude(encoded["input_ids"], occlusion_size_text, step_size=occlussion_step_text)
        
        image_attributions = torch.zeros_like(inputs['image'])
        text_attributions = torch.zeros_like(encoded['input_ids'].float())
        
        prediction_class = int(outputs[self.config.task.outputs[0]])
        original_output = self.model(inputs)['prediction'][:,prediction_class].detach().cpu()
        print(outputs, original_output)
        
        for occluded_image, index in image_occlusions:
            inputs['image'] = occluded_image.to(self.device)
            output = self.model(inputs)['prediction'][:,prediction_class].detach().cpu()
            diff = (original_output - output).sum()

            slices = self._get_occlusion_slices(index, inputs['image'].shape, occlusion_size_image)
            image_attributions[slices] += diff
        
        for occluded_tokens, index in text_occlusions:
            encoded['input_ids'] = occluded_tokens
            inputs['findings'] = self.tokenizer.decode(occluded_tokens[0])
            output = self.model(inputs)['prediction'][:,prediction_class].detach().cpu()
            diff = (original_output - output).square().sum()

            slices = self._get_occlusion_slices(index, encoded['input_ids'].shape, occlusion_size_text)
            text_attributions[slices] += diff

        image_attributions = image_attributions.detach().cpu()
        text_attributions = text_attributions.detach().cpu()
        
        inputs["image"] = inputs["image"].cpu()
        self.visualize_attribution(
            inputs, sample_ix, self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
            image_attributions, text_attributions, sample_output_dir
        )

    def run_input_x_gradient(self, inputs, outputs, sample_ix):
        sample_output_dir = os.path.join(self.output_dir, str(sample_ix))
        os.makedirs(sample_output_dir, exist_ok=True)

        inputs['image'] = torch.Tensor(inputs['image']).unsqueeze(0).to(self.device)

        with autocast(dtype=torch.float16):
            activations, gradients = {}, {}

            def save_activation(name):
                def hook(module, inp, outp):
                    activations[name] = outp
                return hook

            def save_gradient(name):
                def hook(module, grad_inp, grad_outp):
                    gradients[name] = grad_outp[0]
                return hook

            embedding_layer = self.model.inference_map['findings'].model.model.encode_text.text_encoder.embeddings.word_embeddings
            embedding_handle = embedding_layer.register_forward_hook(save_activation("embedding"))
            embedding_grad_handle = embedding_layer.register_backward_hook(save_gradient("embedding"))

            inputs["image"].requires_grad = True
            image_hook_handle = inputs["image"].register_hook(lambda grad: gradients.setdefault("image", grad))

            output = self.model(inputs)['prediction']
            criterion = nn.BCEWithLogitsLoss()
            target = torch.Tensor([[1 - outputs[self.config.task.outputs[0]], outputs[self.config.task.outputs[0]]]]).to(output.device)
            loss = criterion(output, target)
            print(output, target)
            loss.backward()

            encoded = self.tokenizer(inputs['findings'].lower(), return_tensors="pt", padding=True, truncation=True)
            tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

            input_x_grad_embedding = activations["embedding"] * gradients["embedding"]
            input_x_grad_embedding = input_x_grad_embedding[:, :len(tokens)]
            input_x_grad_image = inputs["image"] * gradients["image"]

            image_attributions = input_x_grad_image.detach().cpu()
            findings_attributions = input_x_grad_embedding.detach().sum(2).cpu()

            embedding_handle.remove()
            embedding_grad_handle.remove()
            image_hook_handle.remove()

        inputs["image"] = inputs["image"].cpu()
        self.visualize_attribution(
            inputs, sample_ix, tokens,
            image_attributions, findings_attributions, sample_output_dir
        )

    def visualize_attribution(self, inputs, sample_ix, tokens, image_attributions, findings_attributions, output_dir):
        import html, base64, io
        from PIL import Image

        title = f"Patient {sample_ix} findings importance for {self.config.task.outputs[0]}"

        attr_vals = findings_attributions[0].cpu().numpy()
        max_abs = np.abs(attr_vals).max() or 1
        alpha_min, alpha_max = 0.0, 0.8

        def styled_token(token, val):
            token_text = html.escape(token.replace("Ġ", ""))
            s = val / max_abs
            alpha = alpha_min + (alpha_max - alpha_min) * abs(s)
            color = f"rgba({'255, 0, 0' if s < 0 else '0, 255, 0'}, {alpha:.2f})"
            return f"<span style='background-color:{color}; color:black;'>{token_text}</span>"

        html_tokens = [styled_token(t, v) for t, v in zip(tokens, attr_vals)]
        main_body = " ".join(html_tokens)

        indices = np.arange(len(attr_vals))
        top5_pos = sorted(indices[attr_vals > 0], key=lambda i: attr_vals[i], reverse=True)[:10]
        top5_neg = sorted(indices[attr_vals < 0], key=lambda i: attr_vals[i])[:10]

        top_pos_list = "<ol>" + "".join(f"<li>{styled_token(tokens[i], attr_vals[i])}</li>" for i in top5_pos) + "</ol>"
        top_neg_list = "<ol>" + "".join(f"<li>{styled_token(tokens[i], attr_vals[i])}</li>" for i in top5_neg) + "</ol>"

        text_html_content = main_body + (
            "<hr><h2>Top 10 Positive Firing Tokens</h2>" + top_pos_list +
            "<h2>Top 10 Negative Firing Tokens</h2>" + top_neg_list
        )

        ct_volume = inputs["image"].detach().squeeze().numpy()  # shape: (D, H, W)
        ct_attributions = image_attributions.squeeze().numpy()

        # slice_importance = np.abs(ct_attributions).sum((0, 1))
        # top4_indices = np.argsort(slice_importance)[-4:]
        # top4_indices = np.argsort(ct_attributions.max(axis=(0, 1)))[::-1][:4]  # Sort in descending order

        tol = 1e-8
        max_attributions = ct_attributions.max(axis=(0, 1))
        max_value = max_attributions.max()
        close_to_max = np.where(max_attributions >= max_value - tol)[0]

        top4_indices = (close_to_max[np.linspace(0, len(close_to_max) - 1, 4, dtype=int)]
                        if len(close_to_max) > 4 
                        else np.argsort(max_attributions)[::-1][:4])

        def generate_overlay_image(ct_slice, attrib_slice, alpha_min=0.0, alpha_max=0.7):
            ct_min, ct_max = ct_slice.min(), ct_slice.max()
            if ct_max - ct_min == 0:
                ct_norm = np.zeros_like(ct_slice, dtype=np.uint8)
            else:
                ct_norm = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
            base_img = Image.fromarray(ct_norm, mode='L').convert("RGBA")

            max_abs_attrib = np.abs(attrib_slice).max() or 1.0
            norm_attrib = attrib_slice / max_abs_attrib
            alpha = alpha_min + (alpha_max - alpha_min) * np.abs(norm_attrib)
            alpha_scaled = (alpha * 255).astype(np.uint8)

            H, W = ct_slice.shape
            overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)

            # Red is negative attribution
            overlay_arr[..., 0] = np.where(norm_attrib < 0, 255, 0)
            overlay_arr[..., 1] = np.where(norm_attrib >= 0, 255, 0)
            overlay_arr[..., 3] = alpha_scaled
            overlay_img = Image.fromarray(overlay_arr, mode='RGBA')

            composite_img = Image.alpha_composite(base_img, overlay_img)
            buffer = io.BytesIO()
            composite_img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        images_html = ""
        for idx in top4_indices:
            b64_img = generate_overlay_image(ct_volume[:, :, idx], ct_attributions[:, :, idx])
            images_html += f'<img src="data:image/png;base64,{b64_img}" style="width:100%; margin-bottom:10px;" />\n'

        html_text = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .container {{
                    display: flex;
                    gap: 20px;
                }}
                .attribution {{
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ccc;
                    overflow-y: auto;
                    max-height: 800px;
                }}
                .viewer {{
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ccc;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                }}
                .viewer img {{
                    width: 100%;
                    height: auto;
                    display: block;
                }}
            </style>
            </head>
            <body>
            <h1>{title}</h1>
            <div class="container">
                <div class="attribution">
                {text_html_content}
                </div>
                <div class="viewer">
                {images_html}
                </div>
            </div>
            </body>
            </html>
            """
        with open(os.path.join(output_dir, "colored_text.html"), "w") as f:
            f.write(html_text)

        print(f'Visualization written to {output_dir}')


@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    datasets, dataloaders = get_data(config, device=device, splits=['validation'])
    
    from Model.get_model import ModelBuilder
    model = ModelBuilder(config).get_model()
    model = model.eval()

    keys_to_remove = [k for k in model.inference_map.keys() if model.inference_metadata[k]['compress']]

    for k in keys_to_remove:
        del model.inference_map[k]
        del model.inference_metadata[k]  # Ensure metadata is also removed

    classifier_module = ClassifierModel(config)
    for param in classifier_module.parameters():
        param.requires_grad = True

    classifier_module = classifier_module.to(device)

    model.update_inference_map('multimodal_embedding', classifier_module, 'prediction', False)

    # Load the checkpoint's state dict
    ckpt_files = glob.glob(os.path.join(config.pretrained_model_dir, config.task.predictor_path, 
                                        config.task.outputs[0], '**/*.ckpt'), recursive=True)
    latest_linear_ckpt = max(ckpt_files, key=os.path.getctime) if ckpt_files else None
    print(f'Checkpoint files, taking latest of them: {ckpt_files} which was:\n{latest_linear_ckpt}')

    ckpt_state = torch.load(latest_linear_ckpt, weights_only=False)["state_dict"]

    # Assume checkpoint_state is the loaded state_dict from your checkpoint
    new_state_dict = {}
    for k, v in ckpt_state.items():
        # Remove the "linearevaluation." prefix so that the keys match your model.
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v

    # Then load the modified state_dict into your model
    model.load_state_dict(new_state_dict, strict=False)

    target = datasets['validation'].dataset.dataset[config.task.outputs[0]]
    qs = 10
    k = 10
    n_unique = target.nunique()
    if n_unique < qs:
        bins = np.linspace(target.min(), target.max(), n_unique + 1)
        quantiles = pd.cut(target, bins=bins, labels=False, include_lowest=True)
    else:
        quantiles = pd.qcut(target, q=qs, labels=False, duplicates='drop')

    thresholds = [1, 0]
    for threshold in thresholds:
        sample_ixs = datasets['validation'].dataset.dataset.loc[quantiles == threshold].sample(k).index

        visualizer = AttributionVisualizer(config, model, device, threshold)
        for sample_ix in tqdm(sample_ixs):
            inputs, outputs = datasets['validation'].__getitem__(sample_ix)
            visualizer.run_occlusion(inputs, outputs, sample_ix)
            # visualizer.run_input_x_gradient(inputs, outputs, sample_ix)

if __name__ == "__main__":
    main()
