import torch
import hydra
from Model.get_model import get_model
from Data.get_data import get_data
from Analysis.run.linear_evaluation import LinearEvaluation
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
import os
import random

torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)


class AttributionVisualizer:
    def __init__(self, model, device, base_dir, task_output, threshold, folder_name='feature_attribution'):
        self.model = model.eval().to(device)
        self.device = device
        self.compression_model = self.model.models['merlin'].eval().to(device)
        self.tokenizer = self.compression_model.model.encode_text.tokenizer

        analysis_dir = os.path.join(base_dir, 'Analysis/output')
        self.output_dir = os.path.join(analysis_dir, folder_name, task_output, f">={threshold}")
        print(f"Saving to {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_sample(self, inputs, outputs, sample_ix, outcome):
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

            embedding_layer = self.compression_model.model.encode_text.text_encoder.embeddings.word_embeddings
            embedding_handle = embedding_layer.register_forward_hook(save_activation("embedding"))
            embedding_grad_handle = embedding_layer.register_backward_hook(save_gradient("embedding"))

            inputs["image"].requires_grad = True
            image_hook_handle = inputs["image"].register_hook(lambda grad: gradients.setdefault("image", grad))

            output = self.model(inputs)['ost_prediction']
            criterion = self.model.models['linear'].criterion
            target = torch.Tensor([[1 - outputs['ost'], outputs['ost']]]).to(output.device)
            loss = criterion(output, target)
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
            inputs, sample_ix, outcome, tokens,
            image_attributions, findings_attributions, sample_output_dir
        )

    def visualize_attribution(self, inputs, sample_ix, output, tokens, image_attributions, findings_attributions, output_dir):
        import html, base64, io
        from PIL import Image

        title = f"Patient {sample_ix} findings importance for {output}"

        attr_vals = findings_attributions[0].cpu().numpy()
        max_abs = np.abs(attr_vals).max() or 1
        alpha_min, alpha_max = 0.2, 0.8

        def styled_token(token, val):
            token_text = html.escape(token.replace("Ġ", ""))
            s = val / max_abs
            alpha = alpha_min + (alpha_max - alpha_min) * abs(s)
            color = f"rgba({'255, 0, 0' if s < 0 else '0, 255, 0'}, {alpha:.2f})"
            return f"<span style='background-color:{color}; color:black;'>{token_text}</span>"

        html_tokens = [styled_token(t, v) for t, v in zip(tokens, attr_vals)]
        main_body = " ".join(html_tokens)

        indices = np.arange(len(attr_vals))
        top5_pos = sorted(indices[attr_vals > 0], key=lambda i: attr_vals[i], reverse=True)[:5]
        top5_neg = sorted(indices[attr_vals < 0], key=lambda i: attr_vals[i])[:5]

        top_pos_list = "<ol>" + "".join(f"<li>{styled_token(tokens[i], attr_vals[i])}</li>" for i in top5_pos) + "</ol>"
        top_neg_list = "<ol>" + "".join(f"<li>{styled_token(tokens[i], attr_vals[i])}</li>" for i in top5_neg) + "</ol>"

        text_html_content = main_body + (
            "<hr><h2>Top 5 Positive Firing Tokens</h2>" + top_pos_list +
            "<h2>Top 5 Negative Firing Tokens</h2>" + top_neg_list
        )

        ct_volume = inputs["image"].detach().squeeze().numpy()  # shape: (D, H, W)
        ct_attributions = image_attributions.squeeze().numpy()

        # slice_importance = np.abs(ct_attributions).sum((0, 1))
        # top4_indices = np.argsort(slice_importance)[-4:]
        top4_indices = np.argsort(ct_attributions.max(axis=(0, 1)))[::-1][:4]  # Sort in descending order

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


@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    datasets, dataloaders = get_data(config, splits=['validation'])
    model = get_model(config)
    model.inference_map = {k: v for k, v in model.inference_map.items() if not v['compress']}
    model = model.eval().to(device)

    target = datasets['validation'].dataset.dataset[config.task.outputs[0]]
    qs = 10
    k = 5
    n_unique = target.nunique()
    if n_unique < qs:
        bins = np.linspace(target.min(), target.max(), n_unique + 1)
        quantiles = pd.cut(target, bins=bins, labels=False, include_lowest=True)
    else:
        quantiles = pd.qcut(target, q=qs, labels=False, duplicates='drop')

    threshold = 1
    sample_ixs = datasets['validation'].dataset.dataset.loc[quantiles >= threshold].sample(k).index

    visualizer = AttributionVisualizer(model, device, config.base_dir, config.task.outputs[0], threshold)
    for sample_ix in sample_ixs:
        inputs, outputs = datasets['validation'].__getitem__(sample_ix)
        visualizer.run_sample(inputs, outputs, sample_ix, config.task.outputs[0])

if __name__ == "__main__":
    main()
