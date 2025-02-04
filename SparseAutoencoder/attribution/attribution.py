import torch
import hydra
import torch.utils.checkpoint
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from MultimodalPretraining.data.raw_database.dataset import create_dataloaders
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Override checkpoint function globally to disable it
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

import torch
import nibabel as nib
import numpy as np

def apply_threshold_mask(image_volume, cam_volume, threshold=0.8):
    image_volume = image_volume.squeeze()
    cam_volume = cam_volume.squeeze()
    # mask = cam_volume > threshold

    # mask = torch.ones_like(image_volume) * 0.05
    mask = torch.zeros_like(image_volume)
    mask[cam_volume > 0.3] = 0.7
    mask[cam_volume > 0.9] = 0.85
    mask[cam_volume > 0.95] = 0.95

    masked_volume = image_volume * mask
    # masked_volume = image_volume * cam_volume
    return masked_volume

def save_nifti(volume, output_path, affine=None):
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    if volume.ndim == 4:
        volume = volume[0]

    if affine is None:
        affine = np.eye(4)

    print('Saving attribution to', output_path)
    nib.save(nib.Nifti1Image(volume, affine), output_path)

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    import os
    model_path = config.task.sae_checkpoint
    analysis_dir = os.path.join(config.base_dir, 'SparseAutoencoder/analysis/output', os.path.basename(os.path.dirname(model_path)), os.path.splitext(os.path.basename(model_path))[0])
    output_dir = os.path.join(analysis_dir, 'feature_attribution')
    
    # Load dataset
    datasets, dataloaders = create_dataloaders(config, sets=['validation'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from MultimodalPretraining.model.model import load_model

    # Load model and set evaluation mode
    encoder = load_model(config)
    encoder.eval()  # Ensures proper BatchNorm and Dropout behavior

    # Load SAE
    checkpoint_path = os.path.join(config.pretrained_model_dir, config.task.sae_checkpoint)
    from SparseAutoencoder.run.fit_sae import TrainSparseAutoencoder, ActivationDataModule
    sae = TrainSparseAutoencoder.load_from_checkpoint(
            checkpoint_path, strict=False
    ).sae

    # Create combined model
    from SparseAutoencoder.model.combined_model import CombinedModel
    model = CombinedModel(encoder, sae)
    model.eval()
    model = model.to(device)

    # Load significant associations and vector database
    import pandas as pd
    significant_associations = pd.read_csv(os.path.join(analysis_dir, 'significant_concept_label_associations.csv'))
    sae_known_concept_db = pd.read_pickle(os.path.join(analysis_dir, 'sae_output-known_concepts.pkl'))

    dataset = pd.DataFrame(datasets['validation'].data)['anon_accession']
    filtered_sae_known_concept_db = sae_known_concept_db.loc[sae_known_concept_db['anon_accession'].isin(dataset)]

    for i, association in significant_associations.iterrows():
        # association_dir = os.path.join(output_dir, f"{association['category']}/{association['phenotype']}/{association['SAE Neuron']}")
        association_dir = os.path.join(output_dir, f"{association['Phecode']}/{association['SAE Neuron']}")
        os.makedirs(association_dir, exist_ok=True)

        qs = 10
        n = 5
        quantiles = pd.qcut(filtered_sae_known_concept_db[association['SAE Neuron']], q=qs, labels=False, duplicates='drop')

        # quantiles_of_interest = [0, qs - 1]
        quantiles_of_interest = [quantiles.max()]
        for quantile in quantiles_of_interest:
            save_path = os.path.join(association_dir, f"q={quantile}")
            os.makedirs(save_path, exist_ok=True)
        
            quantile_activating_sample = filtered_sae_known_concept_db.loc[quantiles == quantile].sample(n)
            quantile_activating_sample_data = [datasets['validation'].__getitem__(dataset.index[dataset == aa].item())['image'] for aa in quantile_activating_sample['anon_accession'].tolist()]
        
            # Ensure model outputs are finite before attribution
            for i, data in enumerate(quantile_activating_sample_data):
                data = data.unsqueeze(0).to(device)

                from SparseAutoencoder.attribution.gradcam import gradcam
                target_layers = [model.encoder.model.encode_image.i3_resnet.layer4[-1]]  # Adjust this based on your architecture
                attribution_map = gradcam(model, data, association['Effect'], int(association['SAE Neuron'].replace('Concept ', '')), target_layers)

                masked_data = apply_threshold_mask(data, attribution_map)
                # save_nifti(data[0][0].detach().numpy(), os.path.join(association_dir, "original_volume.nii"))

                save_nifti(masked_data, os.path.join(save_path, f"{i}_masked_volume.nii"))

if __name__ == "__main__":
    main()
