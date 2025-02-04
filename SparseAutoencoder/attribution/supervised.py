import torch
import hydra
import torch.utils.checkpoint
from MultimodalPretraining.data.raw_database.dataset import create_dataloaders
import torch
import nibabel as nib
import numpy as np
import os
import pandas as pd
from functools import partial

# Override checkpoint function globally to disable it
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

class FeatureAttribution:
    def __init__(self, config):
        self.config = config
        self.model_path = config.task.sae_checkpoint
        self.analysis_dir = os.path.join(config.base_dir, 'SparseAutoencoder/analysis/output', os.path.basename(os.path.dirname(self.model_path)), os.path.splitext(os.path.basename(self.model_path))[0])
        self.output_dir = os.path.join(self.analysis_dir, 'feature_attribution')
    
        # Load dataset
        raw_data_path = os.path.join(self.analysis_dir, 'sae_output-known_concepts.pkl')
        raw_data = pd.read_pickle(raw_data_path)

        self.datasets, self.dataloaders = create_dataloaders(config, sets=['validation'])
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        # Load model and set evaluation mode
        from MultimodalPretraining.model.model import load_model
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
        self.model = CombinedModel(encoder, sae)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def apply_threshold_mask(self, image_volume, cam_volume, threshold=0.8):
        image_volume = image_volume.squeeze()
        cam_volume = cam_volume.squeeze()
        # mask = cam_volume > threshold

        # mask = torch.ones_like(image_volume) * 0.05
        mask = torch.zeros_like(image_volume)
        mask[cam_volume > 0.003] = 0.1
        mask[cam_volume > 0.01] = 0.2
        mask[cam_volume > 0.1] = 0.5
        mask[cam_volume > 0.9] = 0.85
        mask[cam_volume > 0.95] = 0.95

        masked_volume = image_volume * mask
        # masked_volume = image_volume * cam_volume
        return masked_volume

    def save_nifti(self, volume, output_path, affine=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if isinstance(volume, torch.Tensor):
            volume = volume.detach().cpu().numpy()

        if volume.ndim == 4:
            volume = volume[0]

        if affine is None:
            affine = np.eye(4)

        print('Saving attribution to', output_path)
        nib.save(nib.Nifti1Image(volume, affine), output_path)

    def supervised_clip_attribution(self):
        # phecode = 'splenomegaly'
        phecodes = [
            'splenomegaly',
            'appendicitis',
            'pancreatic_atrophy',
            'atherosclerosis',
            'hepatomegaly',
            'prostatomegaly',
        ]

        for phecode in phecodes:
            self.model.forward = lambda x: self.model.clip_forward(x, phecode=phecode)
            dataset = pd.DataFrame(self.datasets['validation'].data)['anon_accession']

            association_dir = os.path.join(self.output_dir, "supervised_clip", phecode)
            raw_data = pd.read_csv('/dataNAS/people/akkumar/contrastive-3d/data/merged_labels_diseases_filtered.csv')
            print(f'Total support for {phecode} is {(raw_data[phecode] == 1).sum()}')

            filtered_data = raw_data.loc[raw_data['anon_accession'].isin(dataset)]

            n = 5
            # label_values = [1, 0]
            label_values = [1]
            for label in label_values:
                samples = filtered_data.loc[filtered_data[phecode] == label].sample(n)
                sample_data = [self.datasets['validation'].__getitem__(dataset.index[dataset == aa].item())['image'] for aa in samples['anon_accession'].tolist()]
                save_path = os.path.join(association_dir, f"label={label}")

                # Ensure model outputs are finite before attribution
                for i, data in enumerate(sample_data):
                    data = data.unsqueeze(0).to(self.device)
                    effect = 2 * (label - 0.5) # +1 for true, -1 for false
                    target = label

                    from SparseAutoencoder.attribution.gradcam import gradcam
                    target_layers = [self.model.encoder.model.encode_image.i3_resnet.layer4[-1]]  # Adjust this based on your architecture
                    attribution_map = gradcam(self.model, data, effect, target, target_layers)

                    masked_data = self.apply_threshold_mask(data, attribution_map)
                    # save_nifti(data[0][0].detach().numpy(), os.path.join(association_dir, "original_volume.nii"))

                    self.save_nifti(masked_data, os.path.join(save_path, f"{i}_masked_volume.nii"))

        x = 3

    def supervised_phecode_attribution(self):
        self.model.forward = self.model.supervised_forward
        # phecode = '579.2'
        phecode = '540.1'
        dataset = pd.DataFrame(self.datasets['validation'].data)['anon_accession']

        association_dir = os.path.join(self.output_dir, "supervised", phecode)
        raw_data = pd.read_csv('/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv')
        print(f'Total support for {phecode} is {(raw_data[phecode] == 1).sum()}')

        filtered_data = raw_data.loc[raw_data['anon_accession'].isin(dataset)]

        n = 5
        label_values = [1, 0]
        for label in label_values:
            samples = filtered_data.loc[filtered_data[phecode] == label].sample(n)
            sample_data = [self.datasets['validation'].__getitem__(dataset.index[dataset == aa].item())['image'] for aa in samples['anon_accession'].tolist()]
            save_path = os.path.join(association_dir, f"label={label}")

            # Ensure model outputs are finite before attribution
            for i, data in enumerate(sample_data):
                data = data.unsqueeze(0).to(self.device)
                effect = 2 * (label - 0.5) # +1 for true, -1 for false
                target = raw_data.columns[1:1693].get_loc(phecode)

                from SparseAutoencoder.attribution.gradcam import gradcam
                target_layers = [self.model.encoder.model.encode_image.i3_resnet.layer4[-1]]  # Adjust this based on your architecture
                attribution_map = gradcam(self.model, data, effect, target, target_layers)

                masked_data = self.apply_threshold_mask(data, attribution_map)
                # save_nifti(data[0][0].detach().numpy(), os.path.join(association_dir, "original_volume.nii"))

                self.save_nifti(masked_data, os.path.join(save_path, f"{i}_masked_volume.nii"))

        x = 3

    def concept_attribution(self):
        self.model.forward = self.model.concept_forward

        # Load significant associations and vector database
        significant_associations = pd.read_csv(os.path.join(self.analysis_dir, 'significant_concept_label_associations.csv'))
        sae_known_concept_db = pd.read_pickle(os.path.join(self.analysis_dir, 'sae_output-known_concepts.pkl'))

        dataset = pd.DataFrame(self.datasets['validation'].data)['anon_accession']
        filtered_sae_known_concept_db = sae_known_concept_db.loc[sae_known_concept_db['anon_accession'].isin(dataset)]

        for i, association in significant_associations.iterrows():
            # association_dir = os.path.join(output_dir, f"{association['category']}/{association['phenotype']}/{association['SAE Neuron']}")
            association_dir = os.path.join(self.output_dir, f"{association['Phecode']}/{association['SAE Neuron']}")

            qs = 10
            n = 5
            quantiles = pd.qcut(filtered_sae_known_concept_db[association['SAE Neuron']], q=qs, labels=False, duplicates='drop')

            # quantiles_of_interest = [0, qs - 1]
            quantiles_of_interest = [quantiles.max()]
            for quantile in quantiles_of_interest:
                save_path = os.path.join(association_dir, f"q={quantile}")
            
                quantile_activating_sample = filtered_sae_known_concept_db.loc[quantiles == quantile].sample(n)
                quantile_activating_sample_data = [self.datasets['validation'].__getitem__(dataset.index[dataset == aa].item())['image'] for aa in quantile_activating_sample['anon_accession'].tolist()]
            
                # Ensure model outputs are finite before attribution
                for i, data in enumerate(quantile_activating_sample_data):
                    data = data.unsqueeze(0).to(self.device)

                    from SparseAutoencoder.attribution.gradcam import gradcam
                    target_layers = [self.model.encoder.model.encode_image.i3_resnet.layer4[-1]]  # Adjust this based on your architecture
                    attribution_map = gradcam(self.model, data, association['Effect'], int(association['SAE Neuron'].replace('Concept ', '')), target_layers)

                    masked_data = self.apply_threshold_mask(data, attribution_map)
                    # save_nifti(data[0][0].detach().numpy(), os.path.join(association_dir, "original_volume.nii"))

                    self.save_nifti(masked_data, os.path.join(save_path, f"{i}_masked_volume.nii"))

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    feature_attribution = FeatureAttribution(config)

    # feature_attribution.concept_attribution()
    # feature_attribution.supervised_phecode_attribution()
    feature_attribution.supervised_clip_attribution()


if __name__ == "__main__":
    main()
