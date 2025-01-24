import hydra
import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from SparseAutoencoder.run.fit_sae import TrainSparseAutoencoder, ActivationDataModule
import matplotlib.pyplot as plt

class SAEAnalysis:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config.task.sae_checkpoint
        self.output_dir = os.path.join(config.base_dir, 'MedicalHypothesisGeneration/SparseAutoencoder/analysis/output', os.path.basename(os.path.dirname(self.model_path)), os.path.splitext(os.path.basename(self.model_path))[0])
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_selectivity(self, selectivity_matrix):
        plt.figure(figsize=(15, 15))
        plt.imshow(selectivity_matrix, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label="Selectivity")

        step_x = max(1, len(selectivity_matrix.columns) // 20)
        step_y = max(1, len(selectivity_matrix.index) // 10)
        plt.xticks(
            np.arange(0, len(selectivity_matrix.columns), step_x),
            selectivity_matrix.columns[::step_x], rotation=90, fontsize=8
        )
        plt.yticks(
            np.arange(0, len(selectivity_matrix.index), step_y),
            selectivity_matrix.index[::step_y], fontsize=8
        )

        plt.tight_layout()

    def process_selectivity(self, selectivity, phecode_columns, active_features):
        selectivity_matrix = pd.DataFrame(
            selectivity,
            index=phecode_columns,
            columns=active_features
        )

        thresholds = [0.1, 0.3, 0.5, 0.8, 0.9]
        threshold_summary = {
            thresh: (selectivity_matrix > thresh).sum().sum()
            for thresh in thresholds
        }
        print("Features exceeding selectivity thresholds:", threshold_summary)

        print('Most selective feature-label combinations')
        print((selectivity_matrix > 0.9).stack().loc[lambda x: x].index)

        threshold = 0.8  # Set the selectivity score threshold
        long_df = selectivity_matrix.reset_index().melt(id_vars='index', var_name='concept', value_name='selectivity_score')
        long_df.columns = ['Phecode', 'SAE Neuron', 'Selectivity']
        long_df = long_df.sort_values(by='Selectivity', ascending=False).reset_index(drop=True)

        most_significant_phecodes = long_df.loc[long_df['Selectivity'] > threshold]

        # Load PheWAS mapping file
        phewas_mapping_file = os.path.join(self.config.base_dir, 'MedicalHypothesisGeneration/MultimodalPretraining/data/raw_database/phewas-catalog.csv')
        phewas_mapping = pd.read_csv(phewas_mapping_file)

        # Ensure consistent data types
        phewas_mapping['phewas code'] = phewas_mapping['phewas code'].astype(str)

        # Merge PheWAS mapping with most significant Phecodes to include their descriptions and concepts
        significant_phecodes_df = pd.merge(
            most_significant_phecodes,
            phewas_mapping,
            left_on='Phecode',
            right_on='phewas code',
            how='inner'
        )

        # Output the results
        significant_phecodes_df = significant_phecodes_df[['Phecode', 'phewas phenotype', 'SAE Neuron', 'Selectivity']].drop_duplicates()
        print("Most significant Phecodes, descriptions, and concepts:")
        print(significant_phecodes_df)

        return selectivity_matrix, significant_phecodes_df

    def compute_statistics(self, sae_output):
        label_path = '/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv'
        raw_data = pd.read_csv(label_path)
        phecode_columns = raw_data.columns[1:1693].tolist()
        raw_data = raw_data[['anon_accession'] + phecode_columns]

        assert set(sae_output['anon_accession']) == set(raw_data['anon_accession']), "Mismatch in anon_accession!"
        sae_output = sae_output.merge(raw_data, on='anon_accession', how='inner')

        feature_columns = [col for col in sae_output.columns if col.startswith("Concept")]
        active_features = [col for col in feature_columns if sae_output[col].sum() > 0]

        feature_matrix = sae_output[active_features].values
        label_matrix = sae_output[phecode_columns].values

        label_sums = label_matrix.sum(axis=0)
        p_active_given_label = (label_matrix.T @ (feature_matrix > 0)) / label_sums[:, None]

        label_neg_sums = (label_matrix == 0).sum(axis=0)
        p_active_given_other = ((1 - label_matrix).T @ (feature_matrix > 0)) / label_neg_sums[:, None]

        avg_selectivity = p_active_given_label - p_active_given_other
        average_selectivity_matrix, significant_phecodes_df = self.process_selectivity(avg_selectivity, phecode_columns, active_features)
        self.plot_selectivity(average_selectivity_matrix)
        plt.title("Average Selectivity Matrix")
        plt.savefig(os.path.join(self.output_dir, 'average_selectivity_matrix.jpg'), dpi=300)
        plt.close()
        significant_phecodes_df.to_csv(os.path.join(self.output_dir, 'most_average_selective_phecode_neurons.csv'))

        max_selectivity = p_active_given_label - np.max(p_active_given_other, axis=0)
        max_selectivity_matrix, significant_phecodes_df = self.process_selectivity(max_selectivity, phecode_columns, active_features)
        self.plot_selectivity(max_selectivity_matrix)
        plt.title("Maximum Selectivity Matrix")
        plt.savefig(os.path.join(self.output_dir, 'maximum_selectivity_matrix.jpg'), dpi=300)
        plt.close()
        significant_phecodes_df.to_csv(os.path.join(self.output_dir, 'most_maximum_selective_phecode_neurons.csv'))

        x = 3

    def analyze(self):
        wandb.init(
            project=self.config.wandb_project,
            group=self.config.task.wandb_group,
            entity=self.config.wandb_entity,
            mode=self.config.wandb_mode,
            config=dict(self.config)
        )

        datasets = {
            "train": torch.load(f"{self.config.data.vector_database_in}/train.pt"),
            "validation": torch.load(f"{self.config.data.vector_database_in}/validation.pt"),
            "test": torch.load(f"{self.config.data.vector_database_in}/test.pt"),
        }

        checkpoint_path = self.config.task.sae_checkpoint
        sae = TrainSparseAutoencoder.load_from_checkpoint(
            checkpoint_path, strict=False
        ).sae

        activation_data_module = ActivationDataModule(self.config, datasets)
        activation_dataloaders = {
            'train': activation_data_module.train_dataloader(),
            'validation': activation_data_module.val_dataloader(),
            'test': activation_data_module.test_dataloader(),
        }

        from MultimodalPretraining.util.extract_model_output import extract_vectors_for_split
        input_field = 0
        fields = [1]

        all_data = []

        for split in ["train", "validation", "test"]:
            split_data = extract_vectors_for_split(self.config, activation_dataloaders[split], sae, input_field, fields)
            split_df = pd.DataFrame(split_data)
            split_df.columns = ['concepts', 'anon_accession']
            split_df['split'] = split

            concepts_array = np.vstack(split_df['concepts'].apply(np.array).to_numpy())
            concepts_df = pd.DataFrame(concepts_array, columns=[f'Concept {i}' for i in range(concepts_array.shape[1])])
            split_df = pd.concat([split_df.drop(columns=['concepts']), concepts_df], axis=1)

            all_data.append(split_df)

        sae_output = pd.concat(all_data, ignore_index=True)
        self.compute_statistics(sae_output)

        wandb.finish()

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    from util.wandb import init_wandb
    init_wandb(config)

    analysis = SAEAnalysis(config)
    analysis.analyze()

if __name__ == "__main__":
    main()
