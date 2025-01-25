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
from statsmodels.stats.multitest import multipletests

class SAEAnalysis:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config.task.sae_checkpoint
        self.output_dir = os.path.join(config.base_dir, 'SparseAutoencoder/analysis/output', os.path.basename(os.path.dirname(self.model_path)), os.path.splitext(os.path.basename(self.model_path))[0])
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_selectivity_matrix(self, selectivity_matrix):
        plt.figure(figsize=(15, 15))
        plt.imshow(selectivity_matrix, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label="Selectivity")

        step_x = max(1, len(selectivity_matrix.columns) // 20)
        step_y = max(1, len(selectivity_matrix.index) // 20)

        plt.xticks(
            np.arange(0, len(selectivity_matrix.columns), step_x),
            selectivity_matrix.columns[::step_x], rotation=90, fontsize=8
        )
        plt.yticks(
            np.arange(0, len(selectivity_matrix.index), step_y),
            selectivity_matrix.index[::step_y], fontsize=8
        )

        plt.tight_layout()

    def compute_selectivity_matrix(self, sae_output, phecode_columns, feature_columns):
        """
        Computes average and maximum selectivity metrics.
        Returns:
            avg_selectivity: [len(phecode_columns), len(feature_columns)]
            max_selectivity: [len(phecode_columns), len(feature_columns)]
        """
        feature_matrix = sae_output[feature_columns].values
        label_matrix = sae_output[phecode_columns].values

        label_sums = label_matrix.sum(axis=0)  # [n_labels]
        p_active_given_label = (label_matrix.T @ (feature_matrix > 0)) / label_sums[:, None]

        label_neg_sums = (label_matrix == 0).sum(axis=0)  # [n_labels]
        p_active_given_other = ((1 - label_matrix).T @ (feature_matrix > 0)) / label_neg_sums[:, None]

        avg_selectivity = p_active_given_label - p_active_given_other
        max_selectivity = p_active_given_label - np.max(p_active_given_other, axis=0)

        return avg_selectivity, max_selectivity

    def compute_statistics(self, sae_output):
        # Load data
        label_path = '/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv'
        raw_data = pd.read_csv(label_path)
        raw_data.columns = [str(float(col)) if i in range(1, 1693) else col for i, col in enumerate(raw_data.columns)]

        phewas_mapping_file = os.path.join(self.config.base_dir, self.config.data.phewas_mapping)
        phewas_mappings = pd.read_csv(phewas_mapping_file)

        # Ensure consistent data types
        phewas_mappings['phecode'] = phewas_mappings['phecode'].astype(float).astype(str).str.strip()

        disease_search_space = self.config.data.disease_search_space['Intensity/contrast']
        phewas_mappings = phewas_mappings.loc[phewas_mappings['phenotype'].isin(disease_search_space)]
        phecode_columns = phewas_mappings['phecode']
        raw_data = raw_data[['anon_accession'] + phecode_columns.tolist()]

        # Merge
        # assert set(sae_output['anon_accession']) == set(raw_data['anon_accession']), "Mismatch in anon_accession!"
        sae_output = sae_output.merge(raw_data, on='anon_accession', how='inner')

        # Features
        feature_columns = [col for col in sae_output.columns if col.startswith("Concept")]
        active_features = [col for col in feature_columns if sae_output[col].sum() > 0]

        # 1) Compute original selectivity
        avg_sel_original, max_sel_original = self.compute_selectivity_matrix(
            sae_output, phecode_columns, active_features
        )

        # Process & plot original average selectivity
        avg_sel_mat = pd.DataFrame(avg_sel_original, index=phecode_columns, columns=active_features)
        self.plot_selectivity_matrix(avg_sel_mat)
        plt.title("Average Selectivity Matrix")
        plt.savefig(os.path.join(self.output_dir, 'average_selectivity_matrix.jpg'), dpi=300)
        plt.close()

        # Process & plot original max selectivity
        max_sel_mat = pd.DataFrame(max_sel_original, index=phecode_columns, columns=active_features)
        self.plot_selectivity_matrix(max_sel_mat)
        plt.title("Maximum Selectivity Matrix")
        plt.savefig(os.path.join(self.output_dir, 'maximum_selectivity_matrix.jpg'), dpi=300)
        plt.close()

        # 2) Bootstrap null distributions
        bootstrap_avg = []
        bootstrap_max = []
        for _ in tqdm(range(self.config.task.n_bootstrap)):
            boot_data = sae_output.sample(frac=1, replace=True)
            avg_sel_boot, max_sel_boot = self.compute_selectivity_matrix(
                boot_data, phecode_columns, active_features
            )
            bootstrap_avg.append(avg_sel_boot)
            bootstrap_max.append(max_sel_boot)

        bootstrap_avg = np.stack(bootstrap_avg, axis=0)  # shape: (n_bootstrap, n_phecodes, n_features)
        bootstrap_max = np.stack(bootstrap_max, axis=0)

        # 3) Compute p-values (fraction of bootstrap samples >= observed)
        avg_p_values = np.mean(bootstrap_avg >= avg_sel_original[None, ...], axis=0)
        max_p_values = np.mean(bootstrap_max >= max_sel_original[None, ...], axis=0)

        # 4) Merge p-values with original selectivities and save
        # Convert arrays to DataFrames
        avg_sel_df = pd.DataFrame(avg_sel_original, index=phecode_columns, columns=active_features)
        max_sel_df = pd.DataFrame(max_sel_original, index=phecode_columns, columns=active_features)
        avg_pvals_df = pd.DataFrame(avg_p_values, index=phecode_columns, columns=active_features)
        max_pvals_df = pd.DataFrame(max_p_values, index=phecode_columns, columns=active_features)

        # Combine for average selectivity
        def link_phewas(selectivities, pvalues):
            df = avg_sel_df.stack().to_frame('Selectivity')
            df['p_value'] = avg_pvals_df.stack()
            df['-log10(p_value)'] = -np.log10(df['p_value'])

            df.reset_index(names=['Phecode', 'SAE Neuron'], inplace=True)
            df.sort_values('Selectivity', ascending=False, inplace=True)
            df['p_value_adjusted'] = multipletests(df['p_value'], method='fdr_bh')[1]

            df['Phecode'] = df['Phecode'].astype(float).astype(str).str.strip()
            
            # Merge PheWAS mapping with most significant Phecodes to include their descriptions and concepts
            significant_phecodes_df = pd.merge(
                df,
                phewas_mappings,
                left_on='Phecode',
                right_on='phecode',
                how='left'
            )
            significant_phecodes_df = significant_phecodes_df.drop_duplicates(['Phecode', 'SAE Neuron', 'phenotype', 'category'])
            return significant_phecodes_df

        def significant_selectivities(df):
            df = df.loc[df['p_value_adjusted'] < 0.1]
            # df = df.loc[df['Selectivity'] > 0.1]
            return df

        combined_avg = link_phewas(avg_sel_df, avg_pvals_df)
        # combined_avg = significant_selectivities(combined_avg)
        combined_avg.to_csv(os.path.join(self.output_dir, 'avg_selectivity_with_pvals.csv'), index=False)

        from util.manhattan_plot import plot_manhattan
        manhattan_dir = os.path.join(self.output_dir, 'manhattan')
        os.makedirs(manhattan_dir, exist_ok=True)
        plot_manhattan(combined_avg, 'SAE Neuron', 'phenotype', 'Selectivity', area_col='-log10(p_value)')
        plt.savefig(os.path.join(manhattan_dir, 'avg_selectivitiy.jpg'), dpi=300)

        plot_manhattan(combined_avg, 'SAE Neuron', 'phenotype', '-log10(p_value)', area_col='-log10(p_value)')
        plt.savefig(os.path.join(manhattan_dir, 'avg_selectivitiy_p_value.jpg'), dpi=300)

        combined_max = link_phewas(max_sel_df, max_pvals_df)
        # combined_max = significant_selectivities(combined_max)
        combined_max.to_csv(os.path.join(self.output_dir, 'max_selectivity_with_pvals.csv'), index=False)
        plot_manhattan(combined_max, 'SAE Neuron', 'phenotype', 'Selectivity', area_col='-log10(p_value)')
        plt.savefig(os.path.join(manhattan_dir, 'max_selectivitiy.jpg'), dpi=300)

        plot_manhattan(combined_max, 'SAE Neuron', 'phenotype', '-log10(p_value)', area_col='-log10(p_value)')
        plt.savefig(os.path.join(manhattan_dir, 'max_selectivitiy_p_value.jpg'), dpi=300)

        x = 3

    def analyze(self):
        wandb.init(
            project=self.config.wandb_project,
            group=self.config.task.wandb_group,
            entity=self.config.wandb_entity,
            mode=self.config.wandb_mode,
            config=dict(self.config)
        )

        vector_db_path = os.path.join(self.config.base_dir, self.config.data.vector_database_in)
        datasets = {
            "train": torch.load(f"{vector_db_path}/train.pt"),
            "validation": torch.load(f"{vector_db_path}/validation.pt"),
            "test": torch.load(f"{vector_db_path}/test.pt"),
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
        # for split in ["test"]:
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
