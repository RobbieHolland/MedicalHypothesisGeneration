import hydra
import torch
import wandb
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

    def compute_statistics(self, sae_output, phecode_columns, phewas_mappings):
        # Features
        feature_columns = [col for col in sae_output.columns if col.startswith("Concept")]
        active_features = [col for col in feature_columns if sae_output[col].var() > 0]

        ### Calculate relationships and p-values
        from SparseAutoencoder.statistics.logistic_regression import LogisticRegressionAnalysis
        selectivity_analysis = LogisticRegressionAnalysis(
            sae_output=sae_output,
            phecode_columns=phecode_columns,
            active_features=active_features,
            output_dir=self.output_dir,
            config=self.config
        )

        # Perform the selectivity analysis
        effects, p_values = selectivity_analysis.perform_analysis()

        # Combine for average selectivity
        def link_phewas(effects, pvalues):
            df = effects.stack().to_frame('Effect')
            df['p_value'] = pvalues.stack()
            df['-log10(p_value)'] = -np.log10(df['p_value'])

            df.reset_index(names=['Phecode', 'SAE Neuron'], inplace=True)
            df.sort_values('Effect', ascending=False, inplace=True)

            # df['Phecode'] = df['Phecode'].astype(float).astype(str).str.strip()
            
            # Merge PheWAS mapping with most significant Phecodes to include their descriptions and concepts
            # significant_phecodes_df = pd.merge(
            #     df,
            #     phewas_mappings,
            #     left_on='Phecode',
            #     right_on='phecode',
            #     how='left'
            # )
            # significant_phecodes_df = df.drop_duplicates(['Phecode', 'SAE Neuron', 'phenotype', 'category'])
            return df

        def significant_selectivities(df):
            df = df.loc[df['p_value_adjusted'] < 0.1]
            # df = df.loc[df['Selectivity'] > 0.1]
            return df

        combined_avg = link_phewas(effects, p_values)
        # combined_avg = significant_selectivities(combined_avg)
        combined_avg.to_csv(os.path.join(self.output_dir, 'effects_with_pvals.csv'), index=False)

        from util.manhattan_plot import plot_manhattan
        manhattan_dir = os.path.join(self.output_dir, 'manhattan')
        os.makedirs(manhattan_dir, exist_ok=True)

        plot_manhattan(combined_avg, 'SAE Neuron', 'Phecode', 'Effect', area_col='-log10(p_value)')
        plt.savefig(os.path.join(manhattan_dir, 'effects.jpg'), dpi=300)

        significant_effects = combined_avg[combined_avg['p_value'] <= 0.05]
        plot_manhattan(significant_effects, 'SAE Neuron', 'Phecode', 'Effect', area_col='-log10(p_value)')
        plt.savefig(os.path.join(manhattan_dir, 'significant_effects_not_corrected.jpg'), dpi=300)

        plot_manhattan(combined_avg, 'SAE Neuron', 'Phecode', '-log10(p_value)', area_col='-log10(p_value)')
        num_tests = len(combined_avg)
        bonferroni_threshold = -np.log10(0.05 / num_tests)
        combined_avg['-log10(bonferri)'] = bonferroni_threshold
        plt.axhline(y=bonferroni_threshold, color='red', linestyle='dashed', linewidth=1, label='Bonferroni Threshold')
        plt.savefig(os.path.join(manhattan_dir, 'p_values.jpg'), dpi=300)

        significant_rows = combined_avg[combined_avg['-log10(p_value)'] > bonferroni_threshold]
        significant_rows.to_csv(os.path.join(self.output_dir, 'significant_concept_label_associations.csv'), index=False)

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

        checkpoint_path = os.path.join(self.config.pretrained_model_dir, self.config.task.sae_checkpoint)
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

        # Load known-concept data
        # label_path = '/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv'
        label_path = '/dataNAS/people/akkumar/contrastive-3d/data/merged_labels_diseases_filtered.csv'
        raw_data = pd.read_csv(label_path)
        # raw_data.columns = [str(float(col)) if i in range(1, 1693) else col for i, col in enumerate(raw_data.columns)]

        phewas_mapping_file = os.path.join(self.config.base_dir, self.config.data.phewas_mapping)
        phewas_mappings = pd.read_csv(phewas_mapping_file)

        # Ensure consistent data types
        phewas_mappings['phecode'] = phewas_mappings['phecode'].astype(float).astype(str).str.strip()

        # disease_search_space = self.config.data.disease_search_space['merlin_diseases']
        # existing_phewas_mappings = phewas_mappings.loc[phewas_mappings['phenotype'].isin(disease_search_space)]
        # phecode_columns = existing_phewas_mappings['phecode']
        existing_phewas_mappings = None
        phecode_columns = raw_data.columns[5:]
        raw_data = raw_data[['anon_accession'] + phecode_columns.tolist()]
        raw_data = raw_data.replace(-1, np.nan)
        
        prevalence = raw_data[phecode_columns.tolist()].apply(pd.Series.value_counts)
        print(prevalence)

        # Merge
        # assert set(sae_output['anon_accession']) == set(raw_data['anon_accession']), "Mismatch in anon_accession!"
        sae_output = sae_output.merge(raw_data, on='anon_accession', how='inner')
        sae_output.to_pickle(os.path.join(self.output_dir, 'sae_output-known_concepts.pkl'))

        self.compute_statistics(sae_output, phecode_columns, existing_phewas_mappings)

        wandb.finish()

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    from util.wandb import init_wandb
    init_wandb(config)

    analysis = SAEAnalysis(config)
    analysis.analyze()

if __name__ == "__main__":
    main()
