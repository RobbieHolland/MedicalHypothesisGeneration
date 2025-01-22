import hydra
import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pytorch_lightning as pl
import os
from SparseAutoencoder.run.fit_sae import TrainSparseAutoencoder, ActivationDataModule
import pandas as pd
import numpy as np
from tqdm import tqdm

def plot_selectivity(config, selectivity_matrix):
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot the heatmap
    plt.figure(figsize=(15, 15))  # Adjust size as needed
    plt.imshow(selectivity_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label="Avg Selectivity")
    plt.title("Selectivity Matrix")

    # Show ticks only for a subset of labels
    step_x = max(1, len(selectivity_matrix.columns) // 20)  # Show every 20th feature
    step_y = max(1, len(selectivity_matrix.index) // 10)   # Show every 10th label
    plt.xticks(
        np.arange(0, len(selectivity_matrix.columns), step_x),
        selectivity_matrix.columns[::step_x], rotation=90, fontsize=8
    )
    plt.yticks(
        np.arange(0, len(selectivity_matrix.index), step_y),
        selectivity_matrix.index[::step_y], fontsize=8
    )

    # Save the image
    plt.tight_layout()
    plt.savefig(os.path.join(config.base_dir, config.task.output_dir, 'selectivity_matrix.jpg'), dpi=300)  # Save as PNG with high resolution
    plt.close()

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def analyze_sae(config):
    wandb.init(project=config.wandb_project, group=config.task.wandb_group, entity=config.wandb_entity, mode=config.wandb_mode, config=dict(config))

    datasets = {
        "train": torch.load(f"{config.data.vector_database_in}/train.pt"),
        "validation": torch.load(f"{config.data.vector_database_in}/validation.pt"),
        "test": torch.load(f"{config.data.vector_database_in}/test.pt"),
    }

    checkpoint_path = config.task.sae_checkpoint
    sae = TrainSparseAutoencoder.load_from_checkpoint(
        checkpoint_path,
        strict=False  # Ensures the checkpoint's config is retained
    ).sae

    activation_datasets = {
        "train": torch.load(f"{config.data.vector_database_in}/train.pt"),
        "validation": torch.load(f"{config.data.vector_database_in}/validation.pt"),
        "test": torch.load(f"{config.data.vector_database_in}/test.pt"),
    }

    # Activations
    activation_data_module = ActivationDataModule(config, datasets)
    activation_dataloaders = {
        'train': activation_data_module.train_dataloader(),
        'validation': activation_data_module.val_dataloader(),
        'test': activation_data_module.test_dataloader(),
    }

    # Extract SAE concepts from activations
    from MultimodalPretraining.util.extract_model_output import extract_vectors_for_split
    input_field = 0
    fields = [1]

    all_data = []

    for split in ["train", "validation", "test"]:
        # Extract data and create the initial DataFrame
        split_data = extract_vectors_for_split(config, activation_dataloaders[split], sae, input_field, fields)
        split_df = pd.DataFrame(split_data)
        split_df.columns = ['concepts', 'anon_accession']
        split_df['split'] = split
        
        # Convert 'concepts' to a NumPy array and create a DataFrame for the expanded concepts
        concepts_array = np.vstack(split_df['concepts'].apply(np.array).to_numpy())
        concepts_df = pd.DataFrame(concepts_array, columns=[f'Concept {i}' for i in range(concepts_array.shape[1])])

        # Combine the expanded concepts with the original DataFrame
        split_df = pd.concat([split_df.drop(columns=['concepts']), concepts_df], axis=1)
        
        all_data.append(split_df)

    # Concatenate all splits into one DataFrame
    sae_output = pd.concat(all_data, ignore_index=True)

    # # Raw datasets
    # from MultimodalPretraining.data.raw_database.dataset import create_dataloaders
    # datasets, dataloaders = create_dataloaders(config)

    # Ensure 'anon_accession' exists in both DataFrames and merge them
    # merged_data = pd.merge(sae_output, raw_data, on='anon_accession', how='inner')

    # Load and preprocess the datasets
    label_path = '/dataNAS/people/lblankem/contrastive-3d/data/stanford_labels.csv'
    raw_data = pd.read_csv(label_path)
    phecode_columns = raw_data.columns[1:1693].tolist()  # Identify phecode columns
    raw_data = raw_data[['anon_accession'] + phecode_columns]

    # Concatenate sae_output and ensure 'anon_accession' alignment
    sae_output = pd.concat(all_data, ignore_index=True)
    assert set(sae_output['anon_accession']) == set(raw_data['anon_accession']), "Mismatch in anon_accession!"

    # Join phecodes directly into sae_output
    sae_output = sae_output.merge(raw_data, on='anon_accession', how='inner')

    # Identify feature and label columns
    feature_columns = [col for col in sae_output.columns if col.startswith("Concept")]
    active_features = [col for col in feature_columns if sae_output[col].sum() > 0]

    # Extract feature and label matrices
    feature_matrix = sae_output[active_features].values  # Shape: (n_samples, n_active_features)
    label_matrix = sae_output[phecode_columns].values      # Shape: (n_samples, n_labels)

    # Compute the probabilities of feature activation given each label
    label_sums = label_matrix.sum(axis=0)  # Total counts for each label
    p_active_given_label = (label_matrix.T @ (feature_matrix > 0)) / label_sums[:, None]  # Shape: (n_labels, n_active_features)

    # Compute the probabilities of feature activation given ~label
    label_neg_sums = (label_matrix == 0).sum(axis=0)  # Total counts for ~label
    p_active_given_other = ((1 - label_matrix).T @ (feature_matrix > 0)) / label_neg_sums[:, None]

    # Compute the average and max selectivity scores
    avg_selectivity = p_active_given_label - p_active_given_other
    max_selectivity = p_active_given_label - np.max(p_active_given_other, axis=0)

    # Create a DataFrame for results
    selectivity_matrix = pd.DataFrame(
        avg_selectivity,
        index=phecode_columns,
        columns=active_features
    )
    plot_selectivity(config, selectivity_matrix)

    thresholds = [0.1, 0.3, 0.5, 0.8]
    threshold_summary = {
        thresh: (selectivity_matrix > thresh).sum().sum()  # Count all values exceeding the threshold
        for thresh in thresholds
    }
    print("Features exceeding selectivity thresholds:", threshold_summary)

    print('Most selective feature-label combinations')
    print((selectivity_matrix > 0.8).stack().loc[lambda x: x].index)

    wandb.finish()

if __name__ == "__main__":
    analyze_sae()
