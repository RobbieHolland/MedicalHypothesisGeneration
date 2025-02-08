from Data.raw_database.abdominal_ct import create_dataloaders
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import os
import torch

class ActivationsDataset(Dataset):
    def __init__(self, config, data):
        self.config = config

        self.vectors = data['vectors'].clone() if isinstance(data['vectors'], torch.Tensor) else torch.tensor(data['vectors'])
        self.outputs = pd.DataFrame(data[config.task.outputs])

        if config.task.output_filter:
            mask = self.outputs.isin(config.task.output_filter).all(axis=1)
            self.outputs = self.outputs[mask]
            self.vectors = self.vectors[mask.values]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.vectors[idx], self.data[self.config.output_fields][idx]

class ActivationDataModule(pl.LightningDataModule):
    def __init__(self, config, activations, output_fields='sample_ids'):
        super().__init__()
        self.config = config
        self.activations = activations
        self.output_fields = output_fields

    def _create_dataloader(self, dataset_data):
        dataset = ActivationsDataset(self.config, dataset_data, self.output_fields)
        return DataLoader(dataset, batch_size=self.config.task.batch_size, shuffle=True)

    def train_dataloader(self):
        return self._create_dataloader(self.activations['train'])

    def val_dataloader(self):
        return self._create_dataloader(self.activations['validation'])

    def test_dataloader(self):
        return self._create_dataloader(self.activations['test'])

import numpy as np
import pytorch_lightning as pl

class EmbeddingDataset(pl.LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        
        # Apply filtering if needed
        if config.task.output_filter:
            dataset = dataset.loc[dataset[config.task.outputs].isin(config.task.output_filter).all(axis=1)]
            dataset = dataset.reset_index(drop=True)
        
        # Convert to numpy arrays for efficient access
        self.input_keys = config.data.inputs
        self.output_keys = config.task.outputs
        
        self.input_data = dataset[self.input_keys].to_numpy()
        self.output_data = dataset[self.output_keys].to_numpy()
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_row = self.input_data[idx]
        output_row = self.output_data[idx]
        
        # return dict(zip(self.input_keys, input_row)), dict(zip(self.output_keys, output_row))
        return input_row, output_row

class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, config, datasets):
        super().__init__()
        self.config = config
        self.datasets = datasets

    def _create_dataloader(self, split):
        return DataLoader(self.datasets[split], batch_size=self.config.task.batch_size, shuffle=True)

    def train_dataloader(self):
        return self._create_dataloader('train')

    def val_dataloader(self):
        return self._create_dataloader('validation')

    def test_dataloader(self):
        return self._create_dataloader('test')


def get_data(config, device=None):
    data_name = config.data.name

    if data_name == 'abdominal_ct':
        datasets, dataloaders = create_dataloaders(config)

    elif data_name == 'merlin_embeddings':
        # abdominal_datasets, _ = create_dataloaders(config)

        # Usage:
        vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)
        labels_path = config.paths.five_year_prognosis_labels
        labels_df = pd.read_csv(labels_path).set_index("anon_accession")
        labels_df = labels_df.drop_duplicates()
        labels_df = labels_df.reset_index()
        # merlin_datasets, _ = create_dataloaders(config)

        datasets = {}
        for split in ["train", "validation", "test"]:
            embedding_data = torch.load(os.path.join(vector_db_path, f"{split}.pt"))
            df = pd.DataFrame({
                "sample_ids": embedding_data['sample_ids'],
                "vectors": list(embedding_data["vectors"].detach().cpu().numpy())
            })
            merged_df = df.merge(labels_df, left_on="sample_ids", right_on="anon_accession", how="left")
            merged_df = merged_df.reset_index(drop=True)
            datasets[split] = EmbeddingDataset(config, merged_df)

        activation_data_module = EmbeddingDataModule(config, datasets)
        dataloaders = {
            'train': activation_data_module.train_dataloader(),
            'validation': activation_data_module.val_dataloader(),
            'test': activation_data_module.test_dataloader(),
        }

    elif data_name == 'phecodes':
        labels_df = pd.read_csv(config.paths.abdominal_phecode_labels)
        prognosis_labels_df = pd.read_csv(config.paths.five_year_prognosis_labels).set_index("anon_accession")
        vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)

        datasets = {}
        for split in ["train", "validation", "test"]:
            embedding_data = torch.load(os.path.join(vector_db_path, f"{split}.pt"))

            df = pd.DataFrame({
                "sample_ids": embedding_data['sample_ids'],
            })
            merged_df = df.merge(labels_df, left_on="sample_ids", right_on="anon_accession", how="left")
            merged_df = merged_df.merge(prognosis_labels_df, left_on="anon_accession", right_on="anon_accession", how="left")
            merged_df = merged_df.reset_index(drop=True)
            datasets[split] = EmbeddingDataset(config, merged_df)

        activation_data_module = EmbeddingDataModule(config, datasets)
        dataloaders = {
            'train': activation_data_module.train_dataloader(),
            'validation': activation_data_module.val_dataloader(),
            'test': activation_data_module.test_dataloader(),
        }

    else:
        raise ValueError(f"Unknown data name: {data_name}")

    return datasets, dataloaders
