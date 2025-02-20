import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from Pretraining.util.extract_model_output import extract_vectors_for_split
import pytorch_lightning as pl

class EmbeddingDataset(pl.LightningDataModule):
    def __init__(self, config, dataset, autofilter=True):
        super().__init__()
        self.config = config

        self.original_dataset = dataset
        self.dataset = self.original_dataset.copy()

        # Convert to numpy arrays for efficient access
        self.input_keys = config.data.inputs
        self.output_keys = config.task.outputs

        if autofilter:
            self.filter()

        self.set_inputs_outputs()

    def set_inputs_outputs(self):
        self.input_data = self.dataset[self.input_keys].to_numpy()
        self.output_data = self.dataset[self.output_keys].to_numpy()

    def filter(self):
        if self.config.task.output_filter:
            inclusion_mask = self.dataset[self.config.task.outputs].isin(self.config.task.output_filter).all(axis=1)
            self.dataset = self.dataset.loc[inclusion_mask]
            self.dataset = self.dataset.reset_index(drop=True)

        self.set_inputs_outputs()

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_row = self.input_data[idx]
        output_row = self.output_data[idx]
        
        # return dict(zip(self.input_keys, input_row)), dict(zip(self.output_keys, output_row))
        return input_row, output_row

    @staticmethod
    def custom_collate(batch):
        inputs, outputs = zip(*batch)  # Separate inputs and outputs
        inputs = [list(field) for field in zip(*inputs)]  # Transpose inputs
        outputs = [list(field) for field in zip(*outputs)]  # Transpose outputs
        return inputs, outputs

class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, config, datasets):
        super().__init__()
        self.config = config
        self.datasets = datasets

    def _create_dataloader(self, split):
        return DataLoader(self.datasets[split], batch_size=self.config.task.batch_size, shuffle=True, collate_fn=self.datasets[split].custom_collate)

    def train_dataloader(self):
        return self._create_dataloader('train')

    def val_dataloader(self):
        return self._create_dataloader('validation')

    def test_dataloader(self):
        return self._create_dataloader('test')

class CompressedEmbeddingDataset(EmbeddingDataset):
    def __init__(self, config, dataset, model_field_pairs, split, batch_size=224):
        """
        Args:
            config: Configuration object.
            dataset: Pandas DataFrame containing the dataset.
            model_field_pairs: List of (model, input_field) where:
                - model: An object with a `.latent` method.
                - input_field: Column name to compress using the model.
            split: Dataset split name (e.g., 'train', 'validation', 'test').
            batch_size: Batch size for DataLoader.
        """
        super().__init__(config, dataset, autofilter=False)
        
        self.split = split
        self.batch_size = batch_size

        # Set up DataLoader
        self.dataloader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
        )

        # Load or compute compressed embeddings
        self._apply_cached_compression(config, model_field_pairs)

        self.filter()

    def _get_cache_path(self, config, compressed_field_name):
        """Generates the cache path for the stored embeddings."""
        output_dir = os.path.join(config.base_dir, config.data.vector_database_in, compressed_field_name)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{self.split}.pt")

    def _apply_cached_compression(self, config, model_field_pairs):
        """Loads cached embeddings if available; otherwise, computes and caches them."""
        for model, input_field in model_field_pairs:
            compressed_field_name = f'{config.model_name}/{input_field}'
            cache_path = self._get_cache_path(config, compressed_field_name)

            if os.path.exists(cache_path):
                # Load cached embeddings
                print(f"Loading cached embeddings from {cache_path}")
                compressed_values = torch.load(cache_path, weights_only=False)
            else:
                # Compute embeddings and save them
                print(f"Computing embeddings for {input_field}")
                compressed_values = extract_vectors_for_split(config, self.dataloader, model, self.input_keys.index(input_field), [])
                compressed_values = {k: torch.stack(v) for (k, v) in compressed_values.items()}

                torch.save(compressed_values, cache_path)
                print(f"Saved embeddings to {cache_path}")

            self.dataset[compressed_field_name] = list(compressed_values['output'])

            # Update input keys to reflect compressed fields
            self.input_keys[self.input_keys.index(input_field)] = compressed_field_name
