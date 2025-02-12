from Data.raw_database.abdominal_ct import create_dataloaders, MultimodalCTDataset
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import os
import torch
from Data.util import EmbeddingDataset, EmbeddingDataModule, CompressedEmbeddingDataset

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

def load_abdominal_ct_labels(config):
    labels_path = config.paths.five_year_prognosis_labels
    labels_df = pd.read_csv(labels_path).set_index("anon_accession")
    labels_df = labels_df.drop_duplicates()
    labels_df = labels_df.reset_index()
    # merlin_datasets, _ = create_dataloaders(config)

    phecode_findings_metadata = pd.read_csv(config.paths.abdominal_phecode_labels)
    labels_df = labels_df.merge(phecode_findings_metadata, on='anon_accession', how='left')

    return labels_df

def load_merlin_embeddings_and_labels(config, model_field_pairs=[]):
    # Usage:
    vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)
    labels_df = load_abdominal_ct_labels(config)

    datasets = {}
    for split in ["train", "validation", "test"]:
        embedding_data = torch.load(os.path.join(vector_db_path, f"{split}.pt"))
        df = pd.DataFrame({
            "sample_ids": embedding_data['sample_ids'],
            "vectors": list(embedding_data["vectors"].detach().cpu().numpy())
        })
        merged_df = df.merge(labels_df, left_on="sample_ids", right_on="anon_accession", how="left")
        merged_df = merged_df.reset_index(drop=True)

        # Create datasets for each split
        datasets[split] = CompressedEmbeddingDataset(config.copy(), merged_df, model_field_pairs, split=split)

        # datasets[split] = EmbeddingDataset(config, merged_df)
    return datasets


def get_data(config, specific_data=None, splits=['train', 'validation', 'test']):
    data_type = config.data.type if not specific_data else specific_data

    if data_type == 'abdominal_ct':
        labels_df = load_abdominal_ct_labels(config)

        from contrastive_3d.datasets import dataloaders
        from Data.raw_database.abdominal_ct import get_dataloaders
        dataset_config = {
            "dataset": config.data.merlin_dataset_variant,
            # "dataset": "stanford_disease_prediction_all",
            "fraction_train_data": config.data.fraction_train_data,
            "per_device_train_batch_size": config.task.batch_size,
            "per_device_val_batch_size": config.task.batch_size,
            "per_device_test_batch_size": config.task.batch_size,
        }

        # Louis' API
        raw_dl = get_dataloaders(config, dataset_config, include=splits, output_filter=config.task.output_filter)

        datasets, dataloaders = {}, {}
        for split in splits:
            datasets[split] = MultimodalCTDataset(config, raw_dl[split].dataset, labels_df)
            dataloaders[split] = DataLoader(datasets[split], batch_size=raw_dl[split].batch_size, shuffle=raw_dl[split].sampler is not None, num_workers=raw_dl[split].num_workers, pin_memory=raw_dl[split].pin_memory)
            
        x = 3

    elif data_type in 'abdominal_ct_embeddings':
        datasets = load_merlin_embeddings_and_labels(config)

        activation_data_module = EmbeddingDataModule(config, datasets)
        dataloaders = {
            'train': activation_data_module.train_dataloader(),
            'validation': activation_data_module.val_dataloader(),
            'test': activation_data_module.test_dataloader(),
        }

    elif data_type == 'abdominal_ct_text_embeddings':
        config = config.copy()

        from Model.get_model import get_model
        config.model_name = 'merlin'

        model = get_model(config)
        model.latent = lambda x: model.model.encode_text(x)

        model_field_compression_pairs = [(model, 'findings')]

        datasets = load_merlin_embeddings_and_labels(config, model_field_compression_pairs)

        activation_data_module = EmbeddingDataModule(config, datasets)
        dataloaders = {
            'train': activation_data_module.train_dataloader(),
            'validation': activation_data_module.val_dataloader(),
            'test': activation_data_module.test_dataloader(),
        }
        x = 3

    elif data_type == 'phecodes':
        labels_df = pd.read_csv(config.paths.abdominal_phecode_labels)
        prognosis_labels_df = pd.read_csv(config.paths.five_year_prognosis_labels).set_index("anon_accession")
        vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)

        labels_df['phecodes'] = [np.array(x) for x in labels_df[config.data.inputs].to_numpy()]
        config.data.inputs = ['phecodes']

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
        raise ValueError(f"Unknown data name: {data_type}")

    return datasets, dataloaders
