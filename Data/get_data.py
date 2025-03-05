from Data.raw_database.abdominal_ct import create_dataloaders, MultimodalCTDataset
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pandas as pd
import os
import torch
from Data.util import EmbeddingDataset, EmbeddingDataModule, CompressedEmbeddingDataset
from Data.multimodal_dataset import CompressedMultimodalDataset
from util.df import left_merge_new_fields
from Data.eda.ehr_demographics import get_tabular_data
import yaml

import numpy as np

def load_abdominal_ct_tabular_data(config):
    # labs = pd.read_csv('/dataNAS/data/ct_data/ct_ehr/1/labs.csv')
    # demographics = pd.read_csv('/dataNAS/data/ct_data/ct_ehr/1/demographics.csv')
    # encounters = pd.read_csv('/dataNAS/data/ct_data/ct_ehr/1/encounters.csv')
    # crosswalk = pd.read_csv('/dataNAS/data/ct_data/priority_crosswalk_all.csv')
    # radiology_report = pd.read_csv('/dataNAS/data/ct_data/ct_ehr/1/radiology_report.csv')
    # procedures = pd.read_csv('/dataNAS/data/ct_data/ct_ehr/1/procedures.csv')
    return None

def load_abdominal_ct_labels(config):
    labels_path = config.paths.five_year_prognosis_labels
    labels_df = pd.read_csv(labels_path).set_index("anon_accession")
    labels_df = labels_df.drop_duplicates()
    labels_df = labels_df.reset_index()
    # merlin_datasets, _ = create_dataloaders(config)

    phecode_findings_metadata = pd.read_csv(config.paths.abdominal_phecode_labels)
    phecodes = phecode_findings_metadata.columns[1:1693]
    phecode_findings_metadata['phecodes'] = phecode_findings_metadata.loc[:, phecodes].apply(lambda row: np.array(row), axis=1)
    phecode_findings_metadata = phecode_findings_metadata.drop(columns=phecodes)

    diagnosis_metadata = pd.read_csv(config.paths.thirty_diagnosis_labels)

    labels_df = left_merge_new_fields(labels_df, phecode_findings_metadata, 'anon_accession')
    labels_df = left_merge_new_fields(labels_df, diagnosis_metadata, 'anon_accession')

    tabular_data = get_tabular_data(config)
    with open("Data/raw_database/tabular_data.yaml", "r") as file:
        tabular_fields = yaml.safe_load(file)['tabular_fields']
    tabular_data['tabular_fields'] = list(tabular_data[tabular_fields].to_numpy())
    
    labels_df = left_merge_new_fields(labels_df, tabular_data, 'anon_accession')

    return labels_df

def load_merlin_embeddings_and_labels(config, model_field_pairs=[]):
    # Usage:
    vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)
    labels_df = load_abdominal_ct_labels(config)

    datasets = {}
    for split in ["train", "validation", "test"]:
        embedding_data = torch.load(os.path.join(vector_db_path, f"{split}.pt"), weights_only=False)
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

@staticmethod
def custom_collate(batch):
    inputs, outputs = zip(*batch)  # Separate inputs and outputs

    def stack_dicts(items):
        """Convert a list of dicts into a single dict with stacked values."""
        collated = {}

        for key in items[0].keys():
            values = [item[key] for item in items]

            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                if np.issubdtype(values[0].dtype, np.number):  # Ensure array contains only numbers
                    collated[key] = torch.tensor(np.stack(values))
                else:
                    collated[key] = np.array(values)  # Contains non-numeric types, fallback to list
            elif all(isinstance(v, (int, float, bool)) for v in values):  # Only pure numbers
                collated[key] = torch.tensor(values)
            elif all(isinstance(v, (list, np.ndarray, torch.Tensor)) for v in values):
                if all(isinstance(v, (int, float, bool)) for sublist in values for v in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])):  
                    # Ensures nested lists/arrays are numeric before stacking
                    collated[key] = torch.tensor(np.stack(values))
                else:
                    collated[key] = np.array(values)  # Mixed types in lists, fallback
            else:
                collated[key] = np.array(values)  # Mixed types, return as is
                
        return collated

    return stack_dicts(inputs), stack_dicts(outputs)

def get_data(config, specific_data=None, splits=['train', 'validation', 'test'], device=None):
    data_type = config.data.type if not specific_data else specific_data

    if data_type == 'multimodal_abdominal_ct':
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

        from Model.get_model import ModelBuilder
        model = ModelBuilder(config, device=device).get_model()

        # Louis' API
        raw_dl = get_dataloaders(config, dataset_config, include=splits, output_filter=config.task.output_filter)

        datasets, dataloaders = {}, {}
        for split in splits:
            ds_config = config.copy()

            raw_dataset = MultimodalCTDataset(ds_config, labels_df, raw_dl[split], split, autofilter=False)

            # Create dataset which compresses/loads using inference map, and also updates inference map
            datasets[split] = CompressedMultimodalDataset(ds_config, raw_dataset, model, split=split)

            # Create dataloader for dataset
            dataloaders[split] = DataLoader(
                datasets[split], 
                batch_size=config.task.batch_size, 
                shuffle=True, 
                num_workers=config.task.num_workers, 
                pin_memory=False,  # Avoid unnecessary pinning
                # pin_memory=raw_dl[split].pin_memory, 
                collate_fn=custom_collate
            )

        x = 3

    else:
        raise ValueError(f"Unknown data name: {data_type}")

    return datasets, dataloaders
