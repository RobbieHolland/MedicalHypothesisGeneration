from Data.raw_database.abdominal_ct import create_dataloaders
import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pytorch_lightning as pl

class ActivationsDataset(Dataset):
    def __init__(self, data):
        self.vectors = torch.tensor(data['vectors'], dtype=torch.float32)
        self.sample_ids = data['sample_ids']

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.vectors[idx], self.sample_ids[idx]

class ActivationDataModule(pl.LightningDataModule):
    def __init__(self, config, activations):
        super().__init__()
        self.activations = activations
        self.config = config

    def _create_dataloader(self, dataset_data):
        dataset = ActivationsDataset(dataset_data)
        return DataLoader(dataset, batch_size=self.config.task.batch_size, shuffle=True)

    def train_dataloader(self):
        return self._create_dataloader(self.activations['train'])

    def val_dataloader(self):
        return self._create_dataloader(self.activations['validation'])

    def test_dataloader(self):
        return self._create_dataloader(self.activations['test'])

def get_data(config, device=None):
    data_name = config.data.name

    if data_name == 'abdominal_ct':
        datasets, dataloaders = create_dataloaders(config)
    elif data_name == 'merlin_embeddings':
        abdominal_datasets, _ = create_dataloaders(config)

        vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)
        datasets = {
            "train": torch.load(f"{vector_db_path}/train.pt"),
            "validation": torch.load(f"{vector_db_path}/validation.pt"),
            "test": torch.load(f"{vector_db_path}/test.pt"),
        }
    
        activation_data_module = ActivationDataModule(config, datasets)
        dataloaders = {
            'train': activation_data_module.train_dataloader(),
            'validation': activation_data_module.val_dataloader(),
            'test': activation_data_module.test_dataloader(),
        }

    elif data_name == 'phecodes':
        pass
        # datasets, dataloaders = create_dataloaders(config)
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    return datasets, dataloaders
