from contrastive_3d.datasets import monai_datalists, monai_transforms, dataset_configs, dataloaders
from contrastive_3d.datasets.dataloaders import CTPersistentDataset
from monai.data import DataLoader
from torch.utils.data import Dataset

import os

class MultimodalCTDataset(Dataset):
    def __init__(self, config, original_dataset, labels_df):
        self.config = config
        self.original_dataset = original_dataset  # Reference to the original dataset
        self.labels_df = labels_df.set_index("anon_accession")  # Fast lookup
        self.data = original_dataset.data  # Original dataset’s list of dicts

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Load image & existing metadata using the original dataset
        sample = self.original_dataset.__getitem__(idx)

        # Extract anon_accession and fetch corresponding metadata (no need for a check)
        anon_accession = self.data[idx]["anon_accession"]
        additional_metadata = self.labels_df.loc[anon_accession].to_dict()

        # Inject new metadata into the sample
        sample.update(additional_metadata)

        return sample  # Return the updated sample (image + metadata)

class FilteredDataset(CTPersistentDataset):
    def __init__(self, config, data, transform, cache_dir, label_names):
        self.config = config

        if label_names:
            data = [{**row, **{k: v for k, v in zip(label_names, row['label'])}} for row in data]

        # filtered_data = data
        # if config.task.output_filter:
        #     filtered_data = [item for item in data if item[config.task.outputs] in config.task.output_filter]

        super().__init__(data, transform, cache_dir)

def get_dataloaders(hydra_config, config, train_files=None, val_files=None, test_files=None, include=('train', 'validation', 'test'), output_filter=None):
    try:
        dataset_config = dataset_configs.get_dataset_config(config["dataset"])
        transforms = dataset_config.transforms
        cache_dir = dataset_config.cache_dir
    except:
        dataset_config = config
        transforms = config["transforms"]
        cache_dir = config["cache_dir"]

    batch_sizes = {
        'train': config["per_device_train_batch_size"],
        'validation': config["per_device_val_batch_size"],
        'test': config["per_device_test_batch_size"]
    }
    
    file_sources = {}
    datasets = {}
    
    for key in include:
        if key == 'train' and train_files is None:
            file_sources[key] = dataset_config.get_datalist("train", fraction_train_data=config["fraction_train_data"], config=config, dataset_config=dataset_config)
        elif key == 'validation' and val_files is None:
            file_sources[key] = dataset_config.get_datalist("val", fraction_train_data=config["fraction_train_data"], config=config, dataset_config=dataset_config)
        elif key == 'test' and test_files is None:
            file_sources[key] = dataset_config.get_datalist("test", fraction_train_data=config["fraction_train_data"], config=config, dataset_config=dataset_config)
        else:
            file_sources[key] = locals().get(f"{key}_files")
    
    for key in include:
        if key in file_sources and key not in datasets:
            print(f"Creating {key} dataset...")
            datasets[key] = FilteredDataset(hydra_config, data=file_sources[key], transform=transforms, cache_dir=cache_dir, label_names=dataset_config.label_names)
    
    num_workers = os.cpu_count()
    dataloaders = {key: DataLoader(datasets[key], batch_size=batch_sizes[key], shuffle=(key == 'train'), num_workers=num_workers) for key in include if key in datasets}
    
    return dataloaders

def create_dataloaders(config, sets=['train', 'validation', 'test']):
    """
    Creates DataLoader objects for the provided datasets using a predefined dataloaders module.

    Args:
        datasets (dict): Dictionary of dataset objects for train/val/test splits.

    Returns:
        tuple: (dataloaders, datasets_dict)
            - dataloaders: Dictionary of DataLoader objects for train/val/test splits.
            - datasets_dict: Dictionary of dataset objects for train/val/test splits from DataLoader.
    """
    from contrastive_3d.datasets import dataloaders
    dataset_config = {
        "dataset": config.data.merlin_dataset_variant,
        # "dataset": "stanford_disease_prediction_all",
        "fraction_train_data": config.data.fraction_train_data,
        "per_device_train_batch_size": config.task.batch_size,
        "per_device_val_batch_size": config.task.batch_size,
        "per_device_test_batch_size": config.task.batch_size,
    }

    dataloaders = get_dataloaders(config, dataset_config, include=sets)

    datasets_dict = {split: dl.dataset for split, dl in dataloaders.items()}

    return datasets_dict, dataloaders
