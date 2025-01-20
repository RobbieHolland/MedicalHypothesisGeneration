# DATASET_MAP = {
#     "abdominal_ct": AbdominalCT,
# }

# def load_dataset(name, config):
#     if name not in DATASET_MAP:
#         raise ValueError(f"Dataset {name} is not supported.")
#     return DATASET_MAP[name](config)

def create_dataloaders(config):
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
        "dataset": "stanford",
        # "dataset": "stanford_disease_prediction_all",
        "fraction_train_data": 1.0,
        "per_device_train_batch_size": config.task.batch_size,
        "per_device_val_batch_size": config.task.batch_size,
        "per_device_test_batch_size": config.task.batch_size,
    }

    train_dataloader, validation_dataloader, test_dataloader = dataloaders.get_dataloaders(dataset_config)

    dataloaders = {
        "train": train_dataloader,
        "validation": validation_dataloader,
        "test": test_dataloader,
    }

    datasets_dict = {split: dl.dataset for split, dl in dataloaders.items()}

    return datasets_dict, dataloaders
