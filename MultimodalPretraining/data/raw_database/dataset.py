# DATASET_MAP = {
#     "abdominal_ct": AbdominalCT,
# }

# def load_dataset(name, config):
#     if name not in DATASET_MAP:
#         raise ValueError(f"Dataset {name} is not supported.")
#     return DATASET_MAP[name](config)
from contrastive_3d.datasets import monai_datalists, monai_transforms, dataset_configs, dataloaders
from contrastive_3d.datasets.dataloaders import CTPersistentDataset
from monai.data import DataLoader

def get_dataloaders(config, train_files=None, val_files=None, test_files=None, include=('train', 'validation', 'test')):
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
            datasets[key] = CTPersistentDataset(data=file_sources[key], transform=transforms, cache_dir=cache_dir)
    
    dataloaders = {key: DataLoader(datasets[key], batch_size=batch_sizes[key], shuffle=(key == 'train'), num_workers=0) for key in include if key in datasets}
    
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
        "dataset": "stanford",
        # "dataset": "stanford_disease_prediction_all",
        "fraction_train_data": config.data.fraction_train_data,
        "per_device_train_batch_size": config.task.batch_size,
        "per_device_val_batch_size": config.task.batch_size,
        "per_device_test_batch_size": config.task.batch_size,
    }

    dataloaders = get_dataloaders(dataset_config, include=sets)

    datasets_dict = {split: dl.dataset for split, dl in dataloaders.items()}

    return datasets_dict, dataloaders
