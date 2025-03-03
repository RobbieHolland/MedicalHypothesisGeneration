from contrastive_3d.datasets import monai_datalists, monai_transforms, dataloaders
from monai.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from Data.raw_database.dataset_configs import  get_dataset_config

import os

import torch
import sys
import monai
from monai.data import DataLoader
from torch.utils.data import Subset
import collections.abc
from typing import Sequence, Union
import random
from copy import copy, deepcopy
import shutil
import tempfile
from pathlib import Path
from torch.utils.data import RandomSampler

from monai.utils import MAX_SEED, convert_to_tensor, get_seed, look_up_option, min_version, optional_import
from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing

from contrastive_3d.datasets import dataset_configs
from contrastive_3d.utils import split_reports

REPORT_GENERATION = False
    

class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform, cache_dir=None):
        super().__init__(data=data, transform=transform, cache_dir=cache_dir)
        
        print(f"Size of dataset: {self.__len__()}\n")

    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)
        image_path = item_transformed.get('image')
        image_data = {"image": item_transformed.get('image')}  # Assuming the image data is under the 'image' key

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode("utf-8")  # Hash based on image data
            # data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            # print("Cache hit for", image_data)
            # print("Cache dir", self.cache_dir)
            # sys.stdout.flush()
            cached_image = torch.load(hashfile, weights_only=False)
            _item_transformed['image'] = cached_image  # Update item_transformed with cached image
            return _item_transformed

        # If not cached, apply pre-transforms to the image and cache it
        _image_transformed = self._pre_transform(image_data)['image']
        _item_transformed['image'] = _image_transformed  # Update item_transformed with transformed image
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_image_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(str(temp_hash_file), hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed
    
    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)

class MultimodalCTDataset(Dataset):
    def __init__(self, config, labels_df, raw_dl, split, autofilter=True):
        self.config = config
        self.raw_dl = raw_dl
        self.split = split

        # Temporary other dataset
        # original_dataset = raw_dl[split].dataset
        vector_db_path = os.path.join(config.base_dir, config.data.vector_database_in)
        embedding_data = torch.load(os.path.join(vector_db_path, f"{split}.pt"), weights_only=False)
        df = pd.DataFrame({
            "sample_ids": embedding_data['sample_ids'],
            "vectors": list(embedding_data["vectors"].detach().cpu().numpy())
        })
        original_dataset = df
        original_dataset['anon_accession'] = original_dataset['sample_ids']

        self.original_dataset = original_dataset.merge(pd.DataFrame(self.raw_dl.dataset.data), how='left', on='anon_accession')
        self.original_dataset['image_path'] = self.original_dataset['image']

        self.primary_key = 'anon_accession'
        # self.original_dataset_indexing = pd.DataFrame([l[self.primary_key] for l in self.original_dataset.data], columns=[self.primary_key])

        # self.dataset = pd.DataFrame(self.original_dataset.data)
        self.dataset = self.original_dataset
        self.dataset = self.dataset.merge(labels_df[[self.primary_key] + [col for col in labels_df if col not in self.dataset.columns]], on=self.primary_key, how='left')

        if autofilter:
            self.filter()

    def __len__(self):
        return len(self.dataset)
    
    # def __getitem__(self, idx, input_keys, output_keys):
    #     return self.input_row[idx], self.output_row[idx]

    def __getitem__(self, idx, input_keys, output_keys):
        # sample = self.dataset.loc[idx]

        # Load image & existing metadata using the original dataset
        # all_data = {}

        # if any([k not in self.dataset.columns for k in input_keys]):
        #     original_dataset_index = self.original_dataset_indexing.index[self.original_dataset_indexing[self.primary_key] == sample[self.primary_key]][0]
        #     raw_data = self.original_dataset.__getitem__(original_dataset_index)
        #     assert raw_data[self.primary_key] == sample[self.primary_key]
        #     all_data.update(raw_data)

        # Combine return data
        # all_data.update(sample.loc[[i for i in sample.index if i not in all_data.keys()]].to_dict())

        # input_row = {k: all_data[k] for k in input_keys}
        # output_row = sample[output_keys].to_dict()
        
        return self.dataset.loc[idx, input_keys].tolist(), self.dataset.loc[idx, output_keys].tolist()
        # return input_row, output_row

    def filter(self):
        if self.config.task.output_filter:
            inclusion_mask = self.dataset[self.config.task.outputs].isin(self.config.task.output_filter).all(axis=1)
            self.dataset = self.dataset.loc[inclusion_mask].reset_index(drop=True)
            # self.original_dataset = self.original_dataset.loc[inclusion_mask].reset_index(drop=True)

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
        dataset_config = get_dataset_config(config["dataset"])
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
    
    num_workers = hydra_config.task.num_workers
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
