import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from Data.util import EmbeddingDataset
import pytorch_lightning as pl
from Pretraining.util.extract_model_output import extract_vectors_for_split
import numpy as np
import pandas as pd

class CompressedMultimodalDataset(pl.LightningModule):
    def __init__(self, config, dataset, model, split):
        """
        Args:
            config: Configuration object.
            dataset: Pandas DataFrame containing the dataset.
            inference_map: Dictionary defining compression logic.
            split: Dataset split name (e.g., 'train', 'validation', 'test').
            batch_size: Batch size for DataLoader.
        """
        super().__init__()
        self.config = config
        self.split = split
        self.batch_size = config.task.batch_size
        self.model = model.eval()

        self.input_keys = config.data.inputs
        self.output_keys = config.task.outputs

        self.dataset = dataset

        # Set up DataLoader
        self.dataloader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=False,
            # collate_fn=self.custom_collate,
        )

        self.set_vectorized_data()

        # Load or compute compressed embeddings
        self._apply_cached_compression()

        self.dataset.filter()

        self.set_vectorized_data()

    def set_vectorized_data(self):
        self.vectorized_inputs = self.dataset.dataset[self.input_keys].to_dict('records')
        self.vectorized_outputs = self.dataset.dataset[self.output_keys].to_dict('records')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # sample = self.dataset.loc[idx]

        # Load image & existing metadata using the original dataset
        # all_data = {}

        if any([k not in self.dataset.dataset.columns for k in self.input_keys]):
            raise NotImplementedError("Any keys not in metadata must instead query raw dataset")
        
        return self.vectorized_inputs[idx], self.vectorized_outputs[idx]

        # if any([k not in self.dataset.columns for k in input_keys]):
        #     original_dataset_index = self.original_dataset_indexing.index[self.original_dataset_indexing[self.primary_key] == sample[self.primary_key]][0]
        #     raw_data = self.original_dataset.__getitem__(original_dataset_index)
        #     assert raw_data[self.primary_key] == sample[self.primary_key]
        #     all_data.update(raw_data)

        # Combine return data
        # all_data.update(sample.loc[[i for i in sample.index if i not in all_data.keys()]].to_dict())

        # input_row = {k: all_data[k] for k in input_keys}
        # output_row = sample[output_keys].to_dict()

        # return self.dataset.__getitem__(idx, self.input_keys, self.output_keys)

    def _get_cache_path(self, compressed_field_name):
        """Generates the cache path for stored embeddings."""
        output_dir = os.path.join(self.config.base_dir, self.config.data.vector_database_in, compressed_field_name)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{self.split}.pt")

    def _apply_cached_compression(self):
        """Loads cached embeddings if available; otherwise, computes and caches them."""
        for field in self.model.inference_map.keys():
            inference_model = self.model.inference_map[field]
            inference_metadata = self.model.inference_metadata[field]

            if inference_metadata['compress']:
                compressed_field_name = inference_metadata['output_field']
                cache_path = self._get_cache_path(compressed_field_name)

                if os.path.exists(cache_path):
                    print(f"Loading cached embeddings for {field} from {cache_path}")
                    compressed_values = torch.load(cache_path, weights_only=False)

                else:
                    print(f"Computing embeddings for {field}")
                    self.output_keys.append('anon_accession')
                    self.set_vectorized_data()

                    compressed_values = extract_vectors_for_split(self.config, self.dataloader, inference_model, field, ['anon_accession'])
                    self.output_keys.pop(-1)
                    self.set_vectorized_data()

                    def process_values(v):
                        if all(isinstance(x, torch.Tensor) for x in v):
                            return torch.stack(v)  # Stack tensors
                        elif all(isinstance(x, np.ndarray) for x in v):
                            return torch.stack([torch.tensor(x) for x in v])  # Convert NumPy arrays & stack
                        elif all(isinstance(x, str) for x in v):
                            return v  # Keep strings as lists
                        else:
                            return v  # Mixed types, leave as is

                    compressed_values = {k: process_values(v) for k, v in compressed_values.items()}

                    # compressed_values = {k: torch.stack(v) for (k, v) in compressed_values.items()}

                    torch.save(compressed_values, cache_path)
                    print(f"Saved embeddings to {cache_path}")

                stored_embedding_name = "output" if "output" in compressed_values else "vectors"
                compressed_values[compressed_field_name] = list(np.array(torch.Tensor(compressed_values[stored_embedding_name])))
                compressed_values.pop(stored_embedding_name)

                new_data = pd.DataFrame(compressed_values)
                if 'sample_ids' in new_data.columns:
                    new_data['anon_accession'] = new_data['sample_ids']
                
                # self.dataset.dataset[compressed_field_name] = list(np.array(torch.Tensor(compressed_values["output" if "output" in compressed_values else "vectors"])))
                self.dataset.dataset = self.dataset.dataset.merge(pd.DataFrame(compressed_values), how='left', on='anon_accession')

                self.input_keys[self.input_keys.index(field)] = compressed_field_name
                # self.inference_map.pop(field, None)
                # del self.inference_map[field]
                x = 3
    
    @staticmethod
    def custom_collate(batch):
        inputs, outputs = zip(*batch)  # Separate inputs and outputs
        inputs = [list(field) for field in zip(*inputs)]  # Transpose inputs
        outputs = [list(field) for field in zip(*outputs)]  # Transpose outputs
        return inputs, outputs
