import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from Data.util import EmbeddingDataset
import pytorch_lightning as pl
from Pretraining.util.extract_model_output import extract_vectors_for_split
import numpy as np

class CompressedMultimodalDataset(pl.LightningModule):
    def __init__(self, config, dataset, inference_map, split, batch_size=224):
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
        self.batch_size = batch_size
        self.inference_map = inference_map

        self.input_keys = config.data.inputs
        self.output_keys = config.task.outputs

        self.dataset = dataset

        # Set up DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate,
        )
        
        # Load or compute compressed embeddings
        self._apply_cached_compression()

        self.dataset.filter()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx, self.input_keys, self.output_keys)

    def _get_cache_path(self, compressed_field_name):
        """Generates the cache path for stored embeddings."""
        output_dir = os.path.join(self.config.base_dir, self.config.data.vector_database_in, compressed_field_name)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{self.split}.pt")

    def _apply_cached_compression(self):
        """Loads cached embeddings if available; otherwise, computes and caches them."""
        for field, mapping in list(self.inference_map.items()):
            if mapping['compress']:
                compressed_field_name = mapping['output_field']
                cache_path = self._get_cache_path(compressed_field_name)

                if os.path.exists(cache_path):
                    print(f"Loading cached embeddings for {field} from {cache_path}")
                    compressed_values = torch.load(cache_path)
                else:
                    print(f"Computing embeddings for {field}")
                    compressed_values = extract_vectors_for_split(self.config, self.dataloader, mapping['forward_model'], self.input_keys.index(field), [])
                    compressed_values = {k: torch.stack(v) for (k, v) in compressed_values.items()}

                    torch.save(compressed_values, cache_path)
                    print(f"Saved embeddings to {cache_path}")

                self.dataset.dataset[compressed_field_name] = list(np.array(torch.Tensor(compressed_values["output" if "output" in compressed_values else "vectors"])))
                self.input_keys[self.input_keys.index(field)] = compressed_field_name
                self.inference_map.pop(field, None)
                # del self.inference_map[field]
                x = 3
    
    @staticmethod
    def custom_collate(batch):
        inputs, outputs = zip(*batch)  # Separate inputs and outputs
        inputs = [list(field) for field in zip(*inputs)]  # Transpose inputs
        outputs = [list(field) for field in zip(*outputs)]  # Transpose outputs
        return inputs, outputs
