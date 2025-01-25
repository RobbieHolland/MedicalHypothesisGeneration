import hydra
import torch
from pytorch_lightning import LightningModule, Trainer
from tqdm import tqdm
import os
from pytorch_lightning.callbacks import Callback
from MultimodalPretraining.util.extract_model_output import extract_vectors_for_split

# Experiment to extract embeddings for all splits
@hydra.main(config_path="../../config", config_name="default")
def build_vector_database(config):
    from MultimodalPretraining.model.model import load_model
    from MultimodalPretraining.data.raw_database.dataset import create_dataloaders

    # Load dataset and model
    datasets, dataloaders = create_dataloaders(config)
    model = load_model(config)

    # Extract vectors for each split
    # for split in ["train"]:
    input_field = 'image'
    fields = ['anon_accession']
    
    for split in ["train", "validation", "test"]:
        all_fields = extract_vectors_for_split(config, dataloaders[split], model, input_field, fields)

    # Save the results to a file
    output_dir = os.path.join(config.base_dir, config.data.vector_database_out, config.task.vector_database_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.pt")
    torch.save(all_fields, output_path)
    print(f"Vector database for {split} saved to {output_path}")

# Example usage
if __name__ == "__main__":
    build_vector_database()
