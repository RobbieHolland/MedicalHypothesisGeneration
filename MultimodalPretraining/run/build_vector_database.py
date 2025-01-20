import hydra
import torch
from pytorch_lightning import LightningModule, Trainer
from tqdm import tqdm
import os
from pytorch_lightning.callbacks import Callback

class EmbeddingExtractor(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()  # Ensure the model is in eval mode for embeddings

    def forward(self, x):
        # Forward pass through the model
        return self.model.latent(x)

    def test_step(self, batch, batch_idx):
        # Process the batch
        images = batch['image']
        sample_ids = batch['anon_accession']
        embeddings = self.forward(images)  # Extract embeddings
        return {"sample_ids": sample_ids, "embeddings": embeddings}

    def test_step_end(self, outputs):
        # Optional: You can combine outputs here if needed
        return outputs
    
def extract_vectors_for_split(config, split, dataloader, model):
    # Wrap model in LightningModule
    embedding_model = EmbeddingExtractor(model)

    # Store vectors and sample IDs
    all_vectors = []
    all_sample_ids = []

    # Define a custom callback for embedding extraction
    class EmbeddingExtractionCallback(Callback):
        def __init__(self, all_vectors, all_sample_ids):
            super().__init__()
            self.all_vectors = all_vectors
            self.all_sample_ids = all_sample_ids

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            # Access embeddings and sample IDs from test_step outputs
            embeddings = outputs["embeddings"]  # Extracted in test_step
            sample_ids = outputs["sample_ids"]  # Extracted in test_step

            # Store embeddings and sample IDs
            self.all_vectors.append(embeddings.detach().cpu())
            self.all_sample_ids.extend(sample_ids)

    # Create a Trainer for embedding extraction
    trainer = Trainer(
        accelerator="gpu",  # Use GPU if available
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[EmbeddingExtractionCallback(all_vectors, all_sample_ids)],
        limit_test_batches=config.task.max_steps,
    )

    # Run the embedding extraction for the split
    print(f"Extracting embeddings for {split} split...")
    trainer.test(embedding_model, dataloaders=[dataloader])

    # Concatenate all vectors into a single tensor
    all_vectors = torch.cat(all_vectors, dim=0)

    # Save the results to a file
    output_dir = os.path.join(config.data.vector_database_out, config.task.vector_database_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.pt")
    torch.save({"vectors": all_vectors, "sample_ids": all_sample_ids}, output_path)
    print(f"Vector database for {split} saved to {output_path}")

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
    for split in ["train", "validation", "test"]:
        extract_vectors_for_split(config, split, dataloaders[split], model)

# Example usage
if __name__ == "__main__":
    build_vector_database()
