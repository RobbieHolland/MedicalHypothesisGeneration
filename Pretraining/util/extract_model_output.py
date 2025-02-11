from pytorch_lightning import LightningModule, Trainer, Callback
import torch

class EmbeddingExtractor(LightningModule):
    def __init__(self, model, input_field, fields):
        super().__init__()
        self.model = model
        self.input_field = input_field
        self.fields = fields
        self.model.eval()  # Ensure the model is in eval mode for embeddings

    def forward(self, x):
        # Forward pass through the model
        return self.model.latent(x)

    def test_step(self, batch, batch_idx):
        # Process the batch
        inputs = batch[0][self.input_field]
        batch_fields = {field: batch[1][field] for field in self.fields}
        batch_output = self.forward(inputs)  # Extract embeddings
        return {"output": batch_output.detach().cpu(), **batch_fields}

class EmbeddingExtractionCallback(Callback):
    def __init__(self, all_fields, fields):
        super().__init__()
        self.all_fields = all_fields
        self.fields = fields

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Collect embeddings and associated field values
        for key, value in outputs.items():
            if key not in self.all_fields:
                self.all_fields[key] = []
            self.all_fields[key].extend(value)

def extract_vectors_for_split(config, dataloader, model, input_field, fields):
    # Wrap model in LightningModule
    embedding_model = EmbeddingExtractor(model, input_field, fields)

    # Initialize field storage
    all_fields = {}

    # Create a Trainer for embedding extraction
    trainer = Trainer(
        accelerator="gpu",  # Use GPU if available
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[EmbeddingExtractionCallback(all_fields, fields)],
        limit_test_batches=config.task.max_steps,
    )

    # Run the test phase to extract embeddings
    trainer.test(embedding_model, dataloader)

    return all_fields
