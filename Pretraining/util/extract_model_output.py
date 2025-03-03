from pytorch_lightning import LightningModule, Trainer, Callback
import torch
import gc
import numpy as np

class EmbeddingExtractor(LightningModule):
    def __init__(self, model, input_field, fields):
        super().__init__()
        self.model = model
        self.input_field = input_field
        self.fields = fields
        self.model.eval()  # Ensure the model is in eval mode

    def forward(self, x):
        return self.model.latent(x)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs = batch[0][self.input_field]
            batch_fields = {field: batch[1][field] for field in self.fields}
            batch_output = self.forward(inputs)
            batch_output_np = batch_output.cpu().detach().numpy()
            
            # Clean up references
            del batch_output, inputs, batch
            gc.collect()
            torch.cuda.empty_cache()
            
            return {"output": batch_output_np, **batch_fields}

class EmbeddingExtractionCallback(Callback):
    def __init__(self, all_fields, fields):
        super().__init__()
        self.all_fields = all_fields
        self.fields = fields

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for key, value in outputs.items():
            if key not in self.all_fields:
                self.all_fields[key] = []
            self.all_fields[key].extend(value)

def extract_vectors_for_split(config, dataloader, model, input_field, fields):
    embedding_model = EmbeddingExtractor(model, input_field, fields)
    all_fields = {}
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[EmbeddingExtractionCallback(all_fields, fields)],
        limit_predict_batches=config.task.max_steps,
    )
    
    trainer.predict(embedding_model, dataloader)

    output_fields = {key: np.stack(all_fields[key]) for key in all_fields}
    return output_fields
