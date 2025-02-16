import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric, AUROC
from torch.optim import AdamW
import hydra
from Model.trainable_save import TrainableSave
import wandb
import os

class LinearEvaluation(TrainableSave):
    """Multi-label (multi-head) linear evaluation with BCE loss and exponential LR decay."""
    def __init__(self, config, model):
        super().__init__(config)
        self.save_hyperparameters()

        self.config = config
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Ensure at least 2 heads, even for binary classification
        self.num_heads = max(config.task.num_outs, 2)
        self.classifier = nn.Linear(config.data.latent_dim, self.num_heads)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.loss_metrics = {phase: MeanMetric() for phase in ["train", "validation", "test"]}
        self.auc_metrics = {
            phase: AUROC(task="multilabel", num_labels=self.num_heads) for phase in ["train", "validation", "test"]
        }

    def forward(self, x):
        return self.model(x)['prediction']
    
    def _shared_step(self, batch, phase):
        dtype = torch.float16 if str(self.trainer.precision) in ["16", "16-mixed"] else torch.bfloat16 if str(self.precision) in ["bf16", "bf16-mixed"] else torch.float32

        x = {k: torch.Tensor(v).to(dtype) if torch.is_floating_point(v) else v for (k, v) in batch[0].items()}
        y = {k: torch.Tensor(v).to(dtype) if torch.is_floating_point(v) else v for (k, v) in batch[1].items()}
        y = y[self.config.task.outputs[0]]

        if hasattr(y, "as_tensor"):
            y = y.as_tensor()
        
        y = y.to(torch.long)  # `one_hot` requires integer indices
        y = F.one_hot(y, num_classes=2).float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        with torch.inference_mode():
            self.loss_metrics[phase].update(loss.cpu())
            self.auc_metrics[phase].update(torch.sigmoid(logits.detach().cpu()), y.cpu().int())

        # Log loss per step
        self.log(f"{phase}_loss_step", loss, prog_bar=True, logger=True, batch_size=y.shape[0])
        wandb.log({f"{phase}_loss_step": loss.item()}, step=self.global_step)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "validation")
        # self.log(f"validation_auc_epoch", auc_val, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self._epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._epoch_metrics("validation")

    def on_test_epoch_end(self):
        self._epoch_metrics("test")

    def _epoch_metrics(self, phase):
        epoch_loss = self.loss_metrics[phase].compute()
        auc_val = self.auc_metrics[phase].compute()

        # Use self.log() for ModelCheckpoint compatibility
        self.log(f"{phase}_loss_epoch", epoch_loss, prog_bar=True, logger=True)
        self.log(f"{phase}_auc_epoch", auc_val, prog_bar=True, logger=True)

        # Keep wandb logging
        wandb.log({f"{phase}_loss_epoch": epoch_loss.item()}, step=self.global_step)
        wandb.log({f"{phase}_auc_epoch": auc_val.item()}, step=self.global_step)

        self.loss_metrics[phase].reset()
        self.auc_metrics[phase].reset()

        return auc_val

    def configure_optimizers(self):
        optimizer = AdamW(self.classifier.parameters(), lr=self.config.task.lr)
        return optimizer

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def test_dataloader(config):
    from pytorch_lightning import Trainer
    from util.wandb import init_wandb
    init_wandb(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load dataloaders
    from Data.get_data import get_data
    datasets, dataloaders = get_data(config)
    # datasets, dataloaders = get_data(config, splits=['validation'])

    batch = next(iter(dataloaders['train']))
    
    from tqdm import tqdm
    for i in tqdm(range(20)):
        batch = next(iter(dataloaders['train']))

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def run(config):
    from pytorch_lightning import Trainer
    from util.wandb import init_wandb
    init_wandb(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load dataloaders
    from Data.get_data import get_data
    datasets, dataloaders = get_data(config)
    # datasets, dataloaders = get_data(config, splits=['validation'])
    
    # Load model to evaluate
    from Model.get_model import get_model
    model = get_model(config, device=device)
    
    keys_to_remove = [k for k in model.inference_map.keys() if model.inference_metadata[k]['compress']]

    for k in keys_to_remove:
        del model.inference_map[k]
        del model.inference_metadata[k]  # Ensure metadata is also removed

    linear_eval = LinearEvaluation(config, model)

    multimodal_identity = get_model(config, specific_model_name='identity')

    class ClassifierModel(nn.Module):
        def __init__(self, multimodal_identity, linear_eval):
            super().__init__()
            self.multimodal_identity = multimodal_identity
            self.classifier = linear_eval.classifier

        def forward(self, x):
            z = self.multimodal_identity(x)
            return self.classifier(z)

    model.update_inference_map(('merlin/image', 'merlin/findings'), ClassifierModel(multimodal_identity, linear_eval), 'prediction', False)

    from util.lightning import validation_check_intervals
    val_check_interval, check_val_every_n_epoch = validation_check_intervals(config, len(dataloaders['train']))

    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else 'no_sweep'
    save_path = os.path.join(config.pretrained_model_dir, wandb.run.project, wandb.run.group, sweep_id, config.data.name, config.task.outputs[0], wandb.run.name)
    trainer = Trainer(
        max_steps=config.task.max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=save_path, save_top_k=1, mode='max', monitor="validation_auc_epoch")],
    )
    trainer.validate(linear_eval, dataloaders['validation'])
    trainer.fit(linear_eval, dataloaders['train'], dataloaders['validation'])

    best_model_path = trainer.checkpoint_callback.best_model_path

    # Load the best model
    print(f'Testing {best_model_path}')
    best_model = LinearEvaluation.load_from_checkpoint(best_model_path)

    trainer.test(best_model, dataloaders['test'])

    wandb.finish()

if __name__ == "__main__":
    run()
    # test_dataloader()