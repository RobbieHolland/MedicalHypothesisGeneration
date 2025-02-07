import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric, AUROC
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import hydra
from Model.trainable_save import TrainableSave
import wandb

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
        self.classifier = nn.Linear(config.model.latent_dim, self.num_heads)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.loss_metrics = {phase: MeanMetric() for phase in ["train", "validation", "test"]}
        self.auc_metrics = {
            phase: AUROC(task="multilabel", num_labels=self.num_heads) for phase in ["train", "validation", "test"]
        }

    def forward(self, x):
        with torch.no_grad():
            feats = self.model.latent(x)
        return self.classifier(feats)
    
    def _shared_step(self, batch, phase):
        x = {k: batch[k].as_tensor() if hasattr(batch[k], "as_tensor") else batch[k] for k in self.config.data.inputs}
        y = batch[self.config.task.outputs]

        # Ensure y is a standard PyTorch tensor
        if hasattr(y, "as_tensor"):
            y = y.as_tensor()
        
        y = y.to(torch.long)  # `one_hot` requires integer indices

        # Convert y to one-hot encoding with 2 classes (batch_size, 2)
        y = F.one_hot(y, num_classes=2).float()

        logits = self.forward(x)
        loss = self.criterion(logits, y)  # Now y and logits have the same shape

        with torch.inference_mode():
            self.loss_metrics[phase].update(torch.tensor(loss.item(), device='cpu'))

        # self.auc_metrics[phase].update(torch.sigmoid(logits.detach().cpu()), y.cpu().int())

        wandb.log({f"{phase}_loss_step": loss.item()}, step=self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "validation")

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
        # auc_val = self.auc_metrics[phase].compute()
        wandb.log({f"{phase}_loss_epoch": epoch_loss.item()}, step=self.global_step)
        # wandb.log(f"{phase}_auc_epoch", auc_val, prog_bar=True)
        self.loss_metrics[phase].reset()
        # self.auc_metrics[phase].reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.classifier.parameters(), lr=self.config.task.lr)
        # scheduler = ExponentialLR(optimizer, gamma=self.config.task.gamma)
        return optimizer

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def run(config):
    from pytorch_lightning import Trainer
    from util.wandb import init_wandb
    init_wandb(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model to evaluate
    from Model.get_model import get_model
    model = get_model(config, device)

    # Load dataloaders
    from Data.get_data import get_data
    datasets, dataloaders = get_data(config)
    
    linear_eval = LinearEvaluation(config, model)

    trainer = Trainer(
        max_steps=config.task.max_steps,
        accelerator="gpu",
        devices=1,
        precision=16,
    )
    trainer.validate(linear_eval, dataloaders['validation'])
    trainer.fit(linear_eval, dataloaders['train'], dataloaders['validation'])

    trainer.test(linear_eval, dataloaders['test'])

    wandb.finish()

if __name__ == "__main__":
    run()
