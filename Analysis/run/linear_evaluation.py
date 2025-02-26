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

class ClassifierModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = max(config.task.num_outs, 2)
        self.classifier = nn.Linear(config.data.latent_dim, self.num_heads)

    def forward(self, z):
        z = torch.cat(z)
        if not z.is_floating_point():
            z = z.float()
        return self.classifier(z)

class LinearEvaluation(TrainableSave):
    """Multi-label (multi-head) linear evaluation with BCE loss and exponential LR decay."""
    def __init__(self, config, model):
        super().__init__(config)
        self.save_hyperparameters()

        self.dev = torch.device('cuda:0')

        self.config = config
        self.model = model

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()

        # Ensure at least 2 heads, even for binary classification
        self.num_heads = max(config.task.num_outs, 2)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.loss_metrics = {phase: MeanMetric() for phase in ["train", "validation", "test"]}
        self.auc_metrics = {
            phase: AUROC(task="multilabel", num_labels=self.num_heads) for phase in ["train", "validation", "test"]
        }
        self.max_validation_auc = 0

    def forward(self, x):
        return self.model(x)['classifier/prediction']
    
    def _shared_step(self, batch, phase):
        # dtype = torch.float16 if str(self.trainer.precision) in ["16", "16-mixed"] else torch.bfloat16 if str(self.precision) in ["bf16", "bf16-mixed"] else torch.float32

        x = batch[0]  # expecting a dict of tensors
        y = batch[1]


        # x = torch.Tensor(x).squeeze(0).to(self.dev)
        # y = torch.Tensor(y).squeeze(0).to(self.dev)

        # x = {'merlin/image': x}
        # x = {k: torch.Tensor(v).to(dtype) if torch.is_floating_point(v) else v for (k, v) in batch[0].items()}
        # y = {k: torch.Tensor(v).to(dtype) if torch.is_floating_point(v) else v for (k, v) in batch[1].items()}
        y = y[self.config.task.outputs[0]]

        # if hasattr(y, "as_tensor"):
        #     y = y.as_tensor()
        
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
        auc_val = self._epoch_metrics("validation")
        self.max_validation_auc = max(self.max_validation_auc, auc_val.item())
        self.log(f"{'validation'}_max_auc_epoch", self.max_validation_auc, prog_bar=True, logger=True)
        wandb.log({f"{'validation'}_max_auc_epoch": self.max_validation_auc}, step=self.global_step)

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
        optimizer = AdamW(self.model.parameters(), lr=self.config.task.lr)
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
    # device = torch.device('cpu')

    # Load dataloaders
    from Data.get_data import get_data
    datasets, dataloaders = get_data(config, device=device, splits=['test', 'validation', 'train'])
    # datasets, dataloaders = get_data(config, splits=['validation'])
    
    # Load model to evaluate
    from Model.get_model import ModelBuilder

    model = ModelBuilder(config).get_model()
    model.remove_compressed_entries()
    
    linear_eval = LinearEvaluation(config, model)

    classifier_module = ClassifierModel(config)
    for param in classifier_module.parameters():
        param.requires_grad = True

    classifier_module = classifier_module.to(device)

    model.update_inference_map('identity/multimodal_embedding', 'classifier', classifier_module, 'prediction', False)
    # model.update_inference_map('merlin/image', classifier_module, 'prediction', False)

    from util.lightning import validation_check_intervals
    val_check_interval, check_val_every_n_epoch = validation_check_intervals(config, len(dataloaders['train']))

    from util.path_util import linear_eval_path
    save_path = os.path.join(linear_eval_path(config), wandb.run.name)
    print(f'Checkpoints will be saved to {save_path}')

    trainer = Trainer(
        max_steps=config.task.max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu",
        devices=1,
        # precision=16,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=save_path, save_top_k=1, mode='max', monitor="validation_auc_epoch")],
    )
    trainer.validate(linear_eval, dataloaders['validation'])
    trainer.fit(linear_eval, dataloaders['train'], dataloaders['validation'])

    best_model_path = trainer.checkpoint_callback.best_model_path

    # Load the best model
    print(f'Testing {best_model_path}')
    best_model = LinearEvaluation.load_from_checkpoint(best_model_path, model=model)

    trainer.test(best_model, dataloaders['test'])

    wandb.finish()

if __name__ == "__main__":
    run()
    # test_dataloader()