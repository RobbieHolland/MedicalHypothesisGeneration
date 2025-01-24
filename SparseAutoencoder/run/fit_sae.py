import hydra
import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import TensorDataset, Dataset, DataLoader
import pytorch_lightning as pl
import os
from SparseAutoencoder.model.sparse_autoencoder import SparseAutoencoder

class TrainSparseAutoencoder(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters()

        self.sae = SparseAutoencoder(config)
        self.config = config
        self.frac_active_list = []
        self.n_training_tokens = 0

    def forward(self, x):
        return self.sae(x)

    def _compute_and_log_metrics(self, sae_out, x, frac_active_list, stage="train"):
        l0 = (sae_out["z_sparse"] > 0).float().sum(-1).mean()
        l2_norm = torch.norm(sae_out["z_sparse"], dim=1).mean()
        l2_norm_in = torch.norm(x, dim=-1)
        l2_norm_out = torch.norm(sae_out["x_hat"], dim=-1)
        l2_norm_ratio = l2_norm_out / (1e-6 + l2_norm_in)
        r_squared = 1 - (self.sae.reconstruction_loss(x, sae_out['x_hat'])) / (self.sae.reconstruction_loss(x, x.mean(0)))

        # Fraction of features active
        act_freq_scores = (sae_out["z_sparse"].abs() > 0).float().mean(0)
        # frac_active_list.append(act_freq_scores)
        # if len(frac_active_list) > self.config.task.feature_sampling_window:
        #     frac_active_in_window = torch.stack(
        #         frac_active_list[-self.config.task.feature_sampling_window :], dim=0
        #     )
        #     feature_sparsity = frac_active_in_window.sum(0) / (
        #         self.config.task.feature_sampling_window * len(x)
        #     )
        # else:
        #     frac_active_in_window = torch.stack(frac_active_list, dim=0)
        #     feature_sparsity = frac_active_in_window.sum(0) / (
        #         len(frac_active_list) * len(x)
        #     )

        wandb.log(
            {
                # f"{stage}/n_training_tokens": self.n_training_tokens,
                f"{stage}/losses/mse_loss": sae_out["mse_loss"].item(),
                f"{stage}/losses/l1_loss": sae_out["l1_loss"].item(),
                f"{stage}/losses/overall_loss": sae_out["loss"].item(),
                f"{stage}/metrics/r^2": r_squared.item(),
                # f"{stage}/z_metrics/l0": l0.item(),
                # f"{stage}/z_metrics/l2": l2_norm.item(),
                # f"{stage}/metrics/l2_ratio": l2_norm_ratio.mean().item(),
                # f"{stage}/sparsity/features_below_1e-5": (act_freq_scores < 1e-5).float().mean().item(),
                f"{stage}/sparsity/n_alive": (act_freq_scores > 0).sum().item(),
                # f"{stage}/sparsity/below_1e-6": (feature_sparsity < 1e-6).float().mean().item(),
                # f"{stage}/sparsity/n_dead_features": (feature_sparsity < self.config.task.dead_feature_threshold).float().mean().item(),
            },
            step=self.global_step,
        )

        # Histogram logging (only needed in training, every N steps)
        if stage == "train" and self.global_step % (self.config.wandb_log_frequency * 100) == 0:
            log_feature_sparsity = act_freq_scores / len(act_freq_scores)
            wandb.log(
                {
                    f"{stage}/plots/feature_density_histogram": wandb.Histogram(
                        log_feature_sparsity.tolist()
                    )
                },
                step=self.global_step,
            )

    def _common_step(self, batch, batch_idx, stage):
        x = batch[0]
        if stage == "train":
            self.sae.set_decoder_norm_to_unit_norm()
        sae_out = self(x)
        if stage == "train":
            self.sae.remove_gradient_parallel_to_decoder_directions()
            self.n_training_tokens += len(x)

        # Decide whether to compute/log metrics
        if self.global_step % self.config.wandb_log_frequency == 0:
            # We keep updating self.frac_active_list only for training
            frac_active_list = self.frac_active_list if stage == "train" else []
            self._compute_and_log_metrics(sae_out, x, frac_active_list, stage=stage)

        return sae_out

    def training_step(self, batch, batch_idx):
        sae_out = self._common_step(batch, batch_idx, stage="train")
        self.log("train_loss", sae_out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return sae_out["loss"]

    def validation_step(self, batch, batch_idx):
        sae_out = self._common_step(batch, batch_idx, stage="val")
        self.log("val_loss", sae_out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return sae_out["loss"]

    def test_step(self, batch, batch_idx):
        sae_out = self._common_step(batch, batch_idx, stage="test")
        self.log("test_loss", sae_out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return sae_out["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.sae.parameters(), lr=self.config.task.lr)

class ActivationsDataset(Dataset):
    def __init__(self, data):
        self.vectors = torch.tensor(data['vectors'], dtype=torch.float32)
        self.sample_ids = data['sample_ids']

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.vectors[idx], self.sample_ids[idx]

class ActivationDataModule(pl.LightningDataModule):
    def __init__(self, config, activations):
        super().__init__()
        self.activations = activations
        self.config = config

    def _create_dataloader(self, dataset_data):
        dataset = ActivationsDataset(dataset_data)
        return DataLoader(dataset, batch_size=self.config.task.batch_size, shuffle=True)

    def train_dataloader(self):
        return self._create_dataloader(self.activations['train'])

    def val_dataloader(self):
        return self._create_dataloader(self.activations['validation'])

    def test_dataloader(self):
        return self._create_dataloader(self.activations['test'])


@hydra.main(config_path="../../config", config_name="default", version_base=None)
def fit_sae(config):
    from util.wandb import init_wandb
    init_wandb(config)

    datasets = {
        "train": torch.load(f"{config.data.vector_database_in}/train.pt"),
        "validation": torch.load(f"{config.data.vector_database_in}/validation.pt"),
        "test": torch.load(f"{config.data.vector_database_in}/test.pt"),
    }

    lightning_module = TrainSparseAutoencoder(config=config)
    data_module = ActivationDataModule(config, datasets)

    check_val_every_n_epoch = None
    val_check_interval = None
    if config.task.validation_interval > len(data_module.train_dataloader()):
        check_val_every_n_epoch = config.task.check_val_every_n_epochs
    else:
        val_check_interval = config.task.validation_interval

    save_path = os.path.join(config.pretrained_model_dir, wandb.run.project, 'SAE', wandb.run.name)
    trainer = pl.Trainer(
        max_steps=config.task.max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=save_path, save_top_k=1, mode='min', monitor="val_loss")],
    )
    trainer.validate(lightning_module, datamodule=data_module)
    trainer.fit(lightning_module, data_module)

    # Run test
    trainer.test(lightning_module, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    fit_sae()
