import gzip
import os
import pickle
from typing import Any

import einops
import torch
import pytorch_lightning as pl
from torch import nn

class SparseAutoencoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config.sparse_autoencoder.n_tokens, self.config.sparse_autoencoder.n_concepts, dtype=self.dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(self.config.sparse_autoencoder.n_concepts, dtype=self.dtype))

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config.sparse_autoencoder.n_concepts, self.config.sparse_autoencoder.n_tokens, dtype=self.dtype)
            )
        )
        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_enc = nn.Parameter(torch.zeros(self.config.sparse_autoencoder.n_concepts, dtype=self.dtype))
        self.b_pre = nn.Parameter(torch.zeros(self.config.sparse_autoencoder.n_tokens, dtype=self.dtype))
        self.top_K = 100

        self.reconstruction_loss = torch.nn.MSELoss()

    def forward_topk_sae(self, x):
        # Pre-activation
        z = (x - self.b_pre) @ self.W_enc + self.b_enc  # shape: [batch_size, hidden_dim]
        
        # TopK operator: zero out all but the K largest elements per row
        topk_vals, topk_idx = torch.topk(z, k=self.top_K, dim=1)
        z_sparse = torch.zeros_like(z).scatter_(1, topk_idx, topk_vals)
        # z_sparse = torch.relu(z)
        
        # Reconstruction
        x_hat = z_sparse @ self.W_dec + self.b_pre  # shape: [batch_size, in_dim]
        
        return z, z_sparse, x_hat

    def forward(self, x: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None):
        x = x.to(self.dtype)
        # sae_in = x - self.b_pre
        # z = einops.einsum(
        #     sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        # ) + self.b_enc
        # z_sparse = torch.relu(z)
        # x_hat = einops.einsum(
        #     z_sparse, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
        # ) + self.b_pre

        z, z_sparse, x_hat = self.forward_topk_sae(x)

        # x_centred = x - x.mean(dim=0, keepdim=True)
        # mse_loss = (
        #     (x_hat - x.float()).pow(2)
        #     / (x_centred.pow(2).sum(dim=-1, keepdim=True).sqrt())
        # ).mean()

        mse_loss = self.reconstruction_loss(x_hat, x)

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        if (
            self.config.task.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            residual = x - x_hat
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            z_sparse_dead_neurons_only = torch.exp(z[:, dead_neuron_mask])
            ghost_out = z_sparse_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            mse_loss_ghost_resid = (
                (ghost_out - residual.detach().float()).pow(2)
                / (residual_centred.detach().pow(2).sum(dim=-1, keepdim=True).sqrt())
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
            mse_loss_ghost_resid = mse_loss_ghost_resid.mean()

        sparsity = torch.norm(z_sparse, p=self.config.task.lp_norm, dim=1).mean()
        l1_loss = self.config.task.sparsity_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid
        # loss = mse_loss + l1_loss

        return_dict = {
            "x_hat": x_hat,
            "z": z,
            "z_sparse": z_sparse,
            "loss": loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            # "ghost_grad_loss": mse_loss_ghost_resid,
        }
        return return_dict

    def training_step(self, batch, batch_idx):
        x = batch[0]
        self.set_decoder_norm_to_unit_norm()
        out = self(x)
        self.remove_gradient_parallel_to_decoder_directions()
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return out["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.train.lr)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        if self.W_dec.grad is None:
            return
        parallel_component = einops.einsum(
            self.W_dec.grad, self.W_dec.data, "d_sae d_in, d_sae d_in -> d_sae"
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component, self.W_dec.data, "d_sae, d_sae d_in -> d_sae d_in"
        )
