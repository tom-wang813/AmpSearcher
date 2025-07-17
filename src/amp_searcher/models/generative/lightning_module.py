from typing import Any, Dict, cast

import torch

from amp_searcher.models.base import BaseLightningModule
from amp_searcher.models.lightning_module_factory import LightningModuleFactory
from amp_searcher.models.generative.architectures.vae import VAE


@LightningModuleFactory.register("GenerativeLightningModule")
class GenerativeLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for generative tasks.

    This module wraps a core `torch.nn.Module` architecture (e.g., VAE, GAN) and handles
    training, validation, and metric calculation for generative problems.
    """

    def __init__(
        self,
        model_architecture: VAE,
        latent_dim: int,
        kl_weight: float,  # Added kl_weight here
        optimizer_params: Dict[str, Any] | None = None,
        scheduler_params: Dict[str, Any] | None = None,
    ):
        super().__init__(optimizer_params, scheduler_params)
        self.save_hyperparameters(ignore=["model_architecture"])
        self.model = model_architecture
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generative model architecture.
        """
        return cast(torch.Tensor, self.model(x))

    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generates new feature vectors from the latent space.
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            # Assuming the generative model has a 'decoder' method
            generated_features: torch.Tensor = self.model.decoder(z)
            return generated_features

    def _step(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        # Generative models often have complex loss calculations (e.g., VAE: reconstruction + KL)
        # The `model_architecture` (e.g., VAE) is expected to return the necessary components for loss.
        # For a VAE, this might be (reconstructed_x, mu, log_var)
        # For a GAN, this might involve discriminator outputs

        # For simplicity, we assume the model returns a single loss value for now.
        # More complex generative models will override this or handle loss internally.
        loss = self.model.compute_loss(batch)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")
