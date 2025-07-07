from typing import Any, Dict, cast  # <-- 確保 cast 已導入

import torch
import torch.nn as nn
import torch.nn.functional as F

from amp_searcher.models.base import BaseLightningModule
from amp_searcher.models.lightning_module_factory import LightningModuleFactory


@LightningModuleFactory.register("contrastive")
class ContrastiveLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for contrastive learning tasks.

    This module wraps a core `torch.nn.Module` architecture (e.g., an encoder) and handles
    training, validation, and metric calculation for contrastive learning problems.
    It implements the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    """

    def __init__(
        self,
        model_architecture: nn.Module,  # This is typically the encoder/backbone
        temperature: float = 0.07,
        optimizer_params: Dict[str, Any] | None = None,
        scheduler_params: Dict[str, Any] | None = None,
    ):
        super().__init__(optimizer_params, scheduler_params)
        self.save_hyperparameters(ignore=["model_architecture"])
        self.model = model_architecture
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model architecture (encoder).
        """
        return cast(torch.Tensor, self.model(x))

    def _nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Computes the NT-Xent loss.

        Args:
            z_i: Embeddings of the first view of the batch.
            z_j: Embeddings of the second view of the batch.

        Returns:
            The NT-Xent loss.
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        sim_matrix = (
            F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
            / self.temperature
        )

        # Create labels for positive pairs
        labels = torch.arange(batch_size * 2).to(self.device)
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        positive_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True
            positive_mask[i + batch_size, i] = True

        # Remove self-similarity (diagonal)
        logits = sim_matrix[
            ~torch.eye(batch_size * 2, dtype=torch.bool, device=self.device)
        ].view(batch_size * 2, -1)

        # The labels for cross_entropy_loss are 0 for the positive pair
        labels = torch.zeros(batch_size * 2, dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def _step(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        # For contrastive learning, batch is typically (x_i, x_j) where x_i and x_j are two augmented views of the same data point
        x_i, x_j = batch

        # Get embeddings
        z_i = self.model(x_i)
        z_j = self.model(x_j)

        loss: torch.Tensor = self._nt_xent_loss(z_i, z_j)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")
