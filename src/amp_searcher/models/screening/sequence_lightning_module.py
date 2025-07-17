from typing import Any, Dict, cast

import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MeanSquaredError

from amp_searcher.models.base import BaseLightningModule
from amp_searcher.models.architectures.advanced import TransformerEncoder
from amp_searcher.models.lightning_module_factory import LightningModuleFactory


@LightningModuleFactory.register("SequenceScreeningLightningModule")
class SequenceScreeningLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for virtual screening tasks on sequence embeddings.

    This module takes sequence embeddings (e.g., from a language model) as input
    and processes them through a TransformerEncoder for classification or regression.
    """

    def __init__(
        self,
        model_architecture: TransformerEncoder,  # Expects a TransformerEncoder
        task_type: str,
        optimizer_params: Dict[str, Any] | None = None,
        scheduler_params: Dict[str, Any] | None = None,
        output_dim: int = 1,  # Output dimension for the final prediction layer
    ):
        super().__init__(optimizer_params, scheduler_params)
        self.save_hyperparameters(ignore=["model_architecture"])
        self.model = model_architecture
        self.task_type = task_type.lower()
        self.output_dim = output_dim

        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

        # Final prediction layer after the TransformerEncoder
        self.prediction_head = nn.Linear(self.model.embedding_dim, self.output_dim)

        if self.task_type == "classification":
            self.loss_fn: nn.Module = (
                nn.BCEWithLogitsLoss()
            )  # For binary classification
            self.accuracy = Accuracy(task="binary")
            self.f1_score = F1Score(task="binary")
        else:  # regression
            self.loss_fn = nn.MSELoss()
            self.mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TransformerEncoder and prediction head.

        Args:
            x: Input tensor of sequence embeddings (batch_size, sequence_length, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # TransformerEncoder outputs (batch_size, sequence_length, embedding_dim)
        transformer_output = self.model(x)

        # For sequence classification/regression, typically take the output of the first token (CLS token)
        # or average/pool the sequence outputs.
        # For simplicity, let's average the sequence outputs for now.
        pooled_output = torch.mean(
            transformer_output, dim=1
        )  # (batch_size, embedding_dim)

        logits = self.prediction_head(pooled_output)
        return cast(torch.Tensor, logits)

    def _step(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)

        if self.task_type == "classification":
            # Ensure y is float for BCEWithLogitsLoss
            y = y.float().unsqueeze(1)
            loss = self.loss_fn(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            # Ensure the return type of metrics is handled correctly
            acc_val: torch.Tensor = cast(torch.Tensor, self.accuracy(preds, y.long()))
            f1_val: torch.Tensor = cast(torch.Tensor, self.f1_score(preds, y.long()))
            self.log(f"{stage}_accuracy", acc_val, on_step=False, on_epoch=True)
            self.log(f"{stage}_f1_score", f1_val, on_step=False, on_epoch=True)
        else:  # regression
            loss = self.loss_fn(logits.squeeze(1), y.float())
            mse_val: torch.Tensor = cast(
                torch.Tensor, self.mse(logits.squeeze(1), y.float())
            )
            self.log(f"{stage}_mse", mse_val, on_step=False, on_epoch=True)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return cast(torch.Tensor, loss)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")
