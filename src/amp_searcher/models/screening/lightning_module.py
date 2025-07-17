from typing import Any, Dict, cast

import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MeanSquaredError

from amp_searcher.models.base import BaseLightningModule
from amp_searcher.models.lightning_module_factory import LightningModuleFactory


@LightningModuleFactory.register("ScreeningLightningModule")
class ScreeningLightningModule(BaseLightningModule):
    """
    PyTorch Lightning module for virtual screening tasks (classification or regression).

    This module wraps a core `torch.nn.Module` architecture and handles
    training, validation, and metric calculation for screening problems.
    """

    def __init__(
        self,
        model_architecture: nn.Module,
        task_type: str,
        optimizer_params: Dict[str, Any] | None = None,
        scheduler_params: Dict[str, Any] | None = None,
    ):
        super().__init__(optimizer_params, scheduler_params)
        self.save_hyperparameters(ignore=["model_architecture"])
        self.model = model_architecture
        self.task_type = task_type.lower()

        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")

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
        Forward pass through the model architecture.
        """
        return cast(torch.Tensor, self.model(x))

    def _step(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)

        if self.task_type == "classification":
            # Ensure y is float for BCEWithLogitsLoss
            y = y.float().unsqueeze(1)
            loss = self.loss_fn(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            # Ensure the return type of metrics is handled correctly
            acc_val: torch.Tensor = self.accuracy(preds, y.long())
            f1_val: torch.Tensor = self.f1_score(preds, y.long())
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
