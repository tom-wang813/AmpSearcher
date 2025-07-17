from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule

from amp_searcher.training.scheduler_factory import SchedulerFactory


class BaseLightningModule(LightningModule, ABC):
    """
    Abstract base class for all PyTorch Lightning models in AmpSearcher.

    This class provides a common interface for model initialization, optimizer
    configuration, and abstract methods for training and validation steps.
    """

    def __init__(
        self,
        optimizer_params: Dict[str, Any] | None = None,
        scheduler_params: Dict[str, Any] | None = None,
    ):
        super().__init__()
        self.optimizer_params = (
            optimizer_params if optimizer_params is not None else {"lr": 1e-3}
        )
        self.scheduler_params = scheduler_params

        # Save hyperparameters to checkpoint
        self.save_hyperparameters(ignore=["model_architecture"])

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Abstract method for the training step.
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Abstract method for the validation step.
        """
        pass

    def configure_optimizers(self) -> Any:
        """
        Configures the optimizer and optionally a learning rate scheduler.
        """
        optimizer_name = self.optimizer_params.pop("name", "Adam")  # Pop 'name' key
        optimizer_params_for_adam = self.optimizer_params.get(
            "params", {}
        )  # Get actual Adam params

        optimizer = getattr(optim, optimizer_name)(
            self.parameters(), **optimizer_params_for_adam
        )
        if self.scheduler_params:
            scheduler_name = self.scheduler_params.pop("name")
            scheduler = SchedulerFactory.build_scheduler(
                optimizer, scheduler_name, **self.scheduler_params
            )

            # If the scheduler is ReduceLROnPlateau, build_scheduler returns a dict
            if isinstance(scheduler, dict):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler["scheduler"],
                    "monitor": scheduler["monitor"],
                }
            else:
                return [optimizer], [scheduler]
        return optimizer
