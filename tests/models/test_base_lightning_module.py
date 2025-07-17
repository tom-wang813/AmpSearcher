import pytest
import torch
import torch.nn as nn


from amp_searcher.models import BaseLightningModule


def test_base_lightning_module_inheritance():
    """Tests that a concrete implementation must implement abstract methods."""

    class IncompleteModule(BaseLightningModule):
        # Missing training_step and validation_step
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteModule()

    class CompleteModule(BaseLightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

        def validation_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

    # This should not raise an error
    module = CompleteModule()
    assert isinstance(module, BaseLightningModule)


def test_configure_optimizers_default():
    """Tests default optimizer configuration."""

    class TestModule(BaseLightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

        def validation_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

    module = TestModule()
    optimizers = module.configure_optimizers()

    assert isinstance(optimizers, torch.optim.Adam)
    assert optimizers.defaults["lr"] == 1e-3


def test_configure_optimizers_with_scheduler():
    """Tests optimizer configuration with a scheduler."""

    class TestModule(BaseLightningModule):
        def __init__(self):
            super().__init__(
                optimizer_params={"name": "Adam", "params": {"lr": 0.01}},
                scheduler_params={
                    "name": "StepLR",
                    "step_size": 10,
                    "gamma": 0.1,
                },
            )
            self.linear = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

        def validation_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

    module = TestModule()
    optimizers, schedulers = module.configure_optimizers()

    assert isinstance(optimizers[0], torch.optim.Adam)
    assert optimizers[0].defaults["lr"] == 0.01
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.StepLR)


def test_configure_optimizers_reduce_lr_on_plateau():
    """Tests optimizer configuration with ReduceLROnPlateau scheduler."""

    class TestModule(BaseLightningModule):
        def __init__(self):
            super().__init__(
                optimizer_params={"lr": 0.01},
                scheduler_params={
                    "name": "ReduceLROnPlateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 5,
                },
            )
            self.linear = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

        def validation_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

    module = TestModule()
    optimizer_config = module.configure_optimizers()

    assert isinstance(optimizer_config["optimizer"], torch.optim.Adam)
    assert isinstance(
        optimizer_config["lr_scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert optimizer_config["monitor"] == "val_loss"


def test_configure_optimizers_unsupported_scheduler():
    """Tests that an error is raised for unsupported schedulers."""

    class TestModule(BaseLightningModule):
        def __init__(self):
            super().__init__(
                scheduler_params={
                    "name": "UnsupportedScheduler",
                }
            )
            self.linear = nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

        def validation_step(self, batch, batch_idx):
            x, y = batch
            return self.linear(x) - y

    module = TestModule()
    with pytest.raises(
        ValueError, match="No scheduler registered with name 'UnsupportedScheduler'"
    ):
        module.configure_optimizers()
