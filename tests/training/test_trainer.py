import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from amp_searcher.featurizers import PhysicochemicalFeaturizer

from amp_searcher.models.contrastive import ContrastiveLightningModule
from amp_searcher.models.generative import GenerativeLightningModule
from amp_searcher.models.screening import ScreeningLightningModule
from amp_searcher.models.architectures.feed_forward_nn import FeedForwardNeuralNetwork
from amp_searcher.models.generative.architectures import VAE
from amp_searcher.models.contrastive.architectures import SimCLRBackbone
from amp_searcher.training.callbacks import GradientMonitor
from amp_searcher.training.trainer import AmpTrainer


@pytest.fixture
def sample_config_screening():
    return {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "ScreeningLightningModule",
            "architecture": {
                "name": "FeedForwardNeuralNetwork",
                "params": {
                    "output_dim": 1,
                    "hidden_dims": [64, 32],
                },  # input_dim is set by trainer
            },
            "lightning_module_params": {"task_type": "classification"},
        },
        "trainer": {
            "max_epochs": 1,
            "logger_name": "test_screening",
            "log_dir": "./test_logs",
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "Test_AmpSearcher_Experiment",
            "run_name": "Test_Screening_Run",
        },
    }


@pytest.fixture
def sample_config_generative():
    return {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "GenerativeLightningModule",
            "architecture": {
                "name": "VAE",
                "params": {
                    "latent_dim": 5,
                    "encoder_hidden_dims": [64, 32],
                    "decoder_hidden_dims": [32, 64],
                },  # input_dim is set by trainer
            },
            "lightning_module_params": {"kl_weight": 0.001},
        },
        "trainer": {
            "max_epochs": 1,
            "logger_name": "test_generative",
            "log_dir": "./test_logs",
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "Test_AmpSearcher_Experiment",
            "run_name": "Test_Generative_Run",
        },
    }


@pytest.fixture
def sample_config_contrastive():
    return {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "ContrastiveLightningModule",
            "architecture": {
                "name": "SimCLRBackbone",
                "params": {
                    "embedding_dim": 5,
                    "hidden_dims": [15],
                },  # input_dim is set by trainer
            },
        },
        "trainer": {
            "max_epochs": 1,
            "logger_name": "test_contrastive",
            "log_dir": "./test_logs",
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "Test_AmpSearcher_Experiment",
            "run_name": "Test_Contrastive_Run",
        },
    }


def test_amptrainer_init_screening(sample_config_screening):
    trainer = AmpTrainer(sample_config_screening)
    assert isinstance(trainer.featurizer, PhysicochemicalFeaturizer)
    assert isinstance(trainer.model, ScreeningLightningModule)
    assert isinstance(trainer.model.model, FeedForwardNeuralNetwork)
    assert any(
        isinstance(logger, TensorBoardLogger) for logger in trainer.trainer.loggers
    )
    assert any(isinstance(logger, MLFlowLogger) for logger in trainer.trainer.loggers)


def test_amptrainer_init_generative(sample_config_generative):
    trainer = AmpTrainer(sample_config_generative)
    assert isinstance(trainer.featurizer, PhysicochemicalFeaturizer)
    assert isinstance(trainer.model, GenerativeLightningModule)
    assert isinstance(trainer.model.model, VAE)
    assert isinstance(trainer.trainer, Trainer)
    assert any(
        isinstance(logger, TensorBoardLogger) for logger in trainer.trainer.loggers
    )
    assert any(isinstance(logger, MLFlowLogger) for logger in trainer.trainer.loggers)


def test_amptrainer_init_contrastive(sample_config_contrastive):
    trainer = AmpTrainer(sample_config_contrastive)
    assert isinstance(trainer.featurizer, PhysicochemicalFeaturizer)
    assert isinstance(trainer.model, ContrastiveLightningModule)
    assert isinstance(trainer.model.model, SimCLRBackbone)
    assert isinstance(trainer.trainer, Trainer)
    assert any(
        isinstance(logger, TensorBoardLogger) for logger in trainer.trainer.loggers
    )
    assert any(isinstance(logger, MLFlowLogger) for logger in trainer.trainer.loggers)


# Removed test_amptrainer_load_model_from_mlflow as it requires more complex setup


def test_amptrainer_unsupported_featurizer():
    config = {"featurizer": {"name": "InvalidFeaturizer"}}
    with pytest.raises(
        ValueError, match="No featurizer registered with name 'InvalidFeaturizer'"
    ):
        AmpTrainer(config)


def test_amptrainer_unsupported_model_architecture():
    config = {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "ScreeningLightningModule",
            "architecture": {"name": "InvalidArch"},
        },
    }
    with pytest.raises(ValueError, match="No model registered with name 'InvalidArch'"):
        AmpTrainer(config)


def test_amptrainer_unsupported_model_type():
    config = {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "InvalidLightningModule",
            "architecture": {
                "name": "FeedForwardNeuralNetwork",
                "params": {"input_dim": 10, "output_dim": 1, "hidden_dims": [64, 32]},
            },
        },
    }
    with pytest.raises(
        ValueError,
        match="No LightningModule registered with name 'InvalidLightningModule'",
    ):
        AmpTrainer(config)


def test_amptrainer_gradient_monitor_callback():
    config = {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "ScreeningLightningModule",
            "architecture": {
                "name": "FeedForwardNeuralNetwork",
                "params": {"input_dim": 10, "output_dim": 1, "hidden_dims": [64, 32]},
            },
            "lightning_module_params": {"task_type": "classification"},
        },
        "trainer": {"max_epochs": 1, "monitor_gradients": True},
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "Test_AmpSearcher_Experiment",
            "run_name": "Test_Gradient_Monitor_Run",
        },
    }
    trainer = AmpTrainer(config)
    assert any(isinstance(cb, GradientMonitor) for cb in trainer.trainer.callbacks)


def test_amptrainer_featurize_data():
    config = {
        "featurizer": {"name": "PhysicochemicalFeaturizer"},
        "model": {
            "type": "ScreeningLightningModule",
            "architecture": {
                "name": "FeedForwardNeuralNetwork",
                "params": {"input_dim": 10, "output_dim": 1, "hidden_dims": [64, 32]},
            },
            "lightning_module_params": {"task_type": "classification"},
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "Test_AmpSearcher_Experiment",
            "run_name": "Test_Featurize_Data_Run",
        },
    }
    trainer = AmpTrainer(config)
    sequences = ["ACDEF", "GHIJKL"]
    featurized_data = trainer.featurize_data(sequences)
    assert isinstance(featurized_data, torch.Tensor)
    assert featurized_data.shape[0] == len(sequences)
    assert featurized_data.shape[1] == 10
