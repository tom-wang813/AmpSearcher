import pytest
import torch
import os

from amp_searcher.pipelines.screening_pipeline import ScreeningPipeline
from amp_searcher.models.screening import ScreeningLightningModule

# Define the real checkpoint path
CHECKPOINT_PATH = "/Users/wang-workair/cui/AmpSearcher/lightning_logs/amp_training_run/version_4/checkpoints/epoch=4-step=520.ckpt"

# Define the real model configuration
REAL_MODEL_CONFIG = {
    "lightning_module_name": "ScreeningLightningModule",
    "architecture": {
        "name": "FeedForwardNeuralNetwork",
        "params": {"input_dim": 10, "output_dim": 1, "hidden_dims": [64, 32]},
    },
    "lightning_module_params": {
        "task_type": "classification",
    },
}


@pytest.fixture
def real_model_config():
    return REAL_MODEL_CONFIG


@pytest.fixture
def real_checkpoint_path():
    # Ensure the checkpoint file exists before returning the path
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(
            f"Real checkpoint file not found at {CHECKPOINT_PATH}. Please train the model first."
        )
    return CHECKPOINT_PATH


@pytest.fixture
def featurizer_config():
    return {"name": "PhysicochemicalFeaturizer"}


def test_screening_pipeline_init(
    real_checkpoint_path, featurizer_config, real_model_config
):
    pipeline = ScreeningPipeline(
        real_model_config, real_checkpoint_path, featurizer_config
    )
    assert pipeline.featurizer is not None
    assert pipeline.model is not None
    assert isinstance(pipeline.model, ScreeningLightningModule)
    assert pipeline.model.training is False  # Should be in eval mode


def test_screening_pipeline_predict(
    real_checkpoint_path, featurizer_config, real_model_config
):
    pipeline = ScreeningPipeline(
        real_model_config, real_checkpoint_path, featurizer_config
    )
    sequences = ["ACDEF", "GHIJKL"]
    predictions = pipeline.predict(sequences)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(sequences), 1)
    assert torch.all((predictions >= 0) & (predictions <= 1))  # Probabilities


def test_screening_pipeline_predict_empty_sequences(
    real_checkpoint_path, featurizer_config, real_model_config
):
    pipeline = ScreeningPipeline(
        real_model_config, real_checkpoint_path, featurizer_config
    )
    sequences = []
    predictions = pipeline.predict(sequences)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (0,)


def test_screening_pipeline_unsupported_featurizer(
    real_checkpoint_path, real_model_config
):
    with pytest.raises(
        ValueError, match="No featurizer registered with name 'InvalidFeaturizer'"
    ):
        ScreeningPipeline(
            real_model_config, real_checkpoint_path, {"name": "InvalidFeaturizer"}
        )
