import pytest
import torch
import torch.nn as nn
import pandas as pd

from amp_searcher.models.screening import SequenceScreeningLightningModule
from amp_searcher.models.architectures.advanced import TransformerEncoder
from amp_searcher.featurizers.physicochemical import PhysicochemicalFeaturizer


@pytest.fixture
def transformer_model():
    # Updated embedding_dim to match PhysicochemicalFeaturizer output
    return TransformerEncoder(
        embedding_dim=10, num_heads=2, num_layers=1, dim_feedforward=256
    )


@pytest.fixture
def sequence_screening_model(transformer_model):
    return SequenceScreeningLightningModule(
        model_architecture=transformer_model, task_type="classification", output_dim=1
    )


@pytest.fixture
def sample_sequence_data():
    # Load real data and featurize it
    data_path = "data/processed/data_from_fasta.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        pytest.skip(f"Data file not found at {data_path}. Please ensure it exists.")

    featurizer = PhysicochemicalFeaturizer()
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()

    featurized_data = torch.tensor(
        [featurizer.featurize(seq) for seq in sequences], dtype=torch.float32
    )
    featurized_data = featurized_data.unsqueeze(1)  # Add sequence_length dimension
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return featurized_data, labels_tensor


def test_sequence_screening_lightning_module_init(
    sequence_screening_model, transformer_model
):
    assert isinstance(sequence_screening_model.model, TransformerEncoder)
    assert sequence_screening_model.model == transformer_model
    assert sequence_screening_model.task_type == "classification"
    assert sequence_screening_model.output_dim == 1
    assert isinstance(sequence_screening_model.prediction_head, nn.Linear)


def test_sequence_screening_lightning_module_forward(
    sequence_screening_model, sample_sequence_data
):
    x, _ = sample_sequence_data
    logits = sequence_screening_model(x)
    assert logits.shape == (x.shape[0], sequence_screening_model.output_dim)


def test_sequence_screening_lightning_module_training_step(
    sequence_screening_model, sample_sequence_data
):
    loss = sequence_screening_model.training_step(sample_sequence_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_sequence_screening_lightning_module_validation_step(
    sequence_screening_model, sample_sequence_data
):
    loss = sequence_screening_model.validation_step(sample_sequence_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_sequence_screening_lightning_module_invalid_task_type():
    transformer_model = TransformerEncoder(
        embedding_dim=10, num_heads=2, num_layers=1, dim_feedforward=256
    )
    with pytest.raises(
        ValueError, match="task_type must be 'classification' or 'regression'"
    ):
        SequenceScreeningLightningModule(
            model_architecture=transformer_model, task_type="invalid"
        )
