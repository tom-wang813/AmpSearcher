import pytest
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

from amp_searcher.models.screening import ScreeningLightningModule
from amp_searcher.models.architectures.feed_forward_nn import (
    FeedForwardNeuralNetwork as FFNN,
)
from amp_searcher.featurizers.physicochemical import PhysicochemicalFeaturizer


@pytest.fixture
def sample_data():
    # Load real data and featurize it
    data_path = "data/processed/data_from_fasta.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        pytest.skip(f"Data file not found at {data_path}. Please ensure it exists.")

    featurizer = PhysicochemicalFeaturizer()
    sequences = df["sequence"].tolist()

    featurized_data = torch.tensor(
        [featurizer.featurize(seq) for seq in sequences], dtype=torch.float32
    )

    # Standardize the featurized data
    scaler = StandardScaler()
    featurized_data_np = featurized_data.numpy()
    scaled_data_np = scaler.fit_transform(featurized_data_np)
    X = torch.tensor(scaled_data_np, dtype=torch.float32)

    # Dummy targets for classification and regression
    y_cls = torch.randint(0, 2, (X.shape[0],)).float()  # Binary classification targets
    y_reg = torch.randn(
        X.shape[0],
    ).float()  # Regression targets
    return X, y_cls, y_reg


@pytest.fixture
def classification_model():
    ffnn = FFNN(input_dim=10, output_dim=1, hidden_dims=[20, 10])
    return ScreeningLightningModule(model_architecture=ffnn, task_type="classification")


@pytest.fixture
def regression_model():
    ffnn = FFNN(input_dim=10, output_dim=1, hidden_dims=[20, 10])
    return ScreeningLightningModule(model_architecture=ffnn, task_type="regression")


def test_ffnn_architecture():
    """Test FFNN architecture and forward pass."""
    model = FFNN(
        input_dim=5,
        output_dim=1,
        hidden_dims=[10, 5],
    )
    assert isinstance(model, nn.Module)
    assert len(model.model) == 5  # Linear, ReLU, Linear, ReLU, Linear

    x = torch.randn(1, 5)
    output = model(x)
    assert output.shape == (1, 1)


def test_screening_lightning_module_init_classification(classification_model):
    """Test initialization of classification ScreeningLightningModule."""
    assert isinstance(classification_model.model, FFNN)
    assert classification_model.task_type == "classification"
    assert isinstance(classification_model.loss_fn, nn.BCEWithLogitsLoss)
    assert hasattr(classification_model, "accuracy")
    assert hasattr(classification_model, "f1_score")


def test_screening_lightning_module_init_regression(regression_model):
    """Test initialization of regression ScreeningLightningModule."""
    assert isinstance(regression_model.model, FFNN)
    assert regression_model.task_type == "regression"
    assert isinstance(regression_model.loss_fn, nn.MSELoss)
    assert hasattr(regression_model, "mse")


def test_screening_lightning_module_init_invalid_task_type():
    """Test initialization with invalid task type."""
    ffnn = FFNN(input_dim=10, output_dim=1, hidden_dims=[10])
    with pytest.raises(
        ValueError, match="task_type must be 'classification' or 'regression'"
    ):
        ScreeningLightningModule(model_architecture=ffnn, task_type="invalid")


def test_screening_lightning_module_training_step_classification(
    classification_model, sample_data
):
    """Test training step for classification."""
    X, y_cls, _ = sample_data
    batch = (X, y_cls)
    loss = classification_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative


def test_screening_lightning_module_validation_step_classification(
    classification_model, sample_data
):
    """Test validation step for classification."""
    X, y_cls, _ = sample_data
    batch = (X, y_cls)
    loss = classification_model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative


def test_screening_lightning_module_training_step_regression(
    regression_model, sample_data
):
    """Test training step for regression."""
    X, _, y_reg = sample_data
    batch = (X, y_reg)
    loss = regression_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative


def test_screening_lightning_module_validation_step_regression(
    regression_model, sample_data
):
    """Test validation step for regression."""
    X, _, y_reg = sample_data
    batch = (X, y_reg)
    loss = regression_model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative
