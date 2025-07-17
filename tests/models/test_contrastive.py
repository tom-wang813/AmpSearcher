import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from amp_searcher.models.contrastive import ContrastiveLightningModule
from amp_searcher.models.contrastive.architectures import SimCLRBackbone
from amp_searcher.featurizers.physicochemical import PhysicochemicalFeaturizer
from amp_searcher.featurizers.composition import CompositionFeaturizer


@pytest.fixture
def sample_data_contrastive():
    # Load real data and featurize it with two different featurizers
    data_path = "data/processed/data_from_fasta.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        pytest.skip(f"Data file not found at {data_path}. Please ensure it exists.")

    phys_featurizer = PhysicochemicalFeaturizer()
    comp_featurizer = CompositionFeaturizer(include_aac=True, include_dpc=False)

    sequences = df["sequence"].tolist()

    featurized_data_list = []
    for seq in sequences:
        phys_features = phys_featurizer.featurize(seq)
        comp_features = comp_featurizer.featurize(seq)
        combined_features = np.concatenate((phys_features, comp_features))
        featurized_data_list.append(combined_features)

    # Convert list of numpy arrays to a single numpy array, then to a tensor
    featurized_data = torch.tensor(np.array(featurized_data_list), dtype=torch.float32)

    # For contrastive learning, we need two views. For simplicity, we'll use the same data twice.
    # In a real scenario, you would apply different augmentations to create two distinct views.
    return featurized_data, featurized_data


@pytest.fixture
def simclr_backbone():
    # input_dim = output of PhysicochemicalFeaturizer (10) + output of CompositionFeaturizer (20) = 30
    return SimCLRBackbone(input_dim=30, embedding_dim=10, hidden_dims=[15])


@pytest.fixture
def contrastive_lightning_module(simclr_backbone):
    return ContrastiveLightningModule(
        model_architecture=simclr_backbone, config={"temperature": 0.1}
    )


def test_simclr_backbone_init():
    backbone = SimCLRBackbone(input_dim=30, embedding_dim=5, hidden_dims=[8])
    assert isinstance(backbone, nn.Module)
    assert backbone.embedding_dim == 5


def test_simclr_backbone_forward():
    backbone = SimCLRBackbone(input_dim=30, embedding_dim=5, hidden_dims=[8])
    x = torch.randn(1, 30)
    output = backbone(x)
    assert output.shape == (1, 5)


def test_simclr_backbone_unsupported_activation():
    with pytest.raises(
        ValueError, match="Unsupported activation function: invalid_act"
    ):
        SimCLRBackbone(
            input_dim=30, embedding_dim=1, hidden_dims=[10], activation="invalid_act"
        )


def test_contrastive_lightning_module_init(
    contrastive_lightning_module, simclr_backbone
):
    assert isinstance(contrastive_lightning_module.model, SimCLRBackbone)
    assert contrastive_lightning_module.model == simclr_backbone
    assert contrastive_lightning_module.temperature == 0.1


def test_contrastive_lightning_module_training_step(
    contrastive_lightning_module, sample_data_contrastive
):
    loss = contrastive_lightning_module.training_step(sample_data_contrastive, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_contrastive_lightning_module_validation_step(
    contrastive_lightning_module, sample_data_contrastive
):
    loss = contrastive_lightning_module.validation_step(sample_data_contrastive, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_nt_xent_loss_calculation():
    # Create a dummy module to access the _nt_xent_loss method
    class DummyModule(ContrastiveLightningModule):
        def __init__(self):
            # Use a dummy model_architecture that matches the expected input_dim
            super().__init__(model_architecture=nn.Linear(30, 10), config={"temperature": 0.1})

    dummy_module = DummyModule()
    dummy_module.to(torch.device("cpu"))  # Ensure on CPU for consistent testing

    # Example: two positive pairs, two negative pairs
    # z_i = [[1, 0], [0, 1]]
    # z_j = [[1, 0], [0, 1]]
    z_i = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_j = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = dummy_module._nt_xent_loss(z_i, z_j)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    # The exact value is hard to assert without a reference implementation,
    # but we can check for non-negativity and that it's a scalar.
