import pytest
import torch
import torch.nn as nn
import pandas as pd

from amp_searcher.models.generative import GenerativeLightningModule
from amp_searcher.models.generative.architectures import VAE, VAEEncoder, VAEDecoder
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
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    featurized_data_np = featurized_data.numpy()
    scaled_data_np = scaler.fit_transform(featurized_data_np)
    featurized_data = torch.tensor(scaled_data_np, dtype=torch.float32)
    print(
        f"\nFeaturized data stats: min={featurized_data.min()}, max={featurized_data.max()}, mean={featurized_data.mean()}, has_nan={torch.isnan(featurized_data).any()}"
    )
    # For VAE, we typically don't need a target, but the original fixture returned two values.
    # We'll return a dummy target of zeros.
    dummy_target = torch.zeros(featurized_data.shape[0], 1)

    return featurized_data, dummy_target


@pytest.fixture
def vae_model():
    # input_dim should match PhysicochemicalFeaturizer output (10)
    return VAE(
        input_dim=10, latent_dim=5, encoder_hidden_dims=[10], decoder_hidden_dims=[10]
    )


@pytest.fixture
def generative_lightning_module(vae_model):
    return GenerativeLightningModule(
        model_architecture=vae_model, latent_dim=vae_model.latent_dim, kl_weight=0.001
    )


def test_vae_encoder_init():
    encoder = VAEEncoder(input_dim=10, latent_dim=2, hidden_dims=[5])
    assert isinstance(encoder, nn.Module)
    assert encoder.latent_dim == 2


def test_vae_encoder_forward():
    encoder = VAEEncoder(input_dim=10, latent_dim=2, hidden_dims=[5])
    x = torch.randn(1, 10)
    mu, log_var = encoder(x)
    assert mu.shape == (1, 2)
    assert log_var.shape == (1, 2)


def test_vae_decoder_init():
    decoder = VAEDecoder(latent_dim=2, output_dim=10, hidden_dims=[5])
    assert isinstance(decoder, nn.Module)


def test_vae_decoder_forward():
    decoder = VAEDecoder(latent_dim=2, output_dim=10, hidden_dims=[5])
    z = torch.randn(1, 2)
    output = decoder(z)
    assert output.shape == (1, 10)


def test_vae_init(vae_model):
    assert isinstance(vae_model.encoder, VAEEncoder)
    assert isinstance(vae_model.decoder, VAEDecoder)
    assert vae_model.input_dim == 10
    assert vae_model.latent_dim == 5


def test_vae_forward(vae_model):
    x = torch.randn(1, 10)  # Input dim should be 10
    reconstructed_x, mu, log_var = vae_model(x)
    assert reconstructed_x.shape == (1, 10)
    assert mu.shape == (1, 5)
    assert log_var.shape == (1, 5)


def test_vae_compute_loss_mse(vae_model, sample_data):
    x, _ = sample_data
    loss = vae_model.compute_loss(batch=(x, None), reconstruction_loss_type="mse")
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_vae_compute_loss_bce(vae_model, sample_data):
    x, _ = sample_data
    # For BCE, input should be between 0 and 1
    x_bce = torch.rand(x.shape)  # Use the same shape as featurized data
    loss = vae_model.compute_loss(batch=(x_bce, None), reconstruction_loss_type="bce")
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_vae_compute_loss_unsupported_type(vae_model, sample_data):
    x, _ = sample_data
    with pytest.raises(
        ValueError, match="Unsupported reconstruction loss type: invalid"
    ):
        vae_model.compute_loss(batch=(x, None), reconstruction_loss_type="invalid")


def test_generative_lightning_module_init(generative_lightning_module, vae_model):
    assert isinstance(generative_lightning_module.model, VAE)
    assert generative_lightning_module.model == vae_model


def test_generative_lightning_module_training_step(
    generative_lightning_module, sample_data
):
    loss = generative_lightning_module.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_generative_lightning_module_validation_step(
    generative_lightning_module, sample_data
):
    loss = generative_lightning_module.validation_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
