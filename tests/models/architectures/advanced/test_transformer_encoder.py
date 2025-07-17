import torch

from amp_searcher.models.architectures.advanced import TransformerEncoder


def test_transformer_encoder_init():
    model = TransformerEncoder(
        embedding_dim=128, num_heads=4, num_layers=2, dim_feedforward=256
    )
    assert isinstance(model, TransformerEncoder)
    assert model.embedding_dim == 128


def test_transformer_encoder_forward():
    model = TransformerEncoder(
        embedding_dim=128, num_heads=4, num_layers=2, dim_feedforward=256
    )
    # Input: (batch_size, sequence_length, embedding_dim)
    x = torch.randn(2, 50, 128)
    output = model(x)
    assert output.shape == (2, 50, 128)


def test_transformer_encoder_with_different_dims():
    model = TransformerEncoder(
        embedding_dim=64, num_heads=2, num_layers=1, dim_feedforward=128, dropout=0.5
    )
    x = torch.randn(1, 100, 64)
    output = model(x)
    assert output.shape == (1, 100, 64)
