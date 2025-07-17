from typing import cast

import torch
import torch.nn as nn

from amp_searcher.models.model_factory import ModelFactory  # Import ModelFactory


@ModelFactory.register("TransformerEncoder")  # Register the model
class TransformerEncoder(nn.Module):
    """
    A simplified Transformer Encoder for sequence data.

    This module takes sequence embeddings as input and processes them
    through a series of Transformer Encoder layers.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input and output tensors are (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        return cast(torch.Tensor, self.transformer_encoder(x))
