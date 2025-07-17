from typing import List, cast

import torch
import torch.nn as nn

from amp_searcher.models.model_factory import ModelFactory


@ModelFactory.register("SimCLRBackbone")
class SimCLRBackbone(nn.Module):
    """
    A simple backbone for SimCLR-like contrastive learning.

    This module typically consists of an encoder (e.g., an FFNN) that maps
    input features to a lower-dimensional embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.activation_name = activation.lower()
        self.dropout_rate = dropout_rate

        layers: List[nn.Module] = []
        current_dim = input_dim

        # Hidden layers
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(self._get_activation_fn())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            current_dim = h_dim

        # Output layer (embedding layer)
        layers.append(nn.Linear(current_dim, embedding_dim))

        self.model = nn.Sequential(*layers)

    def _get_activation_fn(self) -> nn.Module:
        if self.activation_name == "relu":
            return nn.ReLU()
        elif self.activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif self.activation_name == "sigmoid":
            return nn.Sigmoid()
        elif self.activation_name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        """
        return cast(torch.Tensor, self.model(x))
