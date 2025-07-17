from typing import List, cast

import torch
import torch.nn as nn

from amp_searcher.models.model_factory import ModelFactory  # Import ModelFactory


@ModelFactory.register("FFNN")  # Register the model
class FFNN(nn.Module):
    """
    A simple Feed-Forward Neural Network (FFNN) architecture.

    This module defines the layers of the network and the forward pass.
    It does not contain any training logic, optimizers, or loss functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

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
        Forward pass through the FFNN.
        """
        return cast(torch.Tensor, self.model(x))
