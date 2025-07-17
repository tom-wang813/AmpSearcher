from typing import List, cast

import torch
import torch.nn as nn


class VAEDecoder(nn.Module):
    """
    Decoder part of a Variational Autoencoder (VAE).

    Reconstructs data from the latent space.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.activation_name = activation.lower()

        layers: List[nn.Module] = []
        current_dim = latent_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(self._get_activation_fn())
            current_dim = h_dim

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z: Latent space tensor.

        Returns:
            Reconstructed output tensor.
        """
        return cast(torch.Tensor, self.model(z))
