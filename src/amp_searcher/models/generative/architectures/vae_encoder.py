from typing import List, cast, Tuple

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    Encoder part of a Variational Autoencoder (VAE).

    Maps input data to a latent space (mu and log_var).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.activation_name = activation.lower()

        layers: List[nn.Module] = []
        current_dim = input_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(self._get_activation_fn())
            current_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_log_var = nn.Linear(current_dim, latent_dim)

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor.

        Returns:
            A tuple containing (mu, log_var) of the latent distribution.
        """
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)
        return cast(Tuple[torch.Tensor, torch.Tensor], (mu, log_var))
