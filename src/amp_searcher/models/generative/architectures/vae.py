import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_encoder import VAEEncoder
from .vae_decoder import VAEDecoder
from amp_searcher.models.model_factory import ModelFactory


@ModelFactory.register("VAE")
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) architecture.

    Combines an encoder and a decoder to learn a latent representation
    and reconstruct input data. Includes the reparameterization trick
    and computes the VAE loss (reconstruction loss + KL divergence).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        config: dict | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        config = config or {}
        self.kl_weight = config.get("kl_weight", 0.001)  # Store kl_weight as an instance attribute

        self.encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
        )
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,  # Output dim of decoder is input dim of VAE
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor.

        Returns:
            A tuple containing (reconstructed_x, mu, log_var).
        """
        mu, log_var = self.encoder(x)
        z = self._reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, log_var

    def compute_loss(
        self,
        batch: tuple[torch.Tensor, ...],
        reconstruction_loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Computes the VAE loss (reconstruction loss + KL divergence).

        Args:
            batch: A tuple containing the input tensor (x, ...).
            reconstruction_loss_type: Type of reconstruction loss ("mse" or "bce").

        Returns:
            The total VAE loss.
        """
        x, _ = batch  # Assuming batch contains (input, target) or just (input,)
        reconstructed_x, mu, log_var = self.forward(x)

        # Reconstruction Loss
        if reconstruction_loss_type == "mse":
            reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction="sum")
        elif reconstruction_loss_type == "bce":
            reconstruction_loss = F.binary_cross_entropy_with_logits(
                reconstructed_x, x, reduction="sum"
            )
        else:
            raise ValueError(
                f"Unsupported reconstruction loss type: {reconstruction_loss_type}"
            )

        # KL Divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = reconstruction_loss + self.kl_weight * kl_divergence
        return total_loss
