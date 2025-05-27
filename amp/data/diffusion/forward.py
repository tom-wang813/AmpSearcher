import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from .schedules import DiffusionSchedule

class DiffusionForward:
    """Base class for forward diffusion process.
    
    This class implements the forward diffusion process for different types of data,
    including continuous (e.g., images, embeddings) and discrete (e.g., text, SMILES) data.
    """
    
    def __init__(
        self,
        schedule: DiffusionSchedule,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the forward diffusion process.
        
        Args:
            schedule: The noise schedule to use for the diffusion process
            device: The device to perform computations on
        """
        self.schedule = schedule
        self.device = device
        
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: Union[torch.Tensor, int],
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: q(x_t | x_0).
        
        Adds noise to the input according to the noise schedule.
        
        Args:
            x_start: The initial data (B x ...)
            t: Timesteps (B,) or single timestep
            noise: Optional pre-generated noise. If None, random noise will be used
            
        Returns:
            Tuple containing:
                - The noised data x_t
                - The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Get diffusion parameters for timestep t
        alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t = self.schedule.get_parameters(t)
        
        # Reshape parameters to match input dimensions
        sqrt_alpha_t = sqrt_alpha_t.view(-1, *([1] * (len(x_start.shape) - 1)))
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(-1, *([1] * (len(x_start.shape) - 1)))
        
        # Forward process: x_t = √α_t * x_0 + √(1-α_t) * ε
        x_t = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: Union[torch.Tensor, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: The initial data (B x ...)
            x_t: The noised data at timestep t
            t: Timesteps (B,) or single timestep
            
        Returns:
            Tuple containing:
                - Posterior mean
                - Posterior variance
        """
        alpha_t, _, _ = self.schedule.get_parameters(t)
        alpha_tm1 = self.schedule.alphas_cumprod[t-1] if t > 0 else torch.ones_like(alpha_t)
        
        # Compute posterior mean coefficient for x_0 and x_t
        posterior_variance = (1 - alpha_tm1) * (1 - alpha_t) / (1 - alpha_t)
        posterior_mean = (
            torch.sqrt(alpha_tm1) * (1 - alpha_t) / (1 - alpha_t) * x_start +
            torch.sqrt(alpha_t) * (1 - alpha_tm1) / (1 - alpha_t) * x_t
        )
        
        return posterior_mean, posterior_variance

class DiscreteDataDiffusion(DiffusionForward):
    """Special handling for discrete data like SMILES or protein sequences."""
    
    def __init__(
        self,
        schedule: DiffusionSchedule,
        vocab_size: int,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize discrete data diffusion.
        
        Args:
            schedule: The noise schedule to use
            vocab_size: Size of the vocabulary for discrete tokens
            device: Device to perform computations on
        """
        super().__init__(schedule, device)
        self.vocab_size = vocab_size
        
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: Union[torch.Tensor, int],
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion for discrete data.
        
        For discrete data, we first convert to a one-hot representation,
        then apply continuous diffusion, and finally apply softmax
        to maintain a probability distribution over tokens.
        
        Args:
            x_start: Input tokens (B x L) or one-hot (B x L x V)
            t: Timesteps
            noise: Optional pre-generated noise
            
        Returns:
            Tuple containing:
                - Noised token probabilities
                - Added noise
        """
        # Convert to one-hot if needed
        if len(x_start.shape) == 2:
            x_start = F.one_hot(x_start, num_classes=self.vocab_size).float()
        
        # Apply continuous diffusion
        x_t, noise = super().q_sample(x_start, t, noise)
        
        # Apply softmax to maintain probability distribution
        x_t = F.softmax(x_t, dim=-1)
        
        return x_t, noise