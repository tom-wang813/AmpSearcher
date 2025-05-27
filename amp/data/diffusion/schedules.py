import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple

class DiffusionSchedule(ABC):
    """Base class for diffusion schedules.
    
    This class defines the interface for different types of noise schedules used in diffusion models.
    Subclasses should implement the specific scheduling logic.
    """
    
    def __init__(self, num_timesteps: int):
        """Initialize the diffusion schedule.
        
        Args:
            num_timesteps (int): Total number of diffusion steps.
        """
        self.num_timesteps = num_timesteps
        self._betas = None
        self._alphas = None
        self._alphas_cumprod = None
        
    @property
    def betas(self) -> torch.Tensor:
        """Get the noise schedule beta values."""
        if self._betas is None:
            self._setup_schedule()
        assert self._betas is not None, "Betas must be initialized in _setup_schedule"
        return self._betas
    
    @property
    def alphas(self) -> torch.Tensor:
        """Get the noise schedule alpha values (1 - beta)."""
        if self._alphas is None:
            self._alphas = 1.0 - self.betas
        return self._alphas
    
    @property
    def alphas_cumprod(self) -> torch.Tensor:
        """Get the cumulative product of alphas."""
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        return self._alphas_cumprod
    
    @abstractmethod
    def _setup_schedule(self) -> None:
        """Set up the noise schedule. Must be implemented by subclasses."""
        pass
    
    def get_parameters(self, t: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get diffusion parameters for timestep t.
        
        Args:
            t: Current timestep or batch of timesteps.
            
        Returns:
            Tuple containing (alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t)
        """
        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        return alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t

class LinearSchedule(DiffusionSchedule):
    """Linear beta schedule as used in the original DDPM paper."""
    
    def __init__(self, num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
        """Initialize the linear schedule.
        
        Args:
            num_timesteps: Total number of diffusion steps.
            beta_start: Starting value for beta schedule.
            beta_end: Ending value for beta schedule.
        """
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def _setup_schedule(self) -> None:
        """Create a linear noise schedule."""
        self._betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_timesteps
        )

class CosineSchedule(DiffusionSchedule):
    """Cosine beta schedule from the improved DDPM paper."""
    
    def __init__(self, num_timesteps: int, s: float = 0.008):
        """Initialize the cosine schedule.
        
        Args:
            num_timesteps: Total number of diffusion steps.
            s: Offset parameter to prevent alphas from being too small.
        """
        super().__init__(num_timesteps)
        self.s = s
    
    def _setup_schedule(self) -> None:
        """Create a cosine noise schedule."""
        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
        alpha_bar = torch.cos(((steps / self.num_timesteps + self.s) / (1 + self.s) * np.pi * 0.5)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        self._betas = torch.clip(betas, 0.0, 0.999)

class QuadraticSchedule(DiffusionSchedule):
    """Quadratic beta schedule for smoother noise addition."""
    
    def __init__(self, num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
        """Initialize the quadratic schedule.
        
        Args:
            num_timesteps: Total number of diffusion steps.
            beta_start: Starting value for beta schedule.
            beta_end: Ending value for beta schedule.
        """
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def _setup_schedule(self) -> None:
        """Create a quadratic noise schedule."""
        steps = torch.linspace(0, 1, self.num_timesteps)
        self._betas = self.beta_start + (self.beta_end - self.beta_start) * steps ** 2