from torch import Tensor
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import cast

from .base import BaseLoss


class BaseNormalizer(ABC, nn.Module):
    """
    Abstract base for running-stat normalizers.
    """
    def __init__(self, momentum: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    @abstractmethod
    def update(self, x: Tensor):
        ...

    @abstractmethod
    def normalize(self, x: Tensor) -> Tensor:
        ...


class MinMaxNormalizer(BaseNormalizer):
    """Running Min-Max normalizer."""
    def __init__(self, momentum: float = 0.1, eps: float = 1e-8):
        super().__init__(momentum, eps)
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(1.0))

    def update(self, x: Tensor):
        batch_min = x.min()
        batch_max = x.max()
        # get buffers as Tensor
        running_min = cast(Tensor, self.running_min)
        running_max = cast(Tensor, self.running_max)
        with torch.no_grad():
            new_min = running_min * (1 - self.momentum) + batch_min * self.momentum
            new_max = running_max * (1 - self.momentum) + batch_max * self.momentum
            running_min.copy_(new_min)
            running_max.copy_(new_max)

    def normalize(self, x: Tensor) -> Tensor:
        running_min = cast(Tensor, self.running_min)
        running_max = cast(Tensor, self.running_max)
        return (x - running_min) / (running_max - running_min + self.eps)


class ZScoreNormalizer(BaseNormalizer):
    """Running Z-Score normalizer."""
    def __init__(self, momentum: float = 0.1, eps: float = 1e-8):
        super().__init__(momentum, eps)
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))

    def update(self, x: Tensor):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        running_mean = cast(Tensor, self.running_mean)
        running_var = cast(Tensor, self.running_var)
        with torch.no_grad():
            new_mean = running_mean * (1 - self.momentum) + batch_mean * self.momentum
            new_var = running_var * (1 - self.momentum) + batch_var * self.momentum
            running_mean.copy_(new_mean)
            running_var.copy_(new_var)

    def normalize(self, x: Tensor) -> Tensor:
        running_mean = cast(Tensor, self.running_mean)
        running_var = cast(Tensor, self.running_var)
        return (x - running_mean) / (torch.sqrt(running_var) + self.eps)


class NormalizedLoss(BaseLoss, nn.Module):
    """
    Wraps any BaseLoss to normalize predictions or targets before computing loss.
    Supports min-max or z-score normalization using running statistics.
    """
    def __init__(self,
                 base_loss_fn: BaseLoss,
                 mode: str = 'minmax',
                 momentum: float = 0.1,
                 eps: float = 1e-8):
        super().__init__()
        if mode not in ('minmax', 'zscore'):
            raise ValueError("mode must be 'minmax' or 'zscore'.")
        self.base_loss_fn = base_loss_fn
        # choose normalizer class
        Normalizer = MinMaxNormalizer if mode == 'minmax' else ZScoreNormalizer
        # separate normalizers for preds and targs
        self.pred_normalizer = Normalizer(momentum, eps)
        self.targ_normalizer = Normalizer(momentum, eps)

    def compute_loss(self,
                     predictions: Tensor,
                     targets: Tensor,
                     **kwargs) -> Tensor:
        # update running stats and normalize
        self.pred_normalizer.update(predictions)
        self.targ_normalizer.update(targets)
        preds = self.pred_normalizer.normalize(predictions)
        targs = self.targ_normalizer.normalize(targets)
        return self.base_loss_fn.compute_loss(preds, targs, **kwargs)

    def __repr__(self):
        return (f"{self.__class__.__name__}(base={repr(self.base_loss_fn)}, "
                f"mode='{self.pred_normalizer.__class__.__name__}', "
                f"momentum={self.pred_normalizer.momentum})")
