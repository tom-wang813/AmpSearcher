from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Deque, Optional
import torch
from torch import Tensor, nn

from .base import BaseLoss

class WeightingStrategy(ABC):
    """
    Abstract strategy for combining multiple task losses.
    """
    @abstractmethod
    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        """Combine individual losses into a single scalar."""
        ...

class HomoscedasticStrategy(nn.Module, WeightingStrategy):
    """
    Homoscedastic uncertainty weighting (Kendall et al.).
    Learn per-task noise precision (log variance).
    """
    def __init__(self, task_names: Optional[list[str]] = None):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in (task_names or [])
        })

    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        total = None
        for name, loss in losses.items():
            log_var = self.log_vars[name]
            w = 0.5 * (torch.exp(-log_var) * loss + log_var)
            total = w if total is None else total + w
        assert total is not None, "No tasks to combine"
        return total

class MovingAverageStrategy(WeightingStrategy):
    """
    Dynamic weighting based on moving average of past losses.
    Weight inversely proportional to recent average loss.
    """
    def __init__(self, task_names: list[str], window_size: int = 10, eps: float = 1e-8):
        self.eps = eps
        self.history: Dict[str, Deque[float]] = {
            name: deque(maxlen=window_size) for name in task_names
        }

    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        # update history
        for name, loss in losses.items():
            self.history[name].append(float(loss.detach().cpu().item()))
        # compute average losses
        avg_losses = {name: sum(hist)/len(hist) for name, hist in self.history.items()}
        # inverse weights
        inv = {name: 1.0/(avg + self.eps) for name, avg in avg_losses.items()}
        # normalize to sum to tasks count
        total_inv = sum(inv.values())
        weights = {name: val * len(inv)/total_inv for name, val in inv.items()}
        # combine
        total = None
        for name, loss in losses.items():
            w = weights[name]
            total = (loss * w) if total is None else total + loss * w
        assert total is not None, "No tasks to combine"
        return total

class EqualStrategy(WeightingStrategy):
    """
    Simple equal weighting for all tasks.
    """
    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        total = None
        for loss in losses.values():
            total = loss if total is None else total + loss
        assert total is not None, "No tasks to combine"
        return total

class SoftmaxNormalizationStrategy(WeightingStrategy):
    """
    Weight tasks via softmax of their losses:
    weight_i = exp(loss_i) / sum_j exp(loss_j) * num_tasks
    """
    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        # stack losses to compute weights
        loss_values = torch.stack(list(losses.values()))
        exp_losses = torch.exp(loss_values)
        weights = exp_losses / exp_losses.sum() * len(losses)
        total = None
        for (name, loss), w in zip(losses.items(), weights):
            total = loss * w if total is None else total + loss * w
        assert total is not None, "No tasks to combine"
        return total

class InverseLossStrategy(WeightingStrategy):
    """
    Instant inverse loss weighting: weight_i = 1/(loss_i + eps), normalized to sum to num tasks.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        inv = {name: 1.0/(loss + self.eps) for name, loss in losses.items()}
        total_inv = sum(inv.values())
        weights = {name: val/total_inv * len(inv) for name, val in inv.items()}
        total = None
        for name, loss in losses.items():
            w = weights[name]
            total = (loss * w) if total is None else total + loss * w
        assert total is not None, "No tasks to combine"
        return total

class DynamicWeightAverageStrategy(WeightingStrategy):
    """
    Dynamic Weight Average (DWA) strategy (Liu et al.).
    weight_i(t) = N * exp(r_i(t)/T) / sum_j exp(r_j(t)/T),
    where r_i(t) = L_i(t-1)/L_i(t-2).
    """
    def __init__(self, task_names: list[str], temperature: float = 2.0):
        self.temperature = temperature
        self.history: Dict[str, Deque[float]] = {name: deque(maxlen=2) for name in task_names}

    def apply(self, losses: Dict[str, Tensor]) -> Tensor:
        # update history of scalar losses
        for name, loss in losses.items():
            self.history[name].append(float(loss.detach().cpu().item()))
        K = len(losses)
        # if insufficient history, fallback to equal sum
        if any(len(hist) < 2 for hist in self.history.values()):
            total = None
            for loss in losses.values():
                total = loss if total is None else total + loss
            assert total is not None, "No tasks to combine"
            return total
        # compute ratios of past losses
        ratios = torch.tensor([self.history[name][-1] / self.history[name][-2] for name in losses.keys()])
        weights = torch.softmax(ratios / self.temperature, dim=0) * K
        total = None
        for (name, loss), w in zip(losses.items(), weights):
            total = (loss * w) if total is None else total + loss * w
        assert total is not None, "No tasks to combine"
        return total