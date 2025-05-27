from abc import ABC, abstractmethod
from typing import Callable, Union, Optional

from torch import Tensor

class BaseLoss(ABC):
    """
    Abstract base class for all post-processing loss functions.
    """

    def __call__(self,
                 predictions: Tensor,
                 targets: Tensor,
                 **kwargs) -> Tensor:
        return self.compute_loss(predictions, targets, **kwargs)

    @abstractmethod
    def compute_loss(self,
                     predictions: Tensor,
                     targets: Tensor,
                     **kwargs) -> Tensor:
        """
        Compute the loss given predictions and targets.
        Returns a Tensor with gradient support.
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ScaledLoss(BaseLoss):
    """
    A scaled wrapper around any loss function (BaseLoss or torch.nn loss).
    """

    def __init__(self,
                 base_loss_fn: Union[BaseLoss, Callable[[Tensor, Tensor], Tensor]]):
        # 校验 base_loss_fn
        if not callable(base_loss_fn):
            raise TypeError("base_loss_fn must be callable (BaseLoss or torch loss).")

        self.base_loss_fn = base_loss_fn


    def compute_loss(self,
                     predictions: Tensor,
                     targets: Tensor,
                     scale_factor: float = 1.0,
                     **kwargs) -> Tensor:
        # 支持 BaseLoss 子类或普通可调用 loss
        if not isinstance(scale_factor, (float, int)):
            raise TypeError("scale_factor must be a number.")
        if isinstance(self.base_loss_fn, BaseLoss):
            base = self.base_loss_fn.compute_loss(predictions, targets, **kwargs)
        else:
            base = self.base_loss_fn(predictions, targets, **kwargs)
        return base * scale_factor
    
    def __repr__(self):
        base_repr = repr(self.base_loss_fn)
        return f"{self.__class__.__name__}({base_repr})"


class MultiTaskLoss(BaseLoss):
    """
    Combine multiple loss functions for multi-task learning, with custom weights.
    """
    def __init__(self,
                 loss_fns: dict[str, Union[BaseLoss, Callable[[Tensor, Tensor], Tensor]]],
                 weights: Optional[dict[str, float]] = None):
        if not isinstance(loss_fns, dict):
            raise TypeError("loss_fns must be a dict mapping task names to loss callables.")
        self.loss_fns = loss_fns
        # default weight = 1.0 for each task
        if weights is None:
            self.weights = {name: 1.0 for name in loss_fns}
        else:
            if not set(weights).issubset(loss_fns.keys()):
                raise ValueError("weights keys must be a subset of loss_fns keys.")
            self.weights = weights

    def compute_loss(self,
                     predictions: dict,
                     targets: dict,
                     **kwargs) -> Tensor:
        """
        predictions and targets should be dicts with matching keys to loss_fns.
        Returns the weighted sum of individual losses.
        """
        total: Optional[Tensor] = None
        for name, fn in self.loss_fns.items():
            if name not in predictions or name not in targets:
                raise KeyError(f"Missing predictions or targets for task '{name}'")
            pred, tgt = predictions[name], targets[name]
            weight = float(self.weights.get(name, 1.0))
            # compute individual loss
            l = fn.compute_loss(pred, tgt, **kwargs) if isinstance(fn, BaseLoss) else fn(pred, tgt, **kwargs)
            weighted = l * weight
            total = weighted if total is None else total + weighted
        assert total is not None, "No tasks to compute loss"
        return total

    def __repr__(self):
        return f"{self.__class__.__name__}(tasks={list(self.loss_fns.keys())}, weights={self.weights})"

