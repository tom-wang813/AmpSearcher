from .base import BaseLoss, ScaledLoss, MultiTaskLoss
from .normalized import BaseNormalizer, MinMaxNormalizer, ZScoreNormalizer, NormalizedLoss
from .contrastive import SimCLRLoss, SupervisedContrastiveLoss
from .weighting import (
    WeightingStrategy,
    HomoscedasticStrategy,
    MovingAverageStrategy,
    EqualStrategy,
    SoftmaxNormalizationStrategy,
    InverseLossStrategy,
    DynamicWeightAverageStrategy,
)

__all__ = [
    'BaseLoss',
    'ScaledLoss',
    'MultiTaskLoss',
    'BaseNormalizer',
    'MinMaxNormalizer',
    'ZScoreNormalizer',
    'NormalizedLoss',
    'SimCLRLoss',
    'SupervisedContrastiveLoss',
    'WeightingStrategy',
    'HomoscedasticStrategy',
    'MovingAverageStrategy',
    'EqualStrategy',
    'SoftmaxNormalizationStrategy',
    'InverseLossStrategy',
    'DynamicWeightAverageStrategy',
]

LOSS_REGISTRY = {
    'BaseLoss': BaseLoss,
    'ScaledLoss': ScaledLoss,
    'MultiTaskLoss': MultiTaskLoss,
    'BaseNormalizer': BaseNormalizer,
    'MinMaxNormalizer': MinMaxNormalizer,
    'ZScoreNormalizer': ZScoreNormalizer,
    'NormalizedLoss': NormalizedLoss,
    'SimCLRLoss': SimCLRLoss,
    'SupervisedContrastiveLoss': SupervisedContrastiveLoss,
    'WeightingStrategy': WeightingStrategy,
    'HomoscedasticStrategy': HomoscedasticStrategy,
    'MovingAverageStrategy': MovingAverageStrategy,
    'EqualStrategy': EqualStrategy,
    'SoftmaxNormalizationStrategy': SoftmaxNormalizationStrategy,
    'InverseLossStrategy': InverseLossStrategy,
    'DynamicWeightAverageStrategy': DynamicWeightAverageStrategy,
}

def get_loss(name: str, *args, **kwargs) -> BaseLoss:
    """
    Instantiate a loss by name.
    """
    try:
        cls = LOSS_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown loss: {name}")
    return cls(*args, **kwargs)