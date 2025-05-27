from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseLoss


# Utility functions for contrastive losses
def normalize_embeddings(x: Tensor) -> Tensor:
    return F.normalize(x, dim=1)


def compute_similarity_matrix(x: Tensor, y: Tensor, temperature: float) -> Tensor:
    # concatenate and compute scaled cosine similarities
    reps = torch.cat([x, y], dim=0) if y is not None else x
    sim = torch.matmul(reps, reps.T) / temperature
    return sim


def mask_self_similarity(sim: Tensor) -> Tensor:
    mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    return sim.masked_fill(mask, -float('inf'))


class SimCLRLoss(BaseLoss):
    """
    Unsupervised contrastive loss (NT-Xent) for SimCLR.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def compute_loss(self,
                     z_i: Tensor,
                     z_j: Tensor,
                     **kwargs) -> Tensor:
        # normalize and compute similarity
        zi, zj = normalize_embeddings(z_i), normalize_embeddings(z_j)
        batch_size = zi.size(0)
        sim = compute_similarity_matrix(zi, zj, self.temperature)
        sim = mask_self_similarity(sim)
        # cross-entropy expects logits and target indices
        labels = torch.arange(batch_size, device=zi.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        return F.cross_entropy(sim, labels)

    def __repr__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature})"


# Supervised variant
class SupervisedContrastiveLoss(BaseLoss):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    """
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def compute_loss(self,
                     features: Tensor,
                     labels: Tensor,
                     mask: Optional[Tensor] = None,
                     **kwargs) -> Tensor:
        device = features.device
        batch_size = features.size(0)
        # default mask from labels
        if mask is None:
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # normalize
        feats = normalize_embeddings(features)
        # compute logits
        anchor_dot = torch.matmul(feats, feats.T) / self.temperature
        # stability
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()
        # mask self
        logits_mask = (~torch.eye(batch_size, device=device, dtype=torch.bool)).float()
        mask = mask * logits_mask
        # log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mean of log-likelihood over positives
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob
        return loss.mean()

    def __repr__(self):
        return (f"{self.__class__.__name__}(temperature={self.temperature}, "
                f"base_temperature={self.base_temperature})")
