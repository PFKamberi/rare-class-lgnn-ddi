"""
Two LDAM variants exist in the thesis code and both are kept:
  LDAMLoss            m_c = max_m / n_c^(1/4)                  (warm start, cold start)
  LDAMLossNormalized  m_c = max_m * (n_c^-1/4 / max_c n_c^-1/4) (episodic)
They coincide only when the smallest (clamped) class count is 1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al., NeurIPS 2019).
    cls_num_list: training count per class."""
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        cls_num_list = torch.clamp(cls_num_list, min=1.0)
        self.m_list = max_m / torch.sqrt(torch.sqrt(cls_num_list))
        self.s = s

    def forward(self, logits, target):
        device = logits.device
        m_list = self.m_list.to(device)
        batch_m = m_list[target]
        logits_m = logits.clone()
        logits_m[torch.arange(logits.size(0), device=device), target] -= batch_m
        return F.cross_entropy(self.s * logits_m, target)


class LDAMLossNormalized(nn.Module):
    """LDAM as written in the episodic notebook: margins rescaled so the
    largest margin equals max_m."""
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.array(cls_num_list, dtype=np.float64).clip(min=1.0)))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        batch_m = self.m_list[target].unsqueeze(1)
        logits_m = logits - index.float() * batch_m
        return F.cross_entropy(self.s * logits_m, target)


def class_weights_from_train(labels, train_mask, num_classes, device,
                             smoothing=0.0, alpha_normalize=False):
    """Inverse-frequency weights from the training split (mean 1 over present classes)."""
    idx = train_mask.nonzero(as_tuple=False).squeeze(1)
    y = labels[idx].detach().cpu()
    counts = torch.bincount(y, minlength=num_classes).float()
    inv = torch.zeros_like(counts)
    nonzero = counts > 0
    inv[nonzero] = 1.0 / counts[nonzero]
    if nonzero.any():
        inv = inv * (nonzero.sum() / inv[nonzero].sum())
    if smoothing > 0:
        inv = (1 - smoothing) * inv + smoothing * torch.ones_like(inv)
    if alpha_normalize and inv.sum() > 0:
        inv = inv * (num_classes / inv.sum())
    return inv.to(device)


def class_weights(labels, C, device, smoothing=0.05):
    """Cold-start version: labels are already the training labels."""
    c = torch.bincount(labels.cpu(), minlength=C).float()
    inv = torch.zeros_like(c)
    nz = c > 0
    inv[nz] = 1.0 / c[nz]
    if nz.any():
        inv = inv * (nz.sum() / inv[nz].sum())
    return ((1 - smoothing) * inv + smoothing * torch.ones_like(inv)).to(device)
