"""
Shared neural network layers for NMT models.

This module provides reusable components:
- RMSNorm: Root Mean Square Layer Normalization
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster alternative to LayerNorm that only uses RMS for normalization,
    without centering (no mean subtraction). Introduced in "Root Mean Square Layer
    Normalization" (Zhang & Sennrich, 2019).

    Args:
        dim: Dimension of the input features
        eps: Small epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x / rms

    def extra_repr(self) -> str:
        return f'dim={self.weight.shape[0]}, eps={self.eps}'
