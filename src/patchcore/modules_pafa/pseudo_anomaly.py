"""Pseudo-anomaly generators for PAFA."""

import torch


class GaussianPseudoAnomalyGenerator:
    """Create isotropic Gaussian perturbations in embedding space."""

    def __init__(self, noise_std: float = 0.015) -> None:
        if noise_std <= 0:
            raise ValueError("Gaussian noise_std must be positive.")
        self.noise_std = noise_std

    def generate(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.noise_std * torch.randn_like(features)
