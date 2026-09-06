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


class DirectedPseudoAnomalyGenerator:
    """Create pseudo anomalies by moving features away from local nominal neighbors."""

    def __init__(self, step_size: float = 0.015, jitter_std: float = 0.001) -> None:
        if step_size <= 0:
            raise ValueError("Directed pseudo-anomaly step_size must be positive.")
        if jitter_std < 0:
            raise ValueError("Directed pseudo-anomaly jitter_std must be non-negative.")
        self.step_size = step_size
        self.jitter_std = jitter_std

    def generate(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Directed pseudo-anomaly features must be a 2D tensor.")

        directions = self._nearest_neighbor_away_directions(features)
        if self.jitter_std > 0:
            directions = directions + self.jitter_std * torch.randn_like(directions)
            directions = torch.nn.functional.normalize(directions, dim=1)
        return features + self.step_size * directions

    @staticmethod
    def _nearest_neighbor_away_directions(features: torch.Tensor) -> torch.Tensor:
        if len(features) <= 1:
            random_directions = torch.randn_like(features)
            return torch.nn.functional.normalize(random_directions, dim=1)

        distances = torch.cdist(features, features, p=2)
        distances.fill_diagonal_(float("inf"))
        nearest_indices = distances.argmin(dim=1)
        directions = features - features[nearest_indices]
        zero_direction_mask = torch.norm(directions, dim=1) == 0
        if zero_direction_mask.any():
            directions = directions.clone()
            directions[zero_direction_mask] = torch.randn_like(
                directions[zero_direction_mask]
            )
        return torch.nn.functional.normalize(directions, dim=1)
