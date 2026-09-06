"""Residual adapter used by PAFA experiments."""

import torch


class ResidualFeatureAdapter(torch.nn.Module):
    """Apply a trainable residual bottleneck to PatchCore embeddings."""

    def __init__(
        self,
        embedding_dimension: int,
        bottleneck_dimension: int = 128,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dimension <= 0 or bottleneck_dimension <= 0:
            raise ValueError("Adapter dimensions must be positive.")
        self.embedding_dimension = embedding_dimension
        self.bottleneck_dimension = bottleneck_dimension
        self.alpha = alpha
        self.down = torch.nn.Linear(embedding_dimension, bottleneck_dimension)
        self.activation = torch.nn.ReLU()
        self.up = torch.nn.Linear(bottleneck_dimension, embedding_dimension)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(features)))
        return features + self.alpha * residual
