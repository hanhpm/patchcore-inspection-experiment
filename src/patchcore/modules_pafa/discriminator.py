"""Discriminator used only while training PAFA adapters."""

import torch


class FeatureDiscriminator(torch.nn.Module):
    """Classify adapted feature vectors as nominal or pseudo-anomalous."""

    def __init__(self, embedding_dimension: int, hidden_dimension: int = 256) -> None:
        super().__init__()
        if embedding_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("Discriminator dimensions must be positive.")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
