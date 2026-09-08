"""Feature adapters for PatchCore embedding experiments."""

import torch

from patchcore.modules_cfa.adapter import CFAFeatureAdapter
from patchcore.modules_pafa.residual_adapter import ResidualFeatureAdapter


class IdentityFeatureAdapter(torch.nn.Module):
    """Return PatchCore embeddings unchanged for E2 control runs."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def create_feature_adapter(
    adapter_type: str,
    embedding_dimension: int = None,
    bottleneck_dimension: int = 128,
    alpha: float = 1.0,
) -> torch.nn.Module:
    if adapter_type == "identity":
        return IdentityFeatureAdapter()
    if adapter_type == "cfa":
        if embedding_dimension is None:
            raise ValueError("CFA adapter requires embedding_dimension.")
        return CFAFeatureAdapter(embedding_dimension)
    if adapter_type == "pafa_residual":
        if embedding_dimension is None:
            raise ValueError("PAFA residual adapter requires embedding_dimension.")
        return ResidualFeatureAdapter(
            embedding_dimension=embedding_dimension,
            bottleneck_dimension=bottleneck_dimension,
            alpha=alpha,
        )
    raise ValueError("Unsupported feature adapter type: {}.".format(adapter_type))
