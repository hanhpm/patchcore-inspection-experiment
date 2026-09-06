"""Feature adapters for PatchCore embedding experiments."""

import torch


class IdentityFeatureAdapter(torch.nn.Module):
    """Return PatchCore embeddings unchanged for E2 control runs."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def create_feature_adapter(adapter_type: str) -> torch.nn.Module:
    if adapter_type == "identity":
        return IdentityFeatureAdapter()
    raise ValueError("Unsupported feature adapter type: {}.".format(adapter_type))
