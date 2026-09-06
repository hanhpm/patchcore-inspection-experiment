import torch

from patchcore.adapter import create_feature_adapter


def test_identity_feature_adapter_preserves_embeddings():
    features = torch.randn(8, 256)
    adapter = create_feature_adapter("identity")

    adapted = adapter(features)

    assert torch.equal(adapted, features)
