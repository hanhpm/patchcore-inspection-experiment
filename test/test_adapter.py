import torch
from torchvision import models

from patchcore.adapter import create_feature_adapter
from patchcore.modules_cfa.trainer import CFAAdapterTrainer
from patchcore.modules_cfa.trainer import CFATrainingConfig
from patchcore.modules_pafa.pseudo_anomaly import GaussianPseudoAnomalyGenerator
from patchcore.modules_pafa.trainer import PAFAAdapterTrainer
from patchcore.modules_pafa.trainer import PAFATrainingConfig
from patchcore.patchcore import PatchCore


def test_identity_feature_adapter_preserves_embeddings():
    features = torch.randn(8, 256)
    adapter = create_feature_adapter("identity")

    adapted = adapter(features)

    assert torch.equal(adapted, features)


def test_pafa_residual_adapter_starts_as_identity():
    features = torch.randn(8, 256)
    adapter = create_feature_adapter(
        "pafa_residual",
        embedding_dimension=256,
        bottleneck_dimension=32,
        alpha=1.0,
    )

    adapted = adapter(features)

    assert torch.equal(adapted, features)


def test_cfa_adapter_starts_as_identity():
    features = torch.randn(8, 256)
    adapter = create_feature_adapter("cfa", embedding_dimension=256)

    adapted = adapter(features)

    assert torch.allclose(adapted, features)


def test_gaussian_pseudo_anomaly_preserves_shape():
    torch.manual_seed(0)
    features = torch.zeros(8, 16)
    generator = GaussianPseudoAnomalyGenerator(noise_std=0.015)

    pseudo_features = generator.generate(features)

    assert pseudo_features.shape == features.shape
    assert not torch.equal(pseudo_features, features)


def test_pafa_trainer_runs_on_patchcore_embeddings():
    torch.manual_seed(0)
    image_dimension = 32
    model = PatchCore(torch.device("cpu"))
    backbone = models.wide_resnet50_2(pretrained=False)
    backbone.name, backbone.seed = "wideresnet50", 0
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        input_shape=[3, image_dimension, image_dimension],
        pretrain_embed_dimension=64,
        target_embed_dimension=64,
        patchsize=3,
        feature_adapter=create_feature_adapter(
            "pafa_residual",
            embedding_dimension=64,
            bottleneck_dimension=16,
        ),
        feature_adapter_type="pafa_residual",
    )
    model.set_feature_adapter_trainer(
        PAFAAdapterTrainer(
            PAFATrainingConfig(
                epochs=1,
                learning_rate=0.001,
                gaussian_noise_std=0.015,
                discriminator_hidden_dimension=16,
            )
        )
    )
    images = torch.rand([2, 3, image_dimension, image_dimension])
    dataloader = torch.utils.data.DataLoader(images, batch_size=1)

    result = model.train_feature_adapter(dataloader)

    assert len(result.losses) == 2
    assert len(result.loss_components) == 2
    assert "pseudo_margin_loss" in result.loss_components[0]
    assert "nominal_preservation_loss" in result.loss_components[0]
    assert result.loss_components[0]["discriminator_loss"] == 0
    assert result.adapter_displacement >= 0


def test_cfa_trainer_runs_on_patchcore_embeddings():
    torch.manual_seed(0)
    image_dimension = 32
    model = PatchCore(torch.device("cpu"))
    backbone = models.wide_resnet50_2(pretrained=False)
    backbone.name, backbone.seed = "wideresnet50", 0
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        input_shape=[3, image_dimension, image_dimension],
        pretrain_embed_dimension=64,
        target_embed_dimension=64,
        patchsize=3,
        feature_adapter=create_feature_adapter("cfa", embedding_dimension=64),
        feature_adapter_type="cfa",
    )
    model.set_feature_adapter_trainer(
        CFAAdapterTrainer(
            CFATrainingConfig(
                epochs=1,
                learning_rate=0.001,
                k_neighbors=1,
                j_neighbors=1,
            )
        )
    )
    images = torch.rand([2, 3, image_dimension, image_dimension])
    dataloader = torch.utils.data.DataLoader(images, batch_size=1)

    result = model.train_feature_adapter(dataloader)

    assert len(result.losses) == 2
    assert "attraction_loss" in result.loss_components[0]
    assert "repulsion_loss" in result.loss_components[0]
    assert result.adapter_displacement >= 0


def test_pafa_nearest_neighbor_distances_are_euclidean():
    query_features = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    memory_features = torch.tensor([[0.0, 1.0], [10.0, 10.0]])

    distances = PAFAAdapterTrainer._nearest_neighbor_distances(
        query_features, memory_features
    )

    assert torch.allclose(distances, torch.tensor([1.0, 18.0**0.5]))
