from pathlib import Path

from patchcore.config import ConfigLoader


def test_baseline_config_loads_with_expected_frozen_values():
    repository_root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(repository_root / "configs/patchcore_baseline.yaml")

    assert config.experiment.seed == 42
    assert config.device.requested == "auto"
    assert config.dataset.name == "mvtec"
    assert config.dataset.class_name == "bottle"
    assert config.dataset.image_size == 224
    assert config.patchcore.backbone == "wideresnet50"
    assert config.patchcore.feature_layers == ("layer2", "layer3")
    assert config.patchcore.coreset_ratio == 0.1
    assert config.patchcore.nearest_neighbors == 1
    assert config.evaluation.au_pro is True
