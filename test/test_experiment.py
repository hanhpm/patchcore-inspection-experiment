from pathlib import Path
from dataclasses import replace

import torch

from experiments_scripts.config.config import AdapterConfig
from experiments_scripts.config.config import ConfigLoader
from experiments_scripts.experiment import ExperimentRunner, PatchCoreFactory


def test_factory_builds_frozen_cpu_model_from_shared_config():
    root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(root / "configs/patchcore_cpu_smoke.yaml")
    model = PatchCoreFactory().create(config, torch.device("cpu"))

    assert model.device.type == "cpu"
    assert model.layers_to_extract_from == ["layer2", "layer3"]
    assert model.feature_adapter_type == "none"
    assert "feature_adapter" not in model.forward_modules
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_factory_adds_identity_adapter_when_enabled():
    root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(root / "configs/patchcore_cpu_smoke.yaml")
    config = replace(config, adapter=AdapterConfig(enabled=True, type="identity"))

    model = PatchCoreFactory().create(config, torch.device("cpu"))

    assert model.feature_adapter_type == "identity"
    assert "feature_adapter" in model.forward_modules


def test_factory_adds_cfa_trainer_when_enabled():
    root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(root / "configs/patchcore_cpu_smoke.yaml")
    config = replace(config, adapter=AdapterConfig(enabled=True, type="cfa", epochs=1))

    model = PatchCoreFactory().create(config, torch.device("cpu"))

    assert model.feature_adapter_type == "cfa"
    assert "feature_adapter" in model.forward_modules
    assert model.feature_adapter_trainer is not None


def test_run_directory_contains_experiment_and_class(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = ConfigLoader().load(root / "configs/patchcore_cpu_smoke.yaml")

    run_directory = ExperimentRunner._create_run_directory(tmp_path, config)

    assert run_directory.parent == tmp_path
    assert "patchcore_cpu_smoke_bottle_" in run_directory.name
