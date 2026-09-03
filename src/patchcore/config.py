"""Typed configuration loading for reproducible PatchCore experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DeviceConfig:
    requested: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    class_name: str
    resize: int
    image_size: int
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass(frozen=True)
class PatchCoreConfig:
    backbone: str
    feature_layers: Tuple[str, ...]
    coreset_ratio: float
    pretrain_embedding_dimension: int
    target_embedding_dimension: int
    nearest_neighbors: int
    patch_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    image_auroc: bool
    pixel_auroc: bool
    au_pro: bool


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig
    device: DeviceConfig
    dataset: DatasetConfig
    patchcore: PatchCoreConfig
    evaluation: EvaluationConfig


class ConfigLoader:
    """Load and validate the shared YAML configuration."""

    def load(self, path: Path) -> AppConfig:
        with path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a mapping.")
        return self._build(raw)

    def _build(self, raw: Dict[str, Any]) -> AppConfig:
        try:
            experiment = ExperimentConfig(**raw["experiment"])
            device = DeviceConfig(**raw["device"])
            dataset = DatasetConfig(
                **{**raw["dataset"], "root": Path(raw["dataset"]["root"])}
            )
            patchcore_raw = dict(raw["patchcore"])
            patchcore_raw["feature_layers"] = tuple(patchcore_raw["feature_layers"])
            patchcore = PatchCoreConfig(**patchcore_raw)
            evaluation = EvaluationConfig(**raw["evaluation"])
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid PatchCore configuration: {}".format(error))

        self._validate(experiment, device, dataset, patchcore)
        return AppConfig(experiment, device, dataset, patchcore, evaluation)

    @staticmethod
    def _validate(
        experiment: ExperimentConfig,
        device: DeviceConfig,
        dataset: DatasetConfig,
        patchcore: PatchCoreConfig,
    ) -> None:
        if experiment.seed < 0:
            raise ValueError("Experiment seed must be non-negative.")
        if device.requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda.")
        if dataset.resize <= 0 or dataset.image_size <= 0:
            raise ValueError("Dataset image dimensions must be positive.")
        if not 0 < patchcore.coreset_ratio <= 1:
            raise ValueError("Coreset ratio must be in the interval (0, 1].")
        if not patchcore.feature_layers:
            raise ValueError("At least one feature layer is required.")
        if patchcore.nearest_neighbors <= 0 or patchcore.patch_size <= 0:
            raise ValueError("Nearest neighbors and patch size must be positive.")
