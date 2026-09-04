"""Config-driven PatchCore experiment construction and artifact persistence."""

import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.backbones
from patchcore.augmentation import RandomIlluminationTransform
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.config import AppConfig, ConfigLoader
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset
from patchcore.device import DeviceManager


class PatchCoreFactory:
    """Build the repository's PatchCore implementation from shared config."""

    def create(
        self, config: AppConfig, device: torch.device
    ) -> patchcore.patchcore.PatchCore:
        backbone = patchcore.backbones.load(config.patchcore.backbone)
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=config.patchcore.coreset_ratio, device=device
        )
        model = patchcore.patchcore.PatchCore(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=list(config.patchcore.feature_layers),
            device=device,
            input_shape=[3, config.dataset.image_size, config.dataset.image_size],
            pretrain_embed_dimension=config.patchcore.pretrain_embedding_dimension,
            target_embed_dimension=config.patchcore.target_embedding_dimension,
            patchsize=config.patchcore.patch_size,
            anomaly_score_num_nn=config.patchcore.nearest_neighbors,
            featuresampler=sampler,
            # The installed dependency is faiss-cpu; feature extraction can still use CUDA.
            nn_method=patchcore.common.FaissNN(on_gpu=False),
        )
        return model


class ExperimentRunner:
    """Execute one configured class and store reproducibility evidence."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()

    def run(self, config_path: Path, output_root: Path) -> Path:
        config_path = config_path.resolve()
        config = ConfigLoader().load(config_path)
        if config.dataset.name != "mvtec":
            raise ValueError("The shared runner currently supports dataset.name=mvtec.")
        device = DeviceManager().resolve(config.device.requested)
        patchcore.utils.fix_seeds(
            config.experiment.seed, with_cuda=device.type == "cuda"
        )
        run_directory = self._create_run_directory(output_root, config)
        shutil.copy2(str(config_path), str(run_directory / "config.yaml"))
        self._write_json(run_directory / "environment.json", self._environment(device))
        self._write_json(run_directory / "git.json", self._git_metadata())
        self._write_git_diff(run_directory / "git.diff.patch")

        event_log = [
            "timestamp_utc={} event=run_started".format(
                datetime.now(timezone.utc).isoformat()
            ),
            "config_path={}".format(config_path),
            "device={}".format(device),
            "augmentation_enabled={}".format(config.augmentation.enabled),
            "augmentation_brightness={}".format(config.augmentation.brightness),
            "augmentation_contrast={}".format(config.augmentation.contrast),
            "augmentation_gamma={}".format(config.augmentation.gamma),
        ]
        train_dataset, test_dataset = self._datasets(config)
        event_log.extend(
            [
                "train_images={}".format(len(train_dataset)),
                "test_images={}".format(len(test_dataset)),
                "event=model_initialization_started",
            ]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=2, shuffle=False, num_workers=0
        )
        model = PatchCoreFactory().create(config, device)
        started = time.perf_counter()
        event_log.append("event=memory_bank_fit_started")
        model.fit(train_loader)
        fit_seconds = time.perf_counter() - started
        event_log.append(
            "event=memory_bank_fit_completed seconds={}".format(fit_seconds)
        )
        self._save_augmentation_samples(
            run_directory / "augmentation_samples.json", train_dataset
        )
        inference_started = time.perf_counter()
        event_log.append("event=inference_started")
        scores, anomaly_maps, labels, masks = model.predict(test_loader)
        inference_seconds = time.perf_counter() - inference_started
        event_log.append(
            "event=inference_completed seconds={}".format(inference_seconds)
        )
        runtime_seconds = time.perf_counter() - started

        image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, labels
        )["auroc"]
        pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            anomaly_maps, masks
        )["auroc"]
        aupro_result = (
            patchcore.metrics.compute_aupro(anomaly_maps, masks, fpr_limit=0.3)
            if config.evaluation.au_pro
            else None
        )
        metrics = {
            "dataset": config.dataset.name,
            "class_name": config.dataset.class_name,
            "i_auroc": float(image_auroc),
            "p_auroc": float(pixel_auroc),
            "au_pro": aupro_result["aupro"] if aupro_result else None,
            "au_pro_status": "computed" if aupro_result else "disabled",
            "au_pro_fpr_limit": aupro_result["fpr_limit"] if aupro_result else None,
            "au_pro_implementation": (
                aupro_result["implementation"] if aupro_result else None
            ),
            "au_pro_protocol": aupro_result["protocol"] if aupro_result else None,
            "au_pro_regions": (
                aupro_result["number_of_regions"] if aupro_result else None
            ),
            "runtime_seconds": runtime_seconds,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "seed": config.experiment.seed,
            "device": str(device),
            "train_images": len(train_dataset),
            "test_images": len(test_dataset),
            "augmentation": {
                "enabled": config.augmentation.enabled,
                "applied_to": "train_normal_only",
                "brightness": list(config.augmentation.brightness),
                "contrast": list(config.augmentation.contrast),
                "gamma": list(config.augmentation.gamma),
            },
        }
        if not np.isfinite([metrics["i_auroc"], metrics["p_auroc"]]).all():
            raise AssertionError("Scientific baseline produced NaN or Inf metrics.")
        self._write_json(run_directory / "metrics.json", metrics)
        np.savez_compressed(
            str(run_directory / "predictions.npz"),
            scores=np.asarray(scores),
            anomaly_maps=np.asarray(anomaly_maps),
            labels=np.asarray(labels),
            masks=np.asarray(masks),
        )
        self._save_anomaly_maps(run_directory / "anomaly_maps", anomaly_maps)
        event_log.extend(
            [
                "event=metrics_computed",
                "i_auroc={}".format(metrics["i_auroc"]),
                "p_auroc={}".format(metrics["p_auroc"]),
                "au_pro={}".format(metrics["au_pro"]),
                "event=artifacts_saved",
                "timestamp_utc={} event=run_completed status=PASS".format(
                    datetime.now(timezone.utc).isoformat()
                ),
            ]
        )
        (run_directory / "run.log").write_text(
            "\n".join(event_log) + "\n",
            encoding="utf-8",
        )
        return run_directory

    @staticmethod
    def _datasets(config: AppConfig) -> Tuple[MVTecDataset, MVTecDataset]:
        common = {
            "source": str(config.dataset.root),
            "classname": config.dataset.class_name,
            "resize": config.dataset.resize,
            "imagesize": config.dataset.image_size,
        }
        train_transform = (
            RandomIlluminationTransform(config.augmentation)
            if config.augmentation.enabled
            else None
        )
        return (
            MVTecDataset(
                split=DatasetSplit.TRAIN,
                image_transform=train_transform,
                **common,
            ),
            MVTecDataset(split=DatasetSplit.TEST, **common),
        )

    @staticmethod
    def _create_run_directory(output_root: Path, config: AppConfig) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_directory = output_root.resolve() / "{}_{}_{}".format(
            config.experiment.name, config.dataset.class_name, timestamp
        )
        run_directory.mkdir(parents=True, exist_ok=False)
        return run_directory

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _save_anomaly_maps(directory: Path, anomaly_maps: Any) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index, anomaly_map in enumerate(anomaly_maps):
            plt.imsave(
                str(directory / "{:04d}.png".format(index)),
                np.asarray(anomaly_map),
                cmap="inferno",
            )

    def _git_metadata(self) -> Dict[str, Any]:
        def git(*arguments: str) -> str:
            return subprocess.check_output(
                ["git", *arguments], cwd=str(self.repository_root), text=True
            ).strip()

        status = git("status", "--porcelain")
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines(),
        }

    def _write_git_diff(self, path: Path) -> None:
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=str(self.repository_root)
        )
        path.write_bytes(diff)

    @staticmethod
    def _save_augmentation_samples(path: Path, dataset: MVTecDataset) -> None:
        transform = dataset.image_transform
        if not isinstance(transform, RandomIlluminationTransform):
            return
        if len(transform.samples) != len(dataset):
            raise AssertionError(
                "Expected one logged augmentation sample per training image."
            )
        records = []
        for item, factors in zip(dataset.data_to_iterate, transform.samples):
            records.append({"image_path": item[2], **factors})
        path.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _environment(device: torch.device) -> Dict[str, Any]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_compiled_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        }
