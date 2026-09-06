"""Training loop for PAFA v1 Gaussian adapter experiments."""

from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

import torch

from patchcore.modules_pafa.discriminator import FeatureDiscriminator
from patchcore.modules_pafa.pseudo_anomaly import DirectedPseudoAnomalyGenerator
from patchcore.modules_pafa.pseudo_anomaly import GaussianPseudoAnomalyGenerator


@dataclass(frozen=True)
class PAFATrainingConfig:
    epochs: int = 1
    learning_rate: float = 0.0001
    gaussian_noise_std: float = 0.015
    pseudo_anomaly_mode: str = "gaussian"
    directed_step_size: float = 0.015
    directed_jitter_std: float = 0.001
    pseudo_margin: float = 0.5
    nominal_preservation_weight: float = 1.0
    discriminator_hidden_dimension: int = 256
    discriminator_loss_weight: float = 0.0


@dataclass(frozen=True)
class PAFATrainingResult:
    losses: List[float]
    loss_components: List[Dict[str, float]]
    adapter_displacement: float
    training_config: Dict[str, object]


class PAFAAdapterTrainer:
    """Train a PatchCore feature adapter with Gaussian pseudo anomalies."""

    def __init__(self, config: PAFATrainingConfig) -> None:
        if config.epochs <= 0:
            raise ValueError("PAFA epochs must be positive.")
        if config.learning_rate <= 0:
            raise ValueError("PAFA learning_rate must be positive.")
        if config.gaussian_noise_std <= 0:
            raise ValueError("PAFA gaussian_noise_std must be positive.")
        if config.pseudo_anomaly_mode not in {"gaussian", "directed"}:
            raise ValueError("PAFA pseudo_anomaly_mode must be gaussian or directed.")
        if config.directed_step_size <= 0:
            raise ValueError("PAFA directed_step_size must be positive.")
        if config.directed_jitter_std < 0:
            raise ValueError("PAFA directed_jitter_std must be non-negative.")
        if config.pseudo_margin <= 0:
            raise ValueError("PAFA pseudo_margin must be positive.")
        if config.nominal_preservation_weight < 0:
            raise ValueError("PAFA nominal_preservation_weight must be non-negative.")
        if config.discriminator_loss_weight < 0:
            raise ValueError("PAFA discriminator_loss_weight must be non-negative.")
        self.config = config

    def train(
        self, patchcore_model: torch.nn.Module, training_data
    ) -> PAFATrainingResult:
        if "feature_adapter" not in patchcore_model.forward_modules:
            raise ValueError("PAFA training requires a feature_adapter module.")

        adapter = patchcore_model.forward_modules["feature_adapter"]
        discriminator: Optional[FeatureDiscriminator] = None
        if self.config.discriminator_loss_weight > 0:
            discriminator = FeatureDiscriminator(
                patchcore_model.target_embed_dimension,
                self.config.discriminator_hidden_dimension,
            ).to(patchcore_model.device)
        generator = self._create_pseudo_anomaly_generator()
        parameters = list(adapter.parameters())
        if discriminator is not None:
            parameters += list(discriminator.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        losses = []
        loss_components = []
        displacements = []

        patchcore_model.forward_modules.eval()
        adapter.train()
        if discriminator is not None:
            discriminator.train()
        for _ in range(self.config.epochs):
            for image in training_data:
                if isinstance(image, dict):
                    image = image["image"]
                input_image = image.to(torch.float).to(patchcore_model.device)
                base_features = patchcore_model._embed(
                    input_image,
                    detach=False,
                    apply_feature_adapter=False,
                )
                pseudo_features = generator.generate(base_features.detach())
                nominal_features = adapter(base_features.detach())
                adapted_pseudo_features = adapter(pseudo_features)

                pseudo_nn_distances = self._nearest_neighbor_distances(
                    adapted_pseudo_features,
                    nominal_features.detach(),
                )
                pseudo_margin_loss = torch.relu(
                    self.config.pseudo_margin - pseudo_nn_distances
                ).mean()
                nominal_preservation_loss = torch.norm(
                    nominal_features - base_features.detach(), dim=1
                ).mean()
                loss = (
                    pseudo_margin_loss
                    + self.config.nominal_preservation_weight
                    * nominal_preservation_loss
                )
                discriminator_loss = torch.zeros((), device=loss.device)
                if discriminator is not None:
                    features = torch.cat(
                        [nominal_features, adapted_pseudo_features], dim=0
                    )
                    labels = torch.cat(
                        [
                            torch.zeros(len(nominal_features), device=features.device),
                            torch.ones(
                                len(adapted_pseudo_features), device=features.device
                            ),
                        ],
                        dim=0,
                    )
                    discriminator_loss = criterion(discriminator(features), labels)
                    loss = (
                        loss
                        + self.config.discriminator_loss_weight * discriminator_loss
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                loss_components.append(
                    {
                        "pseudo_margin_loss": float(
                            pseudo_margin_loss.detach().cpu()
                        ),
                        "nominal_preservation_loss": float(
                            nominal_preservation_loss.detach().cpu()
                        ),
                        "discriminator_loss": float(
                            discriminator_loss.detach().cpu()
                        ),
                    }
                )
                with torch.no_grad():
                    displacement = torch.norm(
                        adapter(base_features) - base_features, dim=1
                    ).mean()
                    displacements.append(float(displacement.detach().cpu()))

        adapter.eval()
        if discriminator is not None:
            discriminator.eval()
        adapter_displacement = (
            float(sum(displacements) / len(displacements)) if displacements else 0.0
        )
        return PAFATrainingResult(
            losses=losses,
            loss_components=loss_components,
            adapter_displacement=adapter_displacement,
            training_config=asdict(self.config),
        )

    def _create_pseudo_anomaly_generator(self):
        if self.config.pseudo_anomaly_mode == "gaussian":
            return GaussianPseudoAnomalyGenerator(self.config.gaussian_noise_std)
        return DirectedPseudoAnomalyGenerator(
            step_size=self.config.directed_step_size,
            jitter_std=self.config.directed_jitter_std,
        )

    @staticmethod
    def _nearest_neighbor_distances(
        query_features: torch.Tensor, memory_features: torch.Tensor
    ) -> torch.Tensor:
        if len(memory_features) == 0:
            raise ValueError("PAFA nearest-neighbor memory must not be empty.")
        distances = torch.cdist(query_features, memory_features, p=2)
        return distances.min(dim=1).values
