"""Composable illumination transforms for normal training images."""

import random
from typing import Any, Dict, List, Tuple

from PIL import Image
from torchvision.transforms import functional

from patchcore.config import IlluminationAugmentationConfig


class RandomIlluminationTransform:
    """Apply configured brightness, contrast, and gamma perturbations."""

    def __init__(self, config: IlluminationAugmentationConfig) -> None:
        self._brightness = config.brightness
        self._contrast = config.contrast
        self._gamma = config.gamma
        self.samples: List[Dict[str, Any]] = []

    def __call__(self, image: Image.Image) -> Image.Image:
        brightness = self._sample(self._brightness)
        contrast = self._sample(self._contrast)
        gamma = self._sample(self._gamma)
        self.samples.append(
            {"brightness": brightness, "contrast": contrast, "gamma": gamma}
        )
        image = functional.adjust_brightness(image, brightness)
        image = functional.adjust_contrast(image, contrast)
        return functional.adjust_gamma(image, gamma)

    @staticmethod
    def _sample(value_range: Tuple[float, float]) -> float:
        return random.uniform(value_range[0], value_range[1])
