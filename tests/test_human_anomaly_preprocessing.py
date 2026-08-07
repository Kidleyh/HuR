from types import SimpleNamespace

import numpy as np
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from astrolabe.scorers.video.human_anomaly.engine import (
    CLASSIFIER_CATEGORIES,
    CLASSIFIER_IMAGE_SIZE,
    _override_classifier_transforms,
)
from astrolabe.scorers.video.human_anomaly.schema import OFFICIAL_THRESHOLDS


def test_human_face_hand_use_direct_bicubic_224_without_center_crop():
    old_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    analyzer = SimpleNamespace(
        transforms={category: old_transform for category in CLASSIFIER_CATEGORIES}
    )

    _override_classifier_transforms(analyzer)

    square_smart_cut = np.zeros((301, 301, 3), dtype=np.uint8)
    for category in CLASSIFIER_CATEGORIES:
        operations = analyzer.transforms[category].transforms
        assert not any(isinstance(item, transforms.CenterCrop) for item in operations)
        resize = next(item for item in operations if isinstance(item, transforms.Resize))
        assert resize.size == (CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)
        assert resize.interpolation == InterpolationMode.BICUBIC
        normalize = next(
            item for item in operations if isinstance(item, transforms.Normalize)
        )
        assert tuple(normalize.mean) == IMAGENET_DEFAULT_MEAN
        assert tuple(normalize.std) == IMAGENET_DEFAULT_STD
        tensor = analyzer.transforms[category](Image.fromarray(square_smart_cut))
        assert tuple(tensor.shape) == (3, 224, 224)


def test_official_thresholds_remain_unchanged():
    assert OFFICIAL_THRESHOLDS == {
        "human": 0.4545454545454546,
        "face": 0.30303030303030304,
        "hand": 0.3232,
    }
