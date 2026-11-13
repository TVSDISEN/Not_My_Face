"""Training-only image augmentations for upload and compression artifacts."""

from __future__ import annotations


def build_train_augmentation():
    """Create the stochastic augmentation pipeline used only during training."""
    try:
        import albumentations as A
    except ImportError as error:
        raise RuntimeError(
            "Albumentations is required; install it with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    return A.Compose(
        [
            A.ImageCompression(quality_range=(25, 95), p=0.65),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ],
                p=0.55,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.08,
                p=0.6,
            ),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.35),
            A.HorizontalFlip(p=0.5),
        ]
    )
