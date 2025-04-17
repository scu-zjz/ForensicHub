import albumentations as albu
from albumentations.pytorch import ToTensorV2
from ForensicHub.core.base_transform import BaseTransform
from ForensicHub.registry import register_transform


@register_transform("AIGCTransform")
class AIGCTransform(BaseTransform):
    """Transform class for AIGC tasks."""

    def __init__(self, output_size: tuple = (224, 224)):
        super().__init__()
        self.output_size = output_size

    def get_post_transform(self) -> albu.Compose:
        """Get post-processing transforms like normalization and conversion to tensor."""
        return albu.Compose([
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(transpose_mask=True)
        ])

    def get_train_transform(self) -> albu.Compose:
        """Get training transforms."""
        return albu.Compose([
            # Flips
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # Brightness and contrast fluctuation
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower=70,
                quality_upper=100,
                p=0.2
            ),
            # Rotate
            albu.RandomRotate90(p=0.5),
            # Blur
            albu.GaussianBlur(
                blur_limit=(3, 7),
                p=0.2
            )
        ])

    def get_test_transform(self) -> albu.Compose:
        """Get testing transforms."""
        return albu.Compose([
        ])
