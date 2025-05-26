import albumentations as albu
from albumentations.pytorch import ToTensorV2
from ForensicHub.core.base_transform import BaseTransform
from ForensicHub.registry import register_transform

@register_transform("DocTransform")
class DocTransform(BaseTransform):
    """Transform class for Doc tasks."""

    def __init__(self, output_size: tuple = (512, 512), norm_type='image_net'):
        super().__init__()
        self.output_size = output_size
        self.norm_type = norm_type

    def get_post_transform(self) -> albu.Compose:
        """Get post-processing transforms like normalization and conversion to tensor."""
        if self.norm_type == 'image_net':
            return albu.Compose([
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'clip':
            return albu.Compose([
                albu.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'standard':
            return albu.Compose([
                albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'none':
            return albu.Compose([
                albu.ToFloat(max_value=255.0),  # 确保 uint8 转 float32，并映射到 [0, 1]
                ToTensorV2(transpose_mask=True)
            ])
        else:
            raise NotImplementedError("Normalization type not supported, use image_net, clip, standard or none")

    def get_train_transform(self) -> albu.Compose:
        """Get training transforms."""
        return albu.Compose([
            # # Flips
            # albu.HorizontalFlip(p=0.5),
            # albu.VerticalFlip(p=0.5),
            # # Brightness and contrast fluctuation
            # albu.RandomBrightnessContrast(
            #     brightness_limit=(-0.1, 0.1),
            #     contrast_limit=0.1,
            #     p=1
            # ),
            # albu.ImageCompression(
            #     quality_lower=70,
            #     quality_upper=100,
            #     p=0.2
            # ),
            # # Rotate
            # albu.RandomRotate90(p=0.5),
            # # Blur
            # albu.GaussianBlur(
            #     blur_limit=(3, 7),
            #     p=0.2
            # )
        ])

    def get_test_transform(self) -> albu.Compose:
        """Get testing transforms."""
        return albu.Compose([
        ])
