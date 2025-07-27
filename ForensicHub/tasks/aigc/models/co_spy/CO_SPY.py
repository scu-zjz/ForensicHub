import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torchvision import transforms
from typing import Dict, Any


from .cospy_utils import data_augment
from .semantic_detector import SemanticDetector
from .artifact_detector import ArtifactDetector
from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

# CO-SPY Detector
@register_model("CO_SPY")
class CO_SPY(BaseModel):
    def __init__(self, clip_model="ViT-L-14", pretrained="datacomp_xl_s13b_b90k", dim_clip=768,num_classes=1):
        super(CO_SPY, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector(clip_model=clip_model, pretrained=pretrained, dim_clip=dim_clip, num_classes=num_classes)
        self.sem_dim = self.sem.fc.in_features

        # Load the artifact detector
        self.art = ArtifactDetector()
        self.art_dim = self.art.fc.in_features

        # Classifier
        self.fc = torch.nn.Linear(self.sem_dim + self.art_dim, num_classes)

        # Transformations inside the forward function
        # Including the normalization and resizing (only for the artifact detector)
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Resolution
        self.loadSize = 224
        self.cropSize = 224

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            flip_func,
            aug_func,
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

    def forward(self, image, label, **kwargs) -> Dict[str, Any]:
        x = image
        label = label.float()
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)

        # Forward pass
        sem_feat, sem_coeff = self.sem(x_sem, return_feat=True)
        art_feat, art_coeff = self.art(x_art, return_feat=True)

        # Dropout
        if self.train():
            # Random dropout
            if random.random() < 0.3:
                # Randomly select a feature to drop
                idx_drop = random.randint(0, 1)
                if idx_drop == 0:
                    sem_coeff = torch.zeros_like(sem_coeff)
                else:
                    art_coeff = torch.zeros_like(art_coeff)

        # Concatenate the features
        x = torch.cat([sem_coeff * sem_feat, art_coeff * art_feat], dim=1)
        x = self.fc(x)

        x = x.squeeze(dim=1) if x.ndim == 2 and x.shape[1] == 1 else x
        combined_loss = F.binary_cross_entropy_with_logits(x, label)
        pred_label = x.sigmoid()

        dict = {
            "backward_loss": combined_loss,
            "pred_label": pred_label,
            "visual_loss": {
                "combined_loss": combined_loss
            }
        }
        return dict






# Define the label smoothing loss
class LabelSmoothingBCEWithLogits(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCEWithLogits, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        target = target.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss

