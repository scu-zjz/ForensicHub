import json
import os
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset
from PIL import Image
import numpy as np


@register_dataset("AIGCLabelDataset")
class AIGCLabelDataset(BaseDataset):
    """Dataset for binary classification of real vs AIGC-generated images."""

    def __init__(self,
                 path: str,
                 image_size: int = 224,
                 gen_mask: bool = False,
                 **kwargs):
        self.image_size = image_size
        self.gen_mask = gen_mask
        super().__init__(path=path, **kwargs)

    def _init_dataset_path(self) -> None:
        """Read JSON file and parse image paths and labels."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"JSON file not found at {self.path}")

        with open(self.path, 'r') as f:
            data = json.load(f)

        self.samples = data  # Expecting a list of {"path": ..., "label": ...}
        self.entry_path = self.path  # For __str__()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = sample["path"]
        label = sample["label"]

        # Resolve full path if needed
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.path), image_path)

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)

        # Apply transforms
        if self.common_transform:
            image = self.common_transform(image=image)['image']
        if self.post_transform:
            image = self.post_transform(image=image)['image']

        output = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float)
        }

        if self.gen_mask:
            mask = torch.full((image.shape[1], image.shape[2]), fill_value=label, dtype=torch.float)
            output["mask"] = mask

        # Apply post-processing functions
        if self.post_funcs:
            self.post_funcs(output)

        return output
