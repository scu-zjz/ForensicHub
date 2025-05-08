import json
import os
import random
import torch
import numpy as np
from PIL import Image
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


@register_dataset("IMDLCrossDataset")
class IMDLCrossDataset(BaseDataset):
    def __init__(self, path, image_size=224, **kwargs):
        self.image_size = image_size
        super().__init__(path=path, **kwargs)

    def _init_dataset_path(self) -> None:
        """Read JSON file and parse image paths, masks, and labels."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"JSON file not found at {self.path}")

        with open(self.path, 'r') as f:
            data = json.load(f)

        self.samples = data
        # Expecting a list of {"path": ..., "mask": ..., "label": ...} mask is "Negative" if label is 0

        self.entry_path = self.path  # For __str__()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["path"]
        mask_path = sample["mask"]
        label = sample["label"]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)
        if mask_path == "Negative":
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            mask = Image.open(mask_path).convert("L")  # 'L'模式表示灰度图
            mask = mask.resize((self.image_size, self.image_size))
            mask = np.array(mask)
            mask = mask / 255.

        # Apply transforms
        if self.common_transform:
            out = self.common_transform(image=image, mask=mask)
            image = out['image']
            mask = out['mask']
        if self.post_transform:
            out = self.post_transform(image=image, mask=mask)
            image = out['image']
            mask = out['mask']

        output = {
            "image": image,
            "mask": mask.float(),
            "label": torch.tensor(label).long()
        }

        # Apply post-processing functions
        if self.post_funcs:
            self.post_funcs(output)

        return output

    def __str__(self):
        label_0 = sum(1 for s in self.samples if s["label"] == 0)
        label_1 = sum(1 for s in self.samples if s["label"] == 1)
        return (f"IMDLCrossDataset from: {self.entry_path}\n"
                f"Total samples: {len(self.samples)}\n"
                f"Label 0 (real): {label_0}\n"
                f"Label 1 (fake): {label_1}")
