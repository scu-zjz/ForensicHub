import json
import os
import random
import torch
import numpy as np
from PIL import Image
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


@register_dataset("AIGCCrossDataset")
class AIGCCrossDataset(BaseDataset):
    def __init__(self, path, image_size=224, **kwargs):
        self.image_size = image_size
        super().__init__(path=path, **kwargs)

    def _init_dataset_path(self) -> None:

        """Read JSON file(s) and parse image paths, and labels."""
        if isinstance(self.path, str):
            path_list = [self.path]
        elif isinstance(self.path, list):
            path_list = self.path
        else:
            raise TypeError(f"path should be a str or list of str, but got {type(self.path)}")

        self.samples = []
        for json_path in path_list:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found at {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.samples.extend(data)
        # Expecting a list of {"path": ..., "label": ...}

        self.entry_path = ','.join(path_list)  # For __str__()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["path"]
        label = sample["label"]

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image)
        except Exception as e:
            print(f"[!] Failed to load image: {image_path} ({e})")
            # 返回一个空样本或递归调用获取下一个（避免死循环）
            return self.__getitem__((idx + 1) % len(self))

        # Apply transforms
        if self.common_transform:
            out = self.common_transform(image=image)
            image = out['image']
        if self.post_transform:
            out = self.post_transform(image=image)
            image = out['image']

        output = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float)
        }

        return output
