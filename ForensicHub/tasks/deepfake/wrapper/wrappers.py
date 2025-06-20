import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml
import os
class Deepfake2ForensicWrapper(nn.Module):
    def __init__(self, base_model_cls, yaml_config_path: str, *args, **kwargs):
        super().__init__()

        config = self._load_config_dict(yaml_config_path)
        self._init_base_model(base_model_cls, config, *args, **kwargs)

    def _load_config_dict(self, path: str):
        if not isinstance(path, str) or not os.path.isfile(path):
            raise ValueError(f"Invalid config_path: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _init_base_model(self, base_model_cls, config: dict, *args, **kwargs):
        base_model = base_model_cls(config, *args, **kwargs)

        for name, module in base_model.named_children():
            self.add_module(name, module)

        self._base_model = base_model

    def forward(self, image, label, *args, **kwargs):
        data_dict = {"image": image, "label": label}
        predictions = self._base_model(data_dict)
        losses = self._base_model.get_losses(data_dict, predictions)

        return {
            "backward_loss": losses["overall"],
            "pred_label": predictions["prob"],
            "visual_loss": losses,
        }