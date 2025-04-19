import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import torch.nn.functional as F

from .guided_diffusion import dist_util, logger
from .guided_diffusion.image_datasets import load_data_for_reverse
from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model
from ForensicHub.common.backbones.resnet import Resnet50


@register_model("Dire")
class Dire(BaseModel):
    def __init__(self,
                 model_path="/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/tasks/aigc/models/dire/lsun_bedroom.pt",
                 backbone="resnet50"):
        super().__init__()
        self.backbone = backbone

        self.classifier = None
        if backbone == 'resnet50':
            self.classifier = Resnet50(pretrained=False, num_classes=1)
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported!")

        use_fp16 = True if self.training else False
        self.model_args = dict(
            attention_resolutions="32,16,8",
            class_cond=False,
            diffusion_steps=1000,
            dropout=0.1,
            image_size=256,
            learn_sigma=True,
            noise_schedule="linear",
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_fp16=use_fp16,
            use_scale_shift_norm=True,
            timestep_respacing="ddim20",  # 用于 ddim 重建
            channel_mult="",
            num_heads=4,
            num_heads_upsample=-1,
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_new_attention_order=False,
        )
        self.use_ddim = True

        self.model, self.diffusion = create_model_and_diffusion(**self.model_args)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        if self.model_args["use_fp16"]:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def compute_dire_value(self, image: torch.Tensor) -> torch.Tensor:
        """
        给定图像张量，计算 DiRe 值（图像 - 重构图像）
        """

        latent = self.diffusion.ddim_reverse_sample_loop(
            self.model,
            image.shape,
            noise=image,
            clip_denoised=True,
            model_kwargs={},
            real_step=0
        )

        sample_fn = self.diffusion.ddim_sample_loop if self.use_ddim else self.diffusion.p_sample_loop
        recons = sample_fn(
            self.model,
            (image.size(0), 3, self.model_args["image_size"], self.model_args["image_size"]),
            noise=latent,
            clip_denoised=True,
            model_kwargs={},
            real_step=0
        )

        dire = torch.abs(image - recons)
        return dire

    def forward(self, image: torch.Tensor, label: torch.Tensor, **kwargs):
        label = label.float()

        dire = self.compute_dire_value(image)
        data_dict = self.classifier(dire, label)

        return data_dict
