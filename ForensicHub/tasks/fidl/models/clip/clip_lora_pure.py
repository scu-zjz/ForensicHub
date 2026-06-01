from typing import Dict, Any, Optional, List, Union, Set
import contextlib
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
import timm

from .models.clip import clip
from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "vit_large_patch14_dinov2.lvd142m": 1024,  # DINOv2
    "vit_large_patch16_siglip_256": 1024,  # SigLIP
}


def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def _parse_qkv_spec(spec: str) -> Set[str]:
    """
    Accepts: "qkv", "q", "k", "v", "qv", "q,k,v", "QKV"...
    Returns set of {"q","k","v"}.
    """
    s = spec.lower().replace(" ", "")
    if s in {"all", "qkv"}:
        return {"q", "k", "v"}
    s = s.replace(",", "")
    allowed = set()
    for ch in s:
        if ch in {"q", "k", "v"}:
            allowed.add(ch)
    if len(allowed) == 0:
        raise ValueError(f"Bad lora_qkv spec={spec}, expected subset of q/k/v or 'qkv'")
    return allowed


class LoRAParametrization(nn.Module):
    """
    ΔW = (B @ A) * scaling
    Optional:
      - row_mask: shape [out_features, 1], to restrict which rows are updated (e.g., only V rows in QKV).
    """

    def __init__(
            self,
            out_features: int,
            in_features: int,
            r: int = 8,
            alpha: float = 16.0,
            row_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.r = int(r)
        self.scaling = float(alpha / r)
        self.enabled = True  # analysis-time switch (python bool)

        self.A = nn.Parameter(torch.zeros(self.r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

        # row mask to select Q/K/V rows (or any subset)
        if row_mask is not None:
            # expect [out, 1] float mask (0/1)
            self.register_buffer("row_mask", row_mask.float())
        else:
            self.row_mask = None

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return W
        dW = (self.B @ self.A) * self.scaling
        if self.row_mask is not None:
            dW = dW * self.row_mask
        return W + dW


@register_model("CLIP_LORA_PURE")
class CLIP_LORA_PURE(BaseModel):
    """
    Kept features:
      (1) choose Q/K/V on in_proj_weight: lora_qkv
      (2) optionally disable out_proj LoRA: lora_apply_out_proj
    Removed:
      - gate per LoRA module
      - layer selection (lora_layers) => now inject into ALL visual transformer layers
    """

    def __init__(
            self,
            name: str = "ViT-L/14",
            num_classes: int = 1,

            # LoRA base
            lora_r: int = 8,
            lora_alpha: float = 16.0,

            # (kept) Q/K/V selection for in_proj_weight
            # e.g. "qkv"(default), "v", "q", "k", "qv", "q,k"
            lora_qkv: str = "qkv",

            # (kept) optionally disable out_proj LoRA
            lora_apply_out_proj: bool = True,

            # training mode
            tune_mode: str = "lora",
    ):
        super().__init__()
        self.is_timm = False
        if name in CHANNELS and "ViT" not in name and "RN" not in name:
            # 这是一个简单的判断逻辑，只要名字看起来像 timm 的名字
            # 或者你可以显式加一个参数 model_source="timm"
            self.is_timm = True
            # 加载 timm 模型 (仅作为特征提取器)
            print(f"[Init] Loading via timm: {name}")
            # self.model = timm.create_model(name, pretrained=True, num_classes=0)
            self.model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)

            data_config = timm.data.resolve_data_config(self.model.default_cfg, model=self.model)
            data_config['input_size'] = (3, 224, 224)

            self.preprocess = timm.data.create_transform(**data_config, is_training=False)

            feat_dim = self.model.num_features
        else:
            # 原有的 CLIP 加载逻辑
            self.model, self.preprocess = clip.load(name, device="cpu")
            feat_dim = CHANNELS[name]
        self.fc = nn.Linear(CHANNELS[name], num_classes)

        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)

        self.lora_apply_out_proj = bool(lora_apply_out_proj)
        self.lora_qkv_set = _parse_qkv_spec(lora_qkv)

        self._lora_params: List[LoRAParametrization] = []
        self._inject_lora_into_visual_attention()

        self.tune_mode = None
        self.set_tune_mode(tune_mode)

        print(f"[Init] tune_mode={self.tune_mode} num_lora_modules={len(self._lora_params)} "
              f"lora_qkv={lora_qkv} out_proj={self.lora_apply_out_proj}")

    def _make_qkv_row_mask(self, out_features: int, in_features: int) -> torch.Tensor:
        """
        For ViT-L/14 MHA in_proj_weight: shape [3d, d]
        mask rows to select q/k/v blocks.
        """
        mask = torch.zeros(out_features, 1)
        if out_features % 3 != 0:
            mask[:] = 1.0
            return mask

        d = out_features // 3
        if "q" in self.lora_qkv_set:
            mask[0:d, 0] = 1.0
        if "k" in self.lora_qkv_set:
            mask[d:2 * d, 0] = 1.0
        if "v" in self.lora_qkv_set:
            mask[2 * d:3 * d, 0] = 1.0
        return mask

    def _inject_lora_into_visual_attention(self):
        """
        Inject LoRA into ALL visual transformer resblocks (no layer selection).
        """
        blocks = None

        # Case A: OpenAI CLIP
        if hasattr(self.model, "visual"):
            visual = getattr(self.model, "visual", None)
            if hasattr(visual, "transformer"):
                blocks = visual.transformer.resblocks

        # Case B: Timm ViT (DINO, SigLIP, etc.)
        elif hasattr(self.model, "blocks"):
            blocks = self.model.blocks

        if blocks is None:
            print("Warning: Could not find transformer blocks.")
            return

        for li, blk in enumerate(blocks):
            layer_id = li + 1

            # --- 寻找 Attention 模块 ---
            attn_layer = getattr(blk, "attn", None)
            if attn_layer is None:
                continue

            # --- 注入逻辑分支 ---

            # 分支 1: OpenAI CLIP 风格 (nn.MultiheadAttention)
            if isinstance(attn_layer, nn.MultiheadAttention):
                # Handle in_proj_weight (Q, K, V packed into one matrix)
                if hasattr(attn_layer, "in_proj_weight"):
                    target_weight = attn_layer.in_proj_weight  # shape [3*dim, dim]
                    out_features, in_features = target_weight.shape

                    # 制作 mask 以区分 Q/K/V
                    row_mask = self._make_qkv_row_mask(out_features, in_features)

                    lp = LoRAParametrization(
                        out_features=out_features,
                        in_features=in_features,
                        r=self.lora_r,
                        alpha=self.lora_alpha,
                        row_mask=row_mask,
                    )
                    lp.layer_id = layer_id
                    lp.lora_name = "in_proj_weight"

                    # 注册 LoRA 到 in_proj_weight
                    parametrize.register_parametrization(attn_layer, "in_proj_weight", lp)
                    self._lora_params.append(lp)

                # Handle out_proj (Linear layer)
                if self.lora_apply_out_proj and hasattr(attn_layer, "out_proj"):
                    # out_proj is usually a NonDynamicallyQuantizableLinear (subclass of Linear)
                    target_out = attn_layer.out_proj
                    lp_out = LoRAParametrization(
                        out_features=target_out.out_features,
                        in_features=target_out.in_features,
                        r=self.lora_r,
                        alpha=self.lora_alpha,
                        row_mask=None,  # no mask needed for output projection
                    )
                    parametrize.register_parametrization(target_out, "weight", lp_out)
                    self._lora_params.append(lp_out)

            # 分支 2: Timm 风格 (Linear qkv)
            elif hasattr(attn_layer, "qkv") and isinstance(attn_layer.qkv, nn.Linear):
                # ... (这一部分你原本的代码是对的，保持不变) ...
                target_linear = attn_layer.qkv
                out_features = target_linear.out_features
                in_features = target_linear.in_features
                row_mask = self._make_qkv_row_mask(out_features, in_features)

                lp = LoRAParametrization(
                    out_features=out_features,
                    in_features=in_features,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    row_mask=row_mask,
                )
                lp.layer_id = layer_id
                lp.lora_name = "qkv.weight"
                lp.qkv = "".join(sorted(list(self.lora_qkv_set)))

                parametrize.register_parametrization(target_linear, "weight", lp)
                self._lora_params.append(lp)

                # 处理 out_proj (Timm 里通常叫 .proj)
                if self.lora_apply_out_proj and hasattr(attn_layer, "proj") and isinstance(attn_layer.proj, nn.Linear):
                    lp2 = LoRAParametrization(
                        out_features=attn_layer.proj.out_features,
                        in_features=attn_layer.proj.in_features,
                        r=self.lora_r,
                        alpha=self.lora_alpha,
                        row_mask=None,
                    )
                    parametrize.register_parametrization(attn_layer.proj, "weight", lp2)
                    self._lora_params.append(lp2)

    # -------- analysis-time LoRA toggle --------
    def _set_lora_forward_enabled(self, enabled: bool):
        for lp in self._lora_params:
            lp.enabled = bool(enabled)

    @contextlib.contextmanager
    def lora_disabled(self):
        old = [lp.enabled for lp in self._lora_params]
        self._set_lora_forward_enabled(False)
        try:
            yield
        finally:
            for lp, v in zip(self._lora_params, old):
                lp.enabled = v

    @contextlib.contextmanager
    def lora_enabled(self):
        old = [lp.enabled for lp in self._lora_params]
        self._set_lora_forward_enabled(True)
        try:
            yield
        finally:
            for lp, v in zip(self._lora_params, old):
                lp.enabled = v

    # -------- training mode policy --------
    def set_tune_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in {"lp", "lora", "fft"}:
            raise ValueError(f"Unknown tune_mode={mode}, expected one of: lp|lora|fft")
        self.tune_mode = mode

        if mode == "lp":
            _set_requires_grad(self.model, False)
            self._set_lora_forward_enabled(False)
            for lp in self._lora_params:
                lp.A.requires_grad_(False)
                lp.B.requires_grad_(False)

        elif mode == "lora":
            _set_requires_grad(self.model, False)
            self._set_lora_forward_enabled(True)
            if len(self._lora_params) == 0:
                raise ValueError("tune_mode='lora' but no LoRA modules were injected (are you using RN50?)")
            for lp in self._lora_params:
                lp.A.requires_grad_(True)
                lp.B.requires_grad_(True)

        elif mode == "fft":
            _set_requires_grad(self.model, True)
            self._set_lora_forward_enabled(False)
            for lp in self._lora_params:
                lp.A.requires_grad_(False)
                lp.B.requires_grad_(False)

        _set_requires_grad(self.fc, True)

    def forward(self, image, label: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        x = image
        y = None if label is None else label.float()

        if self.is_timm:
            # Timm 模型通常输出 features [B, N, C] 或 [B, C]
            # 这里取 CLS token (通常是第一个) 或者 GAP
            feat = self.model.forward_features(x)
            if feat.ndim == 3:
                # 假设第一个 token 是 CLS (DINO/ViT 都是这样)
                # 如果是只做 GAP 的模型 (如某些 CNN)，timm 会直接返回 2D
                feat = feat[:, 0, :]
        else:
            feat = self.model.encode_image(x)
        logit = self.fc(feat)
        logit = logit.squeeze(dim=1) if logit.ndim == 2 and logit.shape[1] == 1 else logit
        pred_label = torch.sigmoid(logit)

        if y is None:
            return {"pred_label": pred_label, "visual_loss": {}}

        bce_main = F.binary_cross_entropy_with_logits(logit, y)
        total = bce_main

        return {
            "backward_loss": total,
            "pred_label": pred_label,
            "visual_loss": {
                "total_loss": total.detach(),
                "bce_main": bce_main.detach(),
            }
        }
