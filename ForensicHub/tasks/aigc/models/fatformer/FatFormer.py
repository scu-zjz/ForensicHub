from .clip import clip 
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip_models import TextEncoder, LanguageGuidedAlignment

import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse

from ForensicHub.core.base_model import BaseModel
from ForensicHub.registry import register_model

'''
Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection
'''


# parser.add_argument('--num_classes', type=int, default=2)
# parser.add_argument('--num_vit_adapter', type=int, default=8)

# # text encoder
# parser.add_argument('--num_context_embedding', type=int, default=8)
# parser.add_argument('--init_context_embedding', type=str, default="")
_tokenizer = _Tokenizer()
VALID_NAMES = [
    'ViT-B/32', 
    'ViT-B/16', 
    'ViT-L/14', 
]

def get_args_parser():
    parser = argparse.ArgumentParser(' ', add_help=True)
    
    parser.add_argument('--backbone', type=str, default="ViT-B/16")
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=3)

    # text encoder
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")

    # frequency
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    
    args, remaining_args = parser.parse_known_args()
    # 获取对应的模型类
    # model_class = MODELS.get(args.model)

    return args
base_args = get_args_parser()

loss_fn = nn.CrossEntropyLoss()
@register_model("FatFormer")
class FatFormer(BaseModel):
    def __init__(self, args=base_args):
        super(FatFormer, self).__init__()
        # init backbone with forgery-aware adapter
        
        self.clip_model = clip.load(args.backbone, device=args.device, args=args)[0] # self.preprecess will not be used during training, which is handled in Dataset class 

        # self.clip_model = clip.load(name, device=args.device, args=args)[0] # self.preprecess will not be used during training, which is handled in Dataset class 
        # init language guided alignment
        self.language_guided_alignment = LanguageGuidedAlignment(self.clip_model, classnames=["real", "fake"], args=args)
        
        self.tokenized_prompts = self.language_guided_alignment.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.num_classes = args.num_classes

        # text-guided interactor in LGA
        d_model = self.clip_model.ln_final.weight.shape[0]
        d_ffn = d_model * 4
        self.text_guided_interactor = nn.MultiheadAttention(d_model, num_heads=16)
        self.norm1 = nn.LayerNorm(d_model)
        # FFN in text-guided interactor
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, image, label, **kwargs):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype), return_full=True)
        image_features_nrom = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.language_guided_alignment(image_features)
        
        # Eq.(1)
        logits = []
        text_feature_list = []
        for pts_i, imf_i in zip(prompts, image_features_nrom):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_feature_list.append(text_features)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i[0] @ text_features_norm.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        # Eq.(9)
        # text-guided interactor
        text_features = torch.stack(text_feature_list, dim=1)
        tgt = image_features[:, 1:].transpose(0, 1)
        tgt2 = self.text_guided_interactor(tgt, text_features, text_features)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt = self.forward_FFN(tgt)
        
        aug_image_features = tgt.transpose(0, 1).mean(dim=1)
        aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        aug_logits = []
        for pts_i, imf_i in zip(aug_image_features, text_features_norm.transpose(0, 1)):
            aug_logits.append(logit_scale * pts_i @ imf_i.t())
        aug_logits = torch.stack(aug_logits)
        final_logits = logits + aug_logits
        # final_logits = final_logits [:, 1] / final_logits.sum(dim=1)
        combined_loss = loss_fn(final_logits, label.long())
        pred_label = torch.softmax(logits, dim=1)[:,1]  # [B, 2]
        dict = {
            "backward_loss": combined_loss,
            "pred_label": pred_label,
            "visual_loss": {
                "combined_loss": combined_loss
            }
        }
        return dict
