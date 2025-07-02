import torch
import torch.nn as nn

from .clip import clip 
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class LanguageGuidedAlignment(nn.Module):
    def __init__(self, clip_model, classnames=["real", "synthetic"], args=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.num_context_embedding
        ctx_init = args.init_context_embedding
        
        # n_ctx = 4
        # ctx_init = ""
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # print("clip_model.visual:", clip_model.visual)
        # print("clip_model.visual.output_dim:", clip_model.visual.output_dim)


        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
        # patch-basaed enhancer in LGA
        d_model = clip_model.ln_final.weight.shape[0]
        d_ffn = d_model * 4
        self.patch_basaed_enhancer = nn.MultiheadAttention(d_model, num_heads=args.num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        # FFN in patch-basaed enhancer
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        return tgt
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)

        # patch-basaed enhancer
        tgt = ctx[:, None].repeat_interleave(im_features.shape[0], dim=1)
        tgt2 = self.patch_basaed_enhancer(tgt, im_features.transpose(0, 1), im_features.transpose(0, 1))[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt = self.forward_FFN(tgt)

        ctx_shifted = tgt.transpose(0, 1)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CLIPModel(nn.Module):
    def __init__(self, name, args=None):
        super(CLIPModel, self).__init__()
        
        # init backbone with forgery-aware adapter
        self.clip_model = clip.load(name, device="cpu", args=args)[0] # self.preprecess will not be used during training, which is handled in Dataset class 
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
        self.text_guided_interactor = nn.MultiheadAttention(d_model, num_heads=args.num_heads)
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

    def forward(self, image):
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
        
        return logits + aug_logits
