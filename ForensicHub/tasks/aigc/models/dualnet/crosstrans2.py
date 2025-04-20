from einops import rearrange
import torch
import torch.nn as nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, noise, rgb, **kwargs):
        n, r = self.fn(self.norm(noise), self.norm(rgb), **kwargs)
        return noise+n, rgb+r



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, noise, rgb):
        return self.net1(noise), self.net2(rgb)


# remove the embedding and projection,because 128 is small enough to compute
class Modality_attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super(Modality_attention, self).__init__()
        in_dim = heads * head_dim
        project_out = not(heads == 1 and dim == in_dim)
        self.noise_kv = nn.Linear(in_dim, 2 * in_dim)
        self.rgb_kv = nn.Linear(in_dim, 2 * in_dim)
        self.softmax = nn.Softmax(dim=1)
        self.scale = head_dim ** -0.5
        self.heads = heads
        self.out1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.out2 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, nv, rv):
        b, n, c = nv.size()
        h = self.heads
        nqk = self.noise_kv(nv).chunk(2, dim=-1)
        noise_q, noise_k = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h = h), nqk)
        rgb_qk = self.rgb_kv(rv).chunk(2, dim=-1)
        rgb_q, rgb_k = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), rgb_qk)
        #b h n d x b h d n -> b h x n x n
        attn1 = (noise_q @ rgb_k.transpose(-1, -2)) * self.scale
        attn2 = (rgb_q @ noise_k.transpose(-1, -2)) * self.scale
        #b h n n  / b h
        rv = rearrange(rv, "b n (h d) -> b h n d", h = h)
        nv = rearrange(nv, "b n (h d) -> b h n d", h = h)
        noise = (self.softmax(attn1) @ rv)
        rgb = (self.softmax(attn2) @ nv)
        noise = rearrange(noise, "b h n d -> b n (h d)")
        rgb = rearrange(rgb, "b h n d -> b n (h d)")
        #noise = self.out1(noise)
        #rgb = self.out2(rgb)
        return noise, rgb

class crosstrans(nn.Module):
    def __init__(self, depth, dim, hidden_dim, heads, head_dim, dropout):
        super(crosstrans, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
            [PreNorm(dim, Modality_attention(dim=dim, heads=heads, head_dim=head_dim, dropout=dropout)),
             PreNorm(dim, FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout))]
            ))

    def forward(self, noise, rgb):
        b, c, h, w = noise.size()
        n = rearrange(noise, "b c h w -> b (h w) c")
        r = rearrange(rgb, "b c h w -> b (h w) c")
        for attn, ffn in self.layers:
            n, r = attn(n, r)
            n, r = ffn(n, r)
        n = n.transpose(-1, -2).view(b, c, h, w)
        r = r.transpose(-1, -2).view(b, c, h, w)
        return n, r
