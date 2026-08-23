# Paper: https://arxiv.org/abs/2110.01024
"""Linformer (Lin et al., 2021): a linear-attention Vision Transformer for
galaxy morphology classification.

Applies a Vision Transformer to galaxy morphology classification, using
Linformer's low-rank attention approximation (Wang et al., 2020,
https://arxiv.org/abs/2006.04768) in place of standard quadratic
self-attention for efficiency.
"""

import math
from typing import Tuple, Union

import torch
from torch import nn

from ..registry import register_model


class LinformerAttention(nn.Module):
    """Self-attention with keys/values projected to a fixed low-rank length k,
    reducing attention from O(seq_len^2) to O(seq_len * k)."""

    def __init__(self, dim: int, seq_len: int, k: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.dim_head = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Parameter(torch.empty(seq_len, k).uniform_(-1 / math.sqrt(k), 1 / math.sqrt(k)))
        self.proj_v = nn.Parameter(torch.empty(seq_len, k).uniform_(-1 / math.sqrt(k), 1 / math.sqrt(k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h, d_h = self.heads, self.dim_head

        q = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)
        k = torch.einsum("bnd,nk->bkd", self.to_k(x), self.proj_k)
        v = torch.einsum("bnd,nk->bkd", self.to_v(x), self.proj_v)
        k = k.reshape(b, -1, h, d_h).transpose(1, 2)
        v = v.reshape(b, -1, h, d_h).transpose(1, 2)

        attn = torch.einsum("bhnd,bhkd->bhnk", q, k) * (d_h**-0.5)
        attn = self.dropout(attn.softmax(dim=-1))
        out = torch.einsum("bhnk,bhkd->bhnd", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, k: int, heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = LinformerAttention(dim, seq_len, k, heads, dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class Linformer(nn.Module):
    """Linformer (Lin et al., 2021): galaxy morphology classifier.

    https://arxiv.org/abs/2110.01024
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        patch_size: int = 28,
        dim: int = 128,
        depth: int = 12,
        heads: int = 8,
        k: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        img_h, img_w = (img_size, img_size) if isinstance(img_size, int) else img_size
        assert img_h % patch_size == 0 and img_w % patch_size == 0, "image dims must divide patch_size"
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        patch_dim = in_chans * patch_size**2
        seq_len = num_patches + 1

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, seq_len, k, heads, dropout) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        p = self.patch_size
        patches = img.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * p * p)

        x = self.to_patch_embedding(patches)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)

        return self.head(x[:, 0])


@register_model
def linformer(**kwargs) -> Linformer:
    return Linformer(**kwargs)
