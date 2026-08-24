# Paper: https://arxiv.org/abs/2405.14930
"""AstroPT (Smith et al., 2024): a GPT-style autoregressive transformer for
galaxy images.

Patches an image in raster order and trains a causal (GPT-2 style) transformer
to predict each patch from the ones before it, following nanoGPT
(https://github.com/karpathy/nanoGPT). The pretrained backbone is reused for
downstream tasks either through the causal embeddings (`forward_features`) or
through an optional classification head trained on top of them.

This reimplements the single-modality, native-backbone path of the reference
code (https://github.com/Smith42/astropt), including LoRA finetuning of the
attention projections (Hu et al., 2021, https://arxiv.org/abs/2106.09685) as
a hand-rolled adapter rather than the `loralib` dependency the reference uses.
The multimodal registry, LLM backbone, AION tokeniser, and masked-autoencoder
objective are out of scope.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register_model


class CausalSelfAttention(nn.Module):
    """Causal self-attention with an optional LoRA adapter on the qkv projection.

    With `lora_r > 0`, `qkv` stays frozen (see `AstroPT.mark_only_lora_as_trainable`)
    and a trainable low-rank update `(x @ A^T @ B^T) * alpha/r` is added on top,
    following the reference model's `finetune.py` recipe.
    """

    def __init__(self, dim: int, heads: int, dropout: float, bias: bool, lora_r: int = 0, lora_alpha: Optional[float] = None):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        self.lora_r = lora_r
        if lora_r > 0:
            self.lora_scaling = (lora_alpha if lora_alpha is not None else lora_r) / lora_r
            self.lora_A = nn.Parameter(torch.empty(lora_r, dim))
            self.lora_B = nn.Parameter(torch.zeros(3 * dim, lora_r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        if self.lora_r > 0:
            qkv = qkv + (x @ self.lora_A.T @ self.lora_B.T) * self.lora_scaling
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, n, self.heads, c // self.heads).transpose(1, 2)
        k = k.view(b, n, self.heads, c // self.heads).transpose(1, 2)
        v = v.view(b, n, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).reshape(b, n, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, bias: bool, lora_r: int = 0, lora_alpha: Optional[float] = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, bias=bias)
        self.attn = CausalSelfAttention(dim, heads, dropout, bias, lora_r, lora_alpha)
        self.ln_2 = nn.LayerNorm(dim, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def _spiral_perm(n: int) -> list:
    """Raster indices of an n x n grid in the reference model's spiral patch
    order (Fig. 8 of https://arxiv.org/abs/2401.08541): centre-out, ending at
    the top-left corner. Reproduces `GalaxyImageDataset._spiral`/`spiralise`
    in Smith42/astropt bit-for-bit, since a pretrained checkpoint's position
    embeddings and causal ordering are tied to that exact permutation.
    """
    grid = [[r * n + c for c in range(n)] for r in range(n)]
    rings = [None]
    while grid and grid[0]:
        rings.extend(grid[0])
        rings.extend(row[-1] for row in grid[1:])
        rings.extend(grid[-1][:-1][::-1])
        rings.extend(row[0] for row in grid[1:-1][::-1])
        grid = [row[1:-1] for row in grid[1:-1]]
    order = rings[1:]
    rank = [0] * (n * n)
    for k, idx in enumerate(order):
        rank[idx] = k
    rank = [abs(v - (n * n - 1)) for v in rank]
    return sorted(range(n * n), key=lambda i: rank[i])


class AstroPT(nn.Module):
    """AstroPT: GPT-style autoregressive transformer over image patches.

    https://arxiv.org/abs/2405.14930

    `forward` returns classification logits when `num_classes > 0` (timm
    convention), otherwise the predicted next patch at every sequence position
    (the pretraining objective). `forward_features` always returns the causal
    per-patch embeddings, for use as a pretrained backbone.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        in_chans: int = 3,
        num_classes: int = 0,
        patch_size: int = 16,
        dim: int = 384,
        depth: int = 8,
        heads: int = 6,
        dropout: float = 0.0,
        bias: bool = False,
        spiral: bool = False,
        lora_r: int = 0,
        lora_alpha: Optional[float] = None,
    ):
        super().__init__()
        img_h, img_w = (img_size, img_size) if isinstance(img_size, int) else img_size
        assert img_h % patch_size == 0 and img_w % patch_size == 0, "image dims must divide patch_size"
        self.grid = (img_h // patch_size, img_w // patch_size)
        self.patch_size = patch_size
        self.patch_dim = in_chans * patch_size**2
        num_patches = self.grid[0] * self.grid[1]

        self.spiral = spiral
        if spiral:
            assert self.grid[0] == self.grid[1], "spiral patch ordering requires a square patch grid"
            self.register_buffer(
                "patch_perm", torch.tensor(_spiral_perm(self.grid[0]), dtype=torch.long), persistent=False
            )

        self.encoder = nn.Linear(self.patch_dim, dim, bias=bias)
        self.pos_embed = nn.Embedding(num_patches, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, dropout, bias, lora_r, lora_alpha) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(dim, bias=bias)
        self.decoder = nn.Linear(dim, self.patch_dim, bias=bias)

        self.head = None
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, num_classes),
            )

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        """Extract flattened patches, each normalized to zero mean and unit
        variance independently (the reference model's `normalise` transform,
        applied in `train.py`/`train_multimodal.py` ahead of both the encoder
        and the reconstruction target): the pretrained encoder and decoder
        are calibrated to this per-patch scale, not the raw pixel range.
        """
        b, c, h, w = img.shape
        gh, gw = self.grid
        p = self.patch_size
        patches = img.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(b, gh * gw, c * p * p)
        if self.spiral:
            patches = patches[:, self.patch_perm]
        std, mean = torch.std_mean(patches, dim=-1, keepdim=True)
        return (patches - mean) / (std + 1e-8)

    def forward_features(self, img: torch.Tensor, draw_from_centre: bool = False) -> torch.Tensor:
        """Causal per-patch embeddings, shape (batch, num_patches, dim).

        With `draw_from_centre=True`, returns the middle transformer layer's
        hidden state instead of the final one: the reference model's scaling
        study finds mid-depth features probe better on downstream tasks than
        the final layer.
        """
        patches = self.patchify(img)
        pos = torch.arange(patches.shape[1], device=img.device)
        x = self.drop(self.encoder(patches) + self.pos_embed(pos))
        mid = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            x = block(x)
            if draw_from_centre and i == mid:
                return x
        return self.ln_f(x)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.head is not None:
            x = self.forward_features(img, draw_from_centre=True)
            return self.head(x.mean(dim=1))
        return self.decoder(self.forward_features(img))

    def loss(self, img: torch.Tensor) -> torch.Tensor:
        """Autoregressive next-patch prediction loss (Huber), for pretraining."""
        patches = self.patchify(img)
        pred = self.forward(img)
        return F.huber_loss(pred[:, :-1], patches[:, 1:])

    def mark_only_lora_as_trainable(self) -> None:
        """Freeze the pretrained backbone, for finetuning.

        Mirrors the reference model's `finetune.py` recipe
        (`lora.mark_only_lora_as_trainable(model)` plus an unfrozen task head),
        with one addition: `encoder` also stays trainable. Unlike the
        reference checkpoint's own patch encoder, ours can't be loaded from
        it (see `load_pretrained_backbone`), so it starts randomly
        initialized rather than pretrained and must be learned here too.
        Requires the model to have been constructed with `lora_r > 0`.
        """
        for name, param in self.named_parameters():
            param.requires_grad = (
                "lora_" in name or name.startswith("head.") or name.startswith("encoder.")
            )


@register_model
def astropt(**kwargs) -> AstroPT:
    return AstroPT(**kwargs)
