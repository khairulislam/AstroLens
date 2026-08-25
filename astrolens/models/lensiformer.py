# Paper: https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_214.pdf
"""Lensiformer / Lensformer (Velôso, Toomey & Gleyzer, NeurIPS ML4PS 2023): a
physics-informed Vision Transformer for classifying dark-matter substructure
(no substructure / CDM / axion) from strong gravitational lensing images.

https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Physics_Informed_Neural_Network_for_Dark_Matter_Morphology_Ashutosh_Ojha

A physics-informed encoder uses a Vision Transformer for Small Datasets
(ViTSD; Lee et al., 2021, https://arxiv.org/abs/2112.13492) with Shifted Patch
Tokenization (SPT) and Locality Self-Attention (LSA) to predict a per-patch
scale field k(x, y) for the gravitational potential ansatz of a Singular
Isothermal Sphere, Psi(x, y) = k(x, y) * sqrt(x^2 + y^2). The lens equation
S = I - grad(Psi(I)) then gives a sampling grid used to reconstruct the
source-plane image, which the decoder cross-attends against the observed
image to classify the substructure.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register_model


def shifted_views(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Concatenate `x` with 4 diagonal zero-padded shifts by `shift` pixels,
    the SPT augmentation that gives each patch token context beyond its own
    patch boundary before it is ever attended to."""
    views = [x]
    for dy, dx in ((-shift, -shift), (-shift, shift), (shift, -shift), (shift, shift)):
        pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
        shifted = F.pad(x, pad)
        h, w = x.shape[-2:]
        top = max(-dy, 0)
        left = max(-dx, 0)
        views.append(shifted[..., top : top + h, left : left + w])
    return torch.cat(views, dim=1)


class ShiftedPatchTokenizer(nn.Module):
    """SPT (Lee et al., 2021): patch-embeds the image concatenated with its
    4 diagonal shifts, then prepends a class token and adds a positional
    embedding."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans * 5, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = shifted_views(x, self.patch_size // 2)
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        return tokens + self.pos_embed


class LocalitySelfAttention(nn.Module):
    """LSA (Lee et al., 2021): multi-head attention with a learnable
    temperature in place of the fixed 1/sqrt(d) scale, and the diagonal
    masked out so each token cannot attend to itself, which otherwise
    dominates attention in small-dataset ViTs. Supports cross-attention when
    `query`/`value` differ from `key` (used by the decoder)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(self.head_dim**-0.5))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, key: torch.Tensor, query: torch.Tensor = None, value: torch.Tensor = None) -> torch.Tensor:
        if query is None:
            query = key
        if value is None:
            value = key
        n_q, n_k = query.shape[1], key.shape[1]

        q = self._split_heads(self.to_q(query))
        k = self._split_heads(self.to_k(key))
        v = self._split_heads(self.to_v(value))

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.temperature
        if n_q == n_k:
            diag_mask = torch.eye(n_q, dtype=torch.bool, device=attn.device)
            attn = attn.masked_fill(diag_mask, float("-inf"))
        attn = self.dropout(attn.softmax(dim=-1))

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(query.shape[0], n_q, -1)
        return self.to_out(out)


class LSABlock(nn.Module):
    """Pre-norm transformer block around `LocalitySelfAttention`. Defaults to
    self-attention (`key` only); pass `query`/`value` for cross-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalitySelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, key: torch.Tensor, query: torch.Tensor = None, value: torch.Tensor = None) -> torch.Tensor:
        residual = key if query is None else query
        x = residual + self.attn(self.norm1(key), query, value)
        x = x + self.mlp(self.norm2(x))
        return x


class PhysicsInformedEncoder(nn.Module):
    """Predicts a per-patch scale field k(x, y) for the Singular Isothermal
    Sphere ansatz Psi(x, y) = k(x, y) * sqrt(x^2 + y^2) with a ViTSD, then
    solves the (dimensionless) lens equation S = I - grad(Psi(I)) to warp the
    observed image into an estimate of the unlensed source.

    grad(Psi) is approximated as k(x, y) * grad(Psi_SIS(x, y)) rather than
    the full product-rule expansion, treating k as slowly varying relative to
    the analytic SIS deflection; grad(Psi_SIS) = (x, y) / sqrt(x^2 + y^2) is
    the exact SIS deflection, a unit vector field of constant magnitude.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
        k_min: float,
        k_max: float,
    ):
        super().__init__()
        self.tokenizer = ShiftedPatchTokenizer(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([LSABlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.grid_size = img_size // patch_size
        self.k_head = nn.Linear(embed_dim, 1)
        self.k_min, self.k_max = k_min, k_max

        lin = torch.linspace(-1, 1, img_size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2).clamp_min(1e-3)
        self.register_buffer("coords", torch.stack((xx, yy), dim=-1), persistent=False)
        self.register_buffer("sis_unit_deflection", torch.stack((xx / r, yy / r), dim=-1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Returns the source-plane sampling grid S = I - grad(Psi(I))."""
        tokens = self.tokenizer(image)
        for block in self.blocks:
            tokens = block(tokens)

        k_patches = torch.sigmoid(self.k_head(tokens[:, 1:])).transpose(1, 2)
        k_patches = k_patches.reshape(-1, 1, self.grid_size, self.grid_size)
        k = F.interpolate(k_patches, size=image.shape[-2:], mode="bilinear", align_corners=False)
        k = self.k_min + (self.k_max - self.k_min) * k.squeeze(1)

        deflection = k.unsqueeze(-1) * self.sis_unit_deflection
        return (self.coords - deflection).clamp(-1, 1)


class Lensiformer(nn.Module):
    """Lensformer (Velôso, Toomey & Gleyzer, 2023): physics-informed ViT for
    dark-matter substructure classification from strong lensing images.

    https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_214.pdf
    """

    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 1,
        num_classes: int = 3,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        k_min: float = 0.8,
        k_max: float = 1.2,
    ):
        super().__init__()
        self.encoder = PhysicsInformedEncoder(
            img_size, patch_size, in_chans, embed_dim, num_heads, encoder_depth, dropout, k_min, k_max
        )
        self.observed_tokenizer = ShiftedPatchTokenizer(img_size, patch_size, in_chans, embed_dim)
        self.source_tokenizer = ShiftedPatchTokenizer(img_size, patch_size, in_chans, embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [LSABlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(decoder_depth)]
        )
        num_tokens = self.observed_tokenizer.num_patches + 1
        # timm convention: num_classes=0 drops the head, forward() then returns pooled features.
        self.head = (
            nn.Sequential(nn.LayerNorm(embed_dim * num_tokens), nn.Linear(embed_dim * num_tokens, num_classes))
            if num_classes > 0
            else nn.Identity()
        )

    def reconstruct_source(self, image: torch.Tensor) -> torch.Tensor:
        """Warp `image` into an estimate of the unlensed source via the lens
        equation S = I - grad(Psi(I)), used as the decoder's physics prior."""
        grid = self.encoder(image)
        return F.grid_sample(image, grid, align_corners=False, padding_mode="zeros")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        source = self.reconstruct_source(image)

        observed_tokens = self.observed_tokenizer(image)
        source_tokens = self.source_tokenizer(source)

        x = source_tokens
        for block in self.decoder_blocks:
            x = block(key=observed_tokens, query=x)

        return self.head(x.flatten(1))


@register_model
def lensiformer(**kwargs) -> Lensiformer:
    return Lensiformer(**kwargs)
