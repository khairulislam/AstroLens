# Paper: https://arxiv.org/abs/2311.01500
"""GCNN (Pandya et al., 2023): a group-equivariant CNN for robust galaxy
morphology classification, equivariant to discrete image rotations (cyclic
group C_N) and, optionally, reflections (dihedral group D_N).

https://github.com/snehjp2/GCNNMorphology

Group convolution (Cohen & Welling, 2016) implemented natively in PyTorch:
each layer holds one base filter bank and derives one spatially transformed
copy per group element by permuting input-orientation channels and
rotating/flipping the kernel. Rotation uses bilinear kernel interpolation
(`F.grid_sample`), exact for N in {1, 2, 4} and approximate otherwise.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register_model

# feature_fields per block, from the reference implementation.
DEFAULT_FEATURE_FIELDS = [12, 24, 48, 48, 48, 48, 96, 96, 96, 112, 192]

# (kernel_size, padding, stride, pool_after) for each of the 11 conv blocks.
_BLOCK_CONFIG = [
    (3, 2, 2, False),
    (3, 1, 1, True),
    (3, 1, 1, False),
    (3, 1, 1, True),
    (3, 1, 1, False),
    (3, 1, 1, False),
    (3, 1, 1, True),
    (3, 1, 1, False),
    (3, 1, 1, True),
    (3, 1, 1, True),
    (3, 0, 1, True),
]


class Group:
    """Discrete rotation group C_N (order N) or dihedral group D_N (order 2N).

    Elements are (k, s) pairs meaning rho^k sigma^s: a rotation by 2*pi*k/N,
    optionally preceded by a reflection.
    """

    def __init__(self, N: int, reflections: bool = False):
        self.N = N
        self.reflections = reflections
        self.elements = [(k, 0) for k in range(N)]
        if reflections:
            self.elements += [(k, 1) for k in range(N)]
        self.order = len(self.elements)
        self._index = {g: i for i, g in enumerate(self.elements)}

    def mult(self, a, b):
        k1, s1 = a
        k2, s2 = b
        if s1 == 0:
            return ((k1 + k2) % self.N, s2)
        return ((k1 - k2) % self.N, (1 + s2) % 2)

    def inverse(self, a):
        k, s = a
        return a if s else ((-k) % self.N, 0)

    def index(self, element) -> int:
        return self._index[element]

    def angle(self, element) -> float:
        k, _ = element
        return 2 * math.pi * k / self.N


def transform_kernel(weight: torch.Tensor, group: Group, element) -> torch.Tensor:
    """Rotate/flip the last two (spatial) dims of `weight` by a group element.

    Reflection (horizontal flip) is applied before rotation, matching the
    (k, s) = rho^k sigma^s convention used by `Group`.
    """
    k, s = element
    *lead, kh, kw = weight.shape
    flat = weight.reshape(-1, 1, kh, kw)
    if s:
        flat = flat.flip(-1)
    if k and (kh > 1 or kw > 1):
        angle = group.angle(element)
        cos, sin = math.cos(angle), math.sin(angle)
        theta = flat.new_tensor([[cos, -sin, 0.0], [sin, cos, 0.0]])
        theta = theta.unsqueeze(0).expand(flat.shape[0], -1, -1)
        grid = F.affine_grid(theta, flat.shape, align_corners=True)
        flat = F.grid_sample(flat, grid, align_corners=True, padding_mode="zeros")
    return flat.reshape(*lead, kh, kw)


def transform_image(x: torch.Tensor, group: Group, element) -> torch.Tensor:
    """Apply a group element to a (B, C, H, W) image the same way
    `transform_kernel` transforms a filter, for use in equivariance checks."""
    return transform_kernel(x, group, element)


class GroupConv2d(nn.Module):
    """Equivariant convolution over a discrete group `group`.

    Lifts a trivial-representation input (`input_regular=False`, e.g. an RGB
    image) or transforms a regular-representation input (`input_regular=True`,
    `group.order` channels per input field, laid out as
    `channel = field_index * group.order + orientation_index`) into a
    regular-representation output with the same layout.
    """

    def __init__(
        self,
        group: Group,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        input_regular: bool = True,
    ):
        super().__init__()
        self.group = group
        self.stride = stride
        self.padding = padding
        in_mult = group.order if input_regular else 1

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, in_mult, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight.reshape(out_channels, -1, kernel_size, kernel_size), a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # perms[o][h] = index of the base in_mult channel that becomes the
        # h-th input-orientation channel of the filter for output element o.
        perms = []
        for g_o in group.elements:
            g_inv = group.inverse(g_o)
            if input_regular:
                perms.append([group.index(group.mult(g_inv, h)) for h in group.elements])
            else:
                perms.append([0])
        self.register_buffer("perms", torch.tensor(perms), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_c, in_c, in_mult, kh, kw = self.weight.shape
        batch = x.shape[0]
        kernels = []
        for o, g_o in enumerate(self.group.elements):
            w = self.weight[:, :, self.perms[o]]
            w = transform_kernel(w, self.group, g_o)
            kernels.append(w.reshape(out_c, in_c * in_mult, kh, kw))
        weight = torch.cat(kernels, dim=0)

        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding)
        h, w_ = out.shape[-2:]
        out = out.reshape(batch, self.group.order, out_c, h, w_).transpose(1, 2)
        out = out.reshape(batch, out_c * self.group.order, h, w_)
        if self.bias is not None:
            out = out + self.bias.repeat_interleave(self.group.order).view(1, -1, 1, 1)
        return out


class MaskModule(nn.Module):
    """Zeroes out pixels outside the inscribed circle, so a rotated image has
    no corners that a non-rotated one lacks."""

    def __init__(self, img_size: int, margin: float = 1.0):
        super().__init__()
        radius = img_size / 2 - margin
        coords = torch.arange(img_size, dtype=torch.float32) - (img_size - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        mask = (xx**2 + yy**2) <= radius**2
        self.register_buffer("mask", mask.float().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


class AntiAliasedAvgPool2d(nn.Module):
    """Isotropic Gaussian blur followed by strided average pooling, so
    downsampling does not alias the (approximately) equivariant features."""

    def __init__(self, channels: int, stride: int = 2, sigma: float = 0.66):
        super().__init__()
        self.channels = channels
        self.stride = stride
        radius = max(1, round(2 * sigma))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = (g[:, None] * g[None, :])
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.expand(channels, 1, *kernel.shape).clone(), persistent=False)
        self.padding = radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)
        return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)


class GroupPooling(nn.Module):
    """Invariant pooling: max over each field's orientation channels."""

    def __init__(self, group: Group, channels: int):
        super().__init__()
        self.group = group
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = x.reshape(b, self.channels, self.group.order, h, w)
        return x.max(dim=2).values


class ConvBnAct(nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size, padding, stride, input_regular, mask=None):
        super().__init__()
        self.mask = mask
        self.conv = GroupConv2d(
            group, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, input_regular=input_regular,
        )
        self.bn = nn.BatchNorm2d(out_channels * group.order)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            x = self.mask(x)
        return self.act(self.bn(self.conv(x)))


class GCNN(nn.Module):
    """GCNN (Pandya et al., 2023): group-equivariant galaxy morphology
    classifier, equivariant to the discrete rotation group C_N or, with
    `reflections=True`, the dihedral group D_N.

    https://arxiv.org/abs/2311.01500
    """

    def __init__(
        self,
        N: int = 8,
        reflections: bool = True,
        in_chans: int = 3,
        num_classes: int = 10,
        img_size: int = 255,
        feature_fields=DEFAULT_FEATURE_FIELDS,
    ):
        super().__init__()
        self.group = Group(N, reflections)
        mask = MaskModule(img_size)

        channels = [in_chans] + list(feature_fields)
        blocks = []
        pools = []
        for i, (k, pad, stride, pool_after) in enumerate(_BLOCK_CONFIG):
            blocks.append(
                ConvBnAct(
                    self.group, channels[i], channels[i + 1], k, pad, stride,
                    input_regular=(i > 0), mask=mask if i == 0 else None,
                )
            )
            pools.append(AntiAliasedAvgPool2d(channels[i + 1] * self.group.order) if pool_after else None)
        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList([p if p is not None else nn.Identity() for p in pools])

        self.gpool = GroupPooling(self.group, feature_fields[-1])
        # timm convention: num_classes=0 drops the head, forward() then returns pooled features.
        self.head = nn.Linear(feature_fields[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Regular-representation features before invariant group pooling."""
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.gpool(x)
        return self.head(x.flatten(1))


def _gcnn(N: int, reflections: bool, **kwargs) -> GCNN:
    kwargs.setdefault("N", N)
    kwargs.setdefault("reflections", reflections)
    return GCNN(**kwargs)


@register_model
def gcnn_c1(**kwargs) -> GCNN:
    return _gcnn(1, False, **kwargs)


@register_model
def gcnn_c2(**kwargs) -> GCNN:
    return _gcnn(2, False, **kwargs)


@register_model
def gcnn_c4(**kwargs) -> GCNN:
    return _gcnn(4, False, **kwargs)


@register_model
def gcnn_c8(**kwargs) -> GCNN:
    return _gcnn(8, False, **kwargs)


@register_model
def gcnn_c16(**kwargs) -> GCNN:
    return _gcnn(16, False, **kwargs)


@register_model
def gcnn_d1(**kwargs) -> GCNN:
    return _gcnn(1, True, **kwargs)


@register_model
def gcnn_d2(**kwargs) -> GCNN:
    return _gcnn(2, True, **kwargs)


@register_model
def gcnn_d4(**kwargs) -> GCNN:
    return _gcnn(4, True, **kwargs)


@register_model
def gcnn_d8(**kwargs) -> GCNN:
    return _gcnn(8, True, **kwargs)


@register_model
def gcnn_d16(**kwargs) -> GCNN:
    return _gcnn(16, True, **kwargs)
