# Paper: https://arxiv.org/abs/2304.05350
"""Astroformer (Dagli, 2023): a hybrid MBConv/transformer image classifier.

Astroformer belongs to the CoAtNet/MaxViT family of architectures, composed of
MBConv stages followed by relative-attention transformer stages. The paper's
headline architectural change is the stage layout, C-C-C-T rather than
CoAtNet's C-C-T-T (Sec. 4); its abstract also claims a different relative
self-attention formulation, but the authors' public reference implementation
(https://github.com/Rishit-dagli/Astroformer) builds that attention from
timm's stock `RelPosBias`/`RelPosMlp` classes unchanged, i.e. the same
mechanism `timm.models.maxxvit.MaxxVit` already uses. Since the reference
code's only real departure from stock MaxxVit is the stage-layout config, this
module reuses `MaxxVit` directly instead of duplicating roughly 1300 lines of
block code, and contributes only that config plus the astrolens-facing class
and factory.
"""

from typing import Tuple, Union

from timm.models.maxxvit import MaxxVit, MaxxVitCfg

from ..registry import register_model

# C-C-C-T stage layout, per the paper's ablation (Sec. 4). embed_dim/depths
# follow the "astroformer_0" size variant from the reference implementation
# (https://github.com/Rishit-dagli/Astroformer) — the paper itself reports a
# single model and does not define named size variants.
ASTROFORMER_CFG = MaxxVitCfg(
    embed_dim=(96, 192, 384, 768),
    depths=(2, 3, 5, 2),
    block_type=("C", "C", "C", "T"),
    stem_width=64,
    head_hidden_size=768,
)


class AstroFormer(MaxxVit):
    """Astroformer (Dagli, 2023): a hybrid MBConv/transformer classifier.

    https://arxiv.org/abs/2304.05350
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 256,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            cfg=ASTROFORMER_CFG,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )


@register_model
def astroformer(**kwargs) -> AstroFormer:
    return AstroFormer(**kwargs)
