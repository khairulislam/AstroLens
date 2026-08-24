"""Optional loader for a released AstroCLIP checkpoint (Polymathic AI), via
the `astroclip` package.

Source: https://huggingface.co/polymathic-ai/astroclip (Parker et al. 2024,
https://doi.org/10.1093/mnras/stae1450), license MIT (Polymathic AI).
AstroCLIP CLIP-aligns a 302M-parameter DINOv2 image encoder (astrodino) with
a 43M-parameter masked-modeling spectrum transformer (specformer) via
cross-attention projection heads — too large and dependency-heavy (a pinned
`facebookresearch/dinov2` fork, PyTorch Lightning) to reimplement as a
single native module the way this library's other models are (compare
`astrolens/models/astropt.py`). This is instead a thin wrapper around the
`astroclip` package, imported lazily so it is not a core runtime
dependency. Install it per the AstroCLIP README
(https://github.com/PolymathicAI/AstroCLIP#installation) — its dinov2 and
astroclip installs both require `--no-deps` git installs, so it cannot be
expressed as a single `pip install astrolens[...]` extra.
"""

REPO_ID = "polymathic-ai/astroclip"
CKPT_FILENAME = "astroclip.ckpt"


def load_pretrained_astroclip(device: str = "cpu"):
    """Load the released AstroCLIP checkpoint.

    Returns an `astroclip.models.AstroClipModel` instance. Call
    `model(image, input_type="image")` or `model(spectrum, input_type="spectrum")`
    to get the aligned embedding for each modality. See
    https://github.com/PolymathicAI/AstroCLIP for image preprocessing
    (144x144 center crop, `decals_to_rgb`) and spectrum preprocessing.
    """
    from astroclip.models import AstroClipModel
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILENAME)
    return AstroClipModel.load_from_checkpoint(checkpoint_path=path).to(device).eval()
