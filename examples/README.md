# Examples

Optional, self-contained notebooks. Each installs its own extra dependencies
in its first cell, none of this is required to use `astrolens` itself.

## `linformer_training.ipynb`

Trains `Linformer` on
[`UniverseTBD/mmu_gz10`](https://huggingface.co/datasets/UniverseTBD/mmu_gz10)
(Galaxy10 DECals, 17,736 galaxies, 10 discrete classes), split 70/10/20
train/val/test. Points to the paper authors' repository for full-scale,
paper-accurate Galaxy Zoo 2 reproduction.

## `gcnn_training.ipynb`

Trains the group-equivariant `gcnn_d4` on the same `UniverseTBD/mmu_gz10`
split, following the reference implementation's `D8` training recipe
(AdamW, `MultiStepLR`, heavy rotation/flip augmentation) at a reduced batch
size and epoch count to fit one notebook session. Saves a checkpoint
(`gcnn_d4.pt`) used by `gcnn_analysis.ipynb`.

## `gcnn_analysis.ipynb`

Robustness and interpretability analysis for the trained GCNN checkpoint —
a one-pixel adversarial attack (differential evolution) and a t-SNE
projection of the model's invariant embedding, adapted from the reference
implementation's `onepixelattack.py` and `latent_space_analysis.py`. Requires
running `gcnn_training.ipynb` first.

## `lensiformer_training.ipynb`

Trains `Lensiformer` on the ML4SCI "Common Test I" strong-lensing dark-matter
substructure benchmark (no substructure / CDM / axion, 150x150
single-channel), subsampled to 3,000/750 images per class for a tutorial-scale
run. The reference dataset's own Google Drive links are dead; this loads a
verified re-upload from Hugging Face — see `utils/deeplense.py`. The
physics-informed encoder's `k_min`/`k_max`/`pixel_scale` follow the reference
implementation's HST-like arcsec/pixel scale and Einstein-radius-like
deflection bounds directly, and the notebook visualizes the model's own
lens-equation source reconstruction alongside the observed images.

## `astropt_pretraining.ipynb`

Self-supervised pretraining of `astropt` from scratch on
[`Smith42/galaxies`](https://huggingface.co/datasets/Smith42/galaxies)
(streamed, unlabeled — the same dataset the reference authors pretrain and
probe their own checkpoints on) with the autoregressive next-patch
objective. Saves a checkpoint, but mainly demonstrates the mechanism — see
`astropt_finetuning.ipynb` for a downstream task built on a real
pretrained model.

## `astropt_finetuning.ipynb`

Downloads a released pretrained checkpoint from
[`Smith42/astroPT`](https://huggingface.co/Smith42/astroPT), maps its weights
onto `astropt`'s causal transformer body (attention, MLP, layer norms,
position embeddings — see `load_pretrained_astropt`/`load_pretrained_backbone`
in `astrolens/pretrained/astropt.py`, re-exported from `utils/astropt.py`,
for the exact key mapping), then LoRA-finetunes a
classification head on `UniverseTBD/mmu_gz10`, evaluated on the same
70/10/20 split used by the other GZ10 notebooks.

## `astropt_similarity_search.ipynb`

Uses the same released checkpoint as a frozen feature extractor (no
finetuning) to embed a stream of `Smith42/galaxies`, then ranks by cosine
similarity to a query galaxy and inspects its nearest neighbors, adapted
from the reference model's
`scripts/euclid/downstream_tasks/similarity_search.py`.

## `astropt_anomaly_detection.ipynb`

Uses the same frozen-embedding approach as `astropt_similarity_search.ipynb`
to score every galaxy with Local Outlier Factor (cosine metric) and inspect
the highest-scoring outliers, cross-checked against `Smith42/galaxies`'
`merging_merger_fraction` column — the same property the reference model's
`scripts/euclid/downstream_tasks/anomaly_detection.py` cross-checks its own
anomalies against.

## `aion_embeddings.ipynb`

Loads the released `polymathic-ai/aion-base` checkpoint via
`astrolens.pretrained.aion.load_pretrained_aion` (a thin wrapper around the
`polymathic-aion` package — AION-1's 39-modality tokenizer stack is too
large to reimplement natively, see `astrolens/pretrained/aion.py`) and
extracts frozen image-only embeddings. Cutout coordinates come from
`UniverseTBD/mmu_gz10`, but the model input is real calibrated 4-band flux
fetched per-coordinate from `legacysurvey.org/viewer/cutout.fits` (the same
source AION's own tutorial uses), not gz10's RGB thumbnails. Ranks by
cosine similarity to a query galaxy and inspects nearest neighbors, same
layout as `astropt_similarity_search.ipynb`.

## `aion_lsdb_embeddings.ipynb`

Same AION-1 embedding + similarity search as `aion_embeddings.ipynb`, but
sources flux cutouts via [`lsdb`](https://github.com/astronomy-commons/lsdb)
(the SOTA framework for large partitioned/HATS astronomical catalogs)
against `hugging-science/mmu_legacysurvey_dr10_south_21` — 123M objects,
61 TiB, the raw-flux catalog family AION-1 was itself pretrained on. A cone
search resolves only the few partition files (tens of MB each) covering a
query region; see `utils/lsdb_legacysurvey.py` for two remote-read
workarounds this environment needed (a dask scheduler bug and a
partition-column read that has to happen locally after download rather
than through lsdb's own remote reader).

## `astroclip_similarity_search.ipynb`

Loads the released `polymathic-ai/astroclip` checkpoint via
`astrolens.pretrained.astroclip.load_pretrained_astroclip` (a thin wrapper
around the `astroclip` package — its 302M-parameter DINOv2 image encoder is
too large and dependency-heavy to reimplement natively, see
`astrolens/pretrained/astroclip.py`) and extracts frozen **image** embeddings
only. Cutout coordinates come from `UniverseTBD/mmu_gz10`, but the model
input is real 3-band (g,r,z) Legacy Survey flux fetched per-coordinate from
`legacysurvey.org/viewer/cutout.fits` and run through `decals_to_rgb` (the
exact preprocessing AstroCLIP's own README gives for cutouts outside its
premade dataset), not gz10's RGB thumbnails. AstroCLIP's own cross-modal
(image/spectrum) retrieval and other downstream tasks need a held-out
dataset that is a ~60 GB download; this notebook sidesteps that and instead
ranks by in-modal (image-to-image) cosine similarity to a query galaxy, same
layout as `astropt_similarity_search.ipynb` and `aion_embeddings.ipynb`.
