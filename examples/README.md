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
in `utils/astropt.py` for the exact key mapping), then LoRA-finetunes a
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
