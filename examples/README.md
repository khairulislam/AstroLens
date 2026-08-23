# Examples

Optional, self-contained notebooks. Each installs its own extra dependencies
in its first cell, none of this is required to use `astrolens` itself.

## `gz10_linformer_training.ipynb`

Trains `Linformer` on
[`UniverseTBD/mmu_gz10`](https://huggingface.co/datasets/UniverseTBD/mmu_gz10)
(Galaxy10 DECals, 17,736 galaxies, 10 discrete classes), split 70/10/20
train/val/test. Points to the paper authors' repository for full-scale,
paper-accurate Galaxy Zoo 2 reproduction.

## `gz10_gcnn_training.ipynb`

Trains the group-equivariant `gcnn_d4` on the same `UniverseTBD/mmu_gz10`
split, following the reference implementation's `D8` training recipe
(AdamW, `MultiStepLR`, heavy rotation/flip augmentation) at a reduced batch
size and epoch count to fit one notebook session. Saves a checkpoint
(`gcnn_d4.pt`) used by `gz10_gcnn_analysis.ipynb`.

## `gz10_gcnn_analysis.ipynb`

Robustness and interpretability analysis for the trained GCNN checkpoint —
a one-pixel adversarial attack (differential evolution) and a t-SNE
projection of the model's invariant embedding, adapted from the reference
implementation's `onepixelattack.py` and `latent_space_analysis.py`. Requires
running `gz10_gcnn_training.ipynb` first.

## `gz10_astropt_pretraining.ipynb`

Self-supervised pretraining of `astropt` from scratch on `UniverseTBD/mmu_gz10`
images with the autoregressive next-patch objective. Saves a checkpoint, but
mainly demonstrates the mechanism — see `gz10_astropt_finetuning.ipynb` for a
downstream task built on a real pretrained model.

## `gz10_astropt_finetuning.ipynb`

Downloads a released pretrained checkpoint from
[`Smith42/astroPT`](https://huggingface.co/Smith42/astroPT), maps its weights
onto `astropt`'s causal transformer body (attention, MLP, layer norms,
position embeddings — see `load_pretrained_backbone` in the notebook for the
exact key mapping), then LoRA-finetunes a classification head on
`UniverseTBD/mmu_gz10`, evaluated on the same 70/10/20 split used by the
other notebooks.
