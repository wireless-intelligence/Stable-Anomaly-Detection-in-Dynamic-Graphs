# Vendor Code Usage Status

This document explains, for each baseline, **whether the experiment runs the
vendored official code or our PyTorch reimplementation**, and why.

## Summary

| Baseline      | Path used in `src/baselines/` | Vendor status                                  |
|---------------|--------------------------------|------------------------------------------------|
| DOMINANT      | `pygod` (official wrapper)     | covered by pygod                               |
| CoLA          | `pygod`                        | covered by pygod                               |
| CONAD         | `pygod`                        | covered by pygod                               |
| AnomalyDAE    | `pygod`                        | covered by pygod                               |
| GUIDE         | `pygod`                        | covered by pygod                               |
| ConsisGAD     | reimpl                         | vendor needs DGL + is semi-supervised (mismatch with our unsupervised eval) |
| ComGA         | reimpl                         | vendor is TensorFlow 1.x (incompatible with our PyTorch pipeline) |
| ANEMONE       | reimpl                         | vendor needs DGL; uses bespoke .mat format     |
| PMP           | reimpl                         | vendor is semi-supervised (needs train/val labels) |
| GAAP          | reimpl                         | vendor is small but uses bespoke checkpoint format |
| Graphormer    | reimpl                         | vendor is fairseq-based (huge framework dep)   |
| GraphGPT      | reimpl (graph-projector proxy) | vendor needs the *fine-tuned* GraphLlama weights, not just base llama-2-7b; per-node LLM inference at N=325k is intractable (~weeks on a 5090) |
| HiGPT         | reimpl (hierarchical proxy)    | same as GraphGPT — needs fine-tuned LLM head   |
| TAPE          | reimpl (text-prototype proxy)  | vendor needs preprocessed GPT-3.5 explanations per node |
| InstructGLM   | reimpl (instruction-head proxy)| vendor needs LoRA-tuned LLM weights per dataset |
| DySAT, NetWalk, MIDAS, sedanspot, pyHGT, GraphSAGE | reimpl | vendor is C++/MATLAB/TF1 or otherwise non-importable |

## What "reimpl" means here

For each baseline, `src/baselines/<name>.py` contains a faithful PyTorch
reimplementation of the **algorithm** described in the paper, not a wrapper
around the vendor binary. Reimpls follow the official architecture
(GCN encoder, scoring head, loss terms) but adapt:

  - **Sparse adjacency** — original code often assumes dense N x N
    reconstruction, which OOMs at N > 30k. We swap in negative-sampled
    BCE for the structural-reconstruction term (`_oom_guard.py`).
  - **Subgraph sampling** — Graphormer's full N x N attention is
    intractable for N > ~30k; we mini-batch via anchor k-hop balls
    (the same approach the original paper used for OGB-LSC).
  - **Unsupervised setup** — semi-supervised methods (ConsisGAD, PMP)
    are scored against the same unsupervised evaluation protocol as
    every other baseline; the labelled pre-training step from the
    vendor is omitted.

## What about the LLM-graph baselines?

GraphGPT, HiGPT, TAPE, and InstructGLM are LLM-graph methods. Faithful
end-to-end reproduction would require:

  1. The dataset-specific instruction-tuned LLM checkpoint (NOT publicly
     released for most datasets).
  2. Per-node prompt construction + LLM inference (~10–60 s per node).
  3. A pre-trained graph-projector aligning the GCN embedding space to
     the LLM token space.

For our largest graph (CTU-13, N = 325k) this is intractable
(~325k * 30 s ~= 113 days). Even on the cloud RTX 5090 it's many days
per dataset.

Following community practice (e.g., the GADBench LLM appendix), we
implement each of these methods as a **graph-only proxy** that uses the
algorithm's GCN-encoder + scoring head described in the paper, **without**
the LLM stage. This gives a fair comparison of the graph-side
architecture in isolation. The proxy nature is documented in each
file's docstring.

## What is in `vendor/`?

The directory contains the cloned official repos and downloaded LLM
checkpoints (llama-2-7b-hf, Qwen3-8B). They are kept so that:

  - Users can swap in the official LLM mode.
  - Any architectural detail in the reimpl can be cross-checked against
    the official source.

The vendor directory is **excluded from the SAG_cloud.tar.gz upload
bundle** to keep the upload small (~1.6 GB without it; ~50 GB with it,
of which ~42 GB is the llama-2-7b-hf and Qwen3-8B checkpoints).
On the cloud GPU instance, vendor files are cloned/downloaded again only
if they're actually needed for an experiment. See
`setup_vendor_baselines.sh` for the clone scripts.
