# MHA-EOIR-Fusion

**Multi-Head Cross-Modal Attention for EO/IR Image Fusion — and a multi-seed, common-budget audit of what it actually contributes to small-UAV detection.**

This repository accompanies our study of a multi-head cross-modal attention fusion network (MHA) for paired electro-optical / infrared (EO/IR) imagery, evaluated **end-to-end on small-target UAV detection** rather than on fusion quality metrics. Beyond the model itself, the main contribution is an evaluation methodology that surfaced two failure modes which single-seed, frozen-threshold comparisons routinely miss:

1. **Outlier-seed mirages** — an apparent +7.4 pt recall gain shrank to +3.4 pt on a 5×3 seed matrix and *flipped sign* (−0.02) under leave-one-seed-out analysis.
2. **Operating-point illusions** — frozen-threshold "wins" can invert completely once methods are compared at a **common false-positive budget** (FROC). We demonstrate this in *both directions*.

## Headline results

Small-target recall (ground-truth boxes < 1024 px²) on a 707-frame EO/IR UAV test split, YOLOv8n detector:

| Method | Frozen-threshold R_small | Operating FP/1000 | **FROC@50 FP/1000** | Seeds |
|---|---|---|---|---|
| CDDFuse (pretrained) | 0.4643 ± 0.054 | 83.9 | **0.3216 ± 0.036** | 3 |
| Simple average | 0.4135 ± 0.036 | 66.9 | 0.3118 | 3 |
| **MHA h4 (ours, full)** | 0.4552 ± 0.035 | 83.8 | **0.3115 ± 0.040** | 9 |
| MHA h1 (single-head) | 0.3809 ± 0.149 | — | — | 9 |
| SwinFusion (pretrained) | 0.4682 ± 0.038 | 86.3 | 0.3109 ± 0.049 | 3 |
| h4 w/o global attention | **0.4829 ± 0.038** | **98.9** | 0.2766 ± 0.054 | 9 |
| h4 w/o DFU gating | **0.4803 ± 0.029** | **101.8** | 0.2734 ± 0.072 | 9 |
| h4 coarse-only decoder | 0.4236 ± 0.034 | 85.2 | 0.2581 ± 0.044 | 9 |

Read the middle block carefully — it is the point of the paper:

- **Ablations look *better* than the full model at their own frozen thresholds** (0.48 vs 0.455) but only because their operating points are ~15 FP/1000 looser. At a matched 50-FP budget the full model leads all three ablations by 3.5–5.3 pt (directionally consistent across 9 paired runs; paired t p ≈ 0.15–0.18, so we report it as a consistent reversal, not a proven difference).
- The **only ablation that is cleanly worse at a matched operating point** is the coarse-only decoder (same FP level, −3.2 pt, and AP50 collapses 0.575 → 0.524): fine-level skip connections contribute regardless of threshold placement. The attention/gating components contribute *ranking quality*, which only shows up under a common budget.
- **What multi-head attention does robustly deliver:** training stability. Across a 5 fusion-seed × 3 detection-seed matrix, h4's per-cell SD is 0.035 vs h1's 0.123 (3.5×), and h4 never produces a catastrophic fusion seed (worst row mean 0.426 vs h1's 0.218). The *mean* advantage (+3.4 pt) is fully attributable to h1's seed f3407 outlier under LOO.
- Off-the-shelf pretrained fusion baselines (CDDFuse, SwinFusion) match or slightly exceed the in-domain-trained model — at a common budget all four are statistically indistinguishable (~0.31–0.32).

Full tables, LOO analysis, paired tests, and the audit trail are in [`results/final_report.md`](results/final_report.md).

## Repository layout

```
model/mha_fusion.py          Reference PyTorch implementation (MHAFusionNet:
                             MultiHeadCrossModalAttention + DualFeatureUnit +
                             ModalityPairAttention + FusionLoss)
ablation/
  pkd_variants.py            Structural ablation variants (w/o global attention,
                             w/o DFU, coarse-only) as subclasses — no production
                             code modified
  train_variant.py           Trainer launcher (symbol-swap injection + --resume)
  export_variant.py          Fused-image export for a variant checkpoint
baselines/
  baseline_wrappers.py       Inference adapters for CDDFuse & SwinFusion
                             pretrained checkpoints (follows each repo's own
                             test-script conventions)
  export_baseline.py         Fused-image export for external baselines
evaluation/
  froc_at_budget.py          Common FP-budget FROC: re-runs full-confidence
                             predictions, sweeps thresholds, reports max
                             small-target recall at 10/25/50/100 FP per 1000
  matrix_5x3.py              5×3 seed-matrix aggregation + LOO sensitivity
  showcase_gen.py            Annotated qualitative comparison tiles (GT +
                             per-method detections + target zoom inset)
scheduling/gpu_pool_scheduler.sh
                             N-chain pool over 4 GPUs with skip/resume logic
                             (fusion -> export -> 3 detections x 3 evals)
results/final_report.md      Complete experimental report (tables + audit)
```

> The tooling under `ablation/`, `baselines/`, `evaluation/` and `scheduling/` is the exact code that produced the reported numbers (reproducibility by construction). Paths are anchored to the original experiment host (`/home/...`); adapt `PKG`/`RUNS`/`CODE` constants to your layout. `model/mha_fusion.py` is self-contained.

## Setup

```bash
conda create -n mha python=3.10 -c conda-forge
conda activate mha
# PyTorch (pick your CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.3.0 einops timm pillow pyyaml
```

Known pitfall (documented from experience): **Ultralytics 8.3.0 `select_device()` overwrites `CUDA_VISIBLE_DEVICES` with the literal string `cuda`**, silently collapsing every job onto physical GPU 0. The scripts here pin the device by pre-initializing a CUDA context (`_ = torch.zeros(1, device='cuda')`) *before* any Ultralytics call — keep that line if you reuse the code. Also cap `OMP_NUM_THREADS` when running several trainings per node.

## Data

Experiments use the public dataset *"Dataset of real and artificial spatial and temporally aligned EO/IR images of labelled UAVs"* (Mendeley Data, CC BY 4.0; DOI: see dataset page). We use only the **real** acquisition series (01–42; synthetic series 43–51 excluded) and re-partition at the **acquisition-series level** so every series belongs to exactly one subset (the original random 80/10/10 split leaks frames of the same sequence across subsets). Final split: 3,233 / 719 / 707 frames (train/val/test), pinned by a SHA-256 file manifest. Fusion models are trained on random 128×128 crops; detection runs on full frames.

## Evaluation protocol

1. **Frozen-threshold protocol** — pick the confidence threshold on the validation split at the ≤50 FP/1000 budget point (max small-target recall), apply it *unchanged* to the test split. Always report the *actual* test FP rate alongside.
2. **Common-budget FROC** — sweep all thresholds on the test split and report the max small-target recall achievable within each FP budget (10/25/50/100 per 1000 frames). This removes operating-point differences and is the only fair basis for cross-method claims.
3. **Multi-seed matrix** — every arm is run over fusion seeds × detection seeds; report SD and leave-one-seed-out sensitivity, not just means.

Training details (Adam 1e-4, 3-epoch warmup + cosine to 1e-6, batch 2, 100-epoch fixed budget, fp32 with deterministic kernels, final-epoch checkpoint with no validation-based selection; detector: YOLOv8n, SGD, 50 epochs, best-val-fitness checkpoint) are documented in the manuscript draft.

## Citation & acknowledgments

If you use this evaluation methodology or code, please cite the accompanying paper (reference to be added) and:

- The EO/IR UAV dataset: Mendeley Data, CC BY 4.0 (DOI on the dataset page).
- **The fusion backbone studied here follows the PKDFusion architecture** (project-internal lineage); `model/mha_fusion.py` is an independent reference re-implementation written for this study. Any derived use must keep this attribution.
- Baselines compared with their released pretrained weights: **CDDFuse** (Zhao et al., CVPR 2023) and **SwinFusion** (Ma et al.).
- Detector: **Ultralytics YOLOv8**.

## License

MIT for the code in this repository (see `LICENSE`), **except** that the architecture-attribution requirement above is a scientific-integrity obligation, not a legal one — keep it. Third-party assets (dataset, pretrained baselines) remain under their own licenses.
