# Forest-Cover Model — Production Bug Investigation & Labeled Validation

**Date:** 2026-07-27
**Scope:** Why the carbon-analysis pipeline produced meaningless forest masks, the
root cause, the fix applied, and a real ground-truth accuracy check.

## TL;DR

The forest-cover U-Net was emitting a near-constant ~0.5 field for **every** image in
production — forest coverage stuck at ~51% regardless of the actual scene, so every
carbon number was disconnected from reality. Root cause was two **inference-preprocessing**
mismatches (input scale + band order), **not** the model weights. The scale bug is fixed
here. With correct input the model discriminates, but a Hansen-labeled check on an
out-of-training Amazon region shows **weak generalization (ROC-AUC 0.59)** — the reported
F1 = 0.49 was in-distribution only. The model needs retraining/fine-tuning + multi-region
validation before its carbon numbers can be trusted.

## Symptom

`process_single_image` / `POST /api/v1/ml/forest-cover` reported ~51% forest coverage for
any input. Synthetic noise, real dense forest, all-zeros, and random tensors all produced
**byte-identical** output: `mean_prob=0.502, min=0.439, max=0.586`.

## Root cause — two inference-preprocessing bugs

1. **Input scale.** The U-Net's first `BatchNorm2d` running stats were accumulated on
   **raw reflectance DN** (`running_mean≈2086`, `running_var≈2.6M`). The inference loader
   normalized input to `[0,1]` (`clip(DN,0,10000)/10000`). In eval mode the first BN then
   computes `(0.5 − 2086)/1600 ≈ −1.28` for every pixel → constant output for any image.
   - Evidence: eval-mode input response `|Δ(rand,zeros)| = 0.00002` (dead) vs **train-mode
     0.18** (responds); byte-identical output across 7 contrasting inputs; weights healthy
     (0% near-zero, normal magnitude). It is the normalization, not the weights.

2. **Band identity/order.** The model was trained on **12 bands in order**
   `[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]`
   (`ml/scripts/extract_sentinel2_patches.py`). The pipeline fed whatever bands the uploaded
   GeoTIFF contained (commonly 4: `B02,B03,B04,B08`), truncated/zero-padded to 12 — so every
   channel was misaligned and 8 were zero.

## Fix applied (this change)

- `ml/inference/production_inference.py` — `_tiled_forest_prediction` now calls
  `load_full_image(..., normalize=False)` → feeds **raw DN**; the model's internal BatchNorm
  does the scaling. Documented the raw-DN + 12-band-order input contract.
- `ml/utils/data_preprocessing.py` — documented that `normalize=True` flatlines the forest
  U-Net (raw DN required).
- The **band-order contract** must be satisfied at the source: the imagery-acquisition step
  must stack the 12 bands in training order as raw DN. (Not yet wired — see follow-on #3.)

## Labeled validation (real ground truth)

- **Scene:** Sentinel-2 L2A over Manaus, Amazon (`S2B_20MRB_20230914`, 0.3% cloud),
  12 bands raw DN, 512×512 @ 10 m.
- **Ground truth:** Hansen GFC-2023 v1.11 `treecover2000 ≥ 30% & lossyear == 0` → **64% forest**.

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.53 (checkpoint-tuned) | **0.995** | 0.116 | 0.207 |
| 0.25 | 0.68 | 0.65 | 0.67 |
| 0.10 | 0.64 | 1.00 | 0.78 (degenerate — predicts all-forest) |

**ROC-AUC = 0.593** (0.5 = random, 1.0 = perfect).

Interpretation: the fix takes the model from AUC **0.500** (dead constant, useless) to
**0.593** (weakly discriminating). Its confident, high-threshold predictions are reliable
(precision → 0.99) but recall is low → it under-detects forest (a conservative underestimate
of carbon). It does **not** generalize well to regions outside its training distribution; the
repo's F1 = 0.4911 was measured on the training-region test set only.

**Caveats:** Hansen is 30 m / year-2000 resampled to 10 m with a 23-year temporal gap →
label noise; the raw-DN reconstruction may not match training's exact resampling grid. AUC
is possibly modestly pessimistic — but not enough to reach "good."

## Follow-on work (not done here)

1. **Retrain/fine-tune** the forest U-Net on the target regions with a proper Hansen-labeled
   train/val/test split; report AUC/F1 per region. This is the real bottleneck for
   trustworthy carbon numbers.
2. **Per-region threshold calibration** — the fixed 0.53 misses most forest on new regions.
3. **Wire the band-order contract at the source** — the imagery-acquisition step must produce
   the 12-band training-order raw-DN stack (a working prototype exists in the investigation
   scratch: STAC search → windowed 12-band read → stack).
4. **Frame carbon numbers as experimental** with mandatory human review until accuracy is
   defensible (the app already avoids fabricated AI confidence).
5. **(Secondary)** The change-detection model uses a different, possibly-unwired
   normalization (`transforms.Normalize`, ImageNet-style, `train_change_detection.py`);
   verify before relying on the change-% stat. It does **not** affect the carbon number
   (that comes from the forest-cover diff).
