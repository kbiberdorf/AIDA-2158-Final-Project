# AIDA 2158A Final Project — Strawberry Harvesting Pipeline
**Student(s):** Mark, Kelsey, Herve

**Course:** Neural Networks and Deep Learning — Dr. M. Tufail, Red Deer Polytechnic

**Date:** April 2026

---

## What Was Built

A complete deep learning pipeline for robotic strawberry harvesting, built and validated in two versions:

```
Field photo → YOLOv11 detects all strawberries
            → Per-fruit ROI crop (asymmetric padding)
            → 3-class U-Net segments background / fruit / peduncle
            → Largest connected component filter
            → PCA extracts stem angle → normalised to [−90°, +90°]
            → Gripper alignment angle for robot arm
```

---

## Folder Guide

### `V1 Files/` — Original Pipeline (Baseline)

The first complete pipeline run. Each phase folder contains a `NOTES.md`, a notebook, and all generated output images.

| Folder | What it contains |
|--------|-----------------|
| `Phase1_Environment/` | Environment setup — RTX 5070, PyTorch + CUDA confirmed, `aida_stable` env |
| `Phase2_YOLOv11_Training/` | YOLOv11-seg trained on 2,800 images — mAP50 0.927 (box), 0.918 (mask) |
| `Phase3_ROI_Crops/` | Largest-fruit crops — 3,100 / 3,100 images processed |
| `Phase4_Peduncle_Annotation/` | 357 annotated pairs from 4 contributors (3.5× the 100-image minimum) |
| `Phase5_UNet_Training/` | Binary U-Net on full images — best val IoU 0.2291 at epoch 10/50 |
| `Phase6_Stem_Angle/` | PCA on binary mask predictions — 50/57 angles (87.7%), mean 59.2° |

**Key V1 outputs to note:**
- `Phase2_YOLOv11_Training/training_curves.png` — loss and mAP over 48 epochs
- `Phase2_YOLOv11_Training/val_predictions_sample.png` — predicted masks overlaid on val images
- `Phase5_UNet_Training/unet_training_curves.png` — shows overfitting after epoch 10
- `Phase6_Stem_Angle/v2_stem_angle_examples.png` — post-processed PCA overlays
- `Phase6_Stem_Angle/v2_stem_angle_distribution.png` — histogram + rose diagram

---

### `V2 Files/` — Upgraded Pipeline (Final)

V2 reframes the segmentation task as **three-class** (background / strawberry / peduncle) and trains on **per-fruit ROI crops** instead of full images. This gives the model structural context and eliminates the class-imbalance problem where peduncle pixels occupied less than 5% of each full frame.

#### Documents

| File | What it is |
|------|-----------|
| `FINAL_REPORT_V2.md` | **Read this first** — complete written report covering all modules with all V2 results and analysis |
| `V2_GUIDE.md` | Technical reference — every change from V1 to V2, why each was made, measured differentials, and the three-run weight tuning journey |
| `MODULE_V2_RETRAIN_JUSTIFICATION.md` | The case for retraining — problem statement, decision rationale |
| `PRESENTATION_UPDATE_GUIDE.md` | Slide-by-slide instructions for updating the existing PowerPoint deck with V2 numbers |

#### Notebooks

| File | What it runs |
|------|-------------|
| `module1_yolov11_training.ipynb` | YOLO training (unchanged from V1 — same weights used) |
| `module2_v2_multiclass_masks.ipynb` | Per-fruit ROI crops + 3-class pixel labels (BG/fruit/peduncle) |
| `module3_v2_unet_multiclass.ipynb` | 3-class U-Net training with class-weighted loss |

**PDF screenshots** of notebooks 2, 3, and 4 are in `V2 Files/` for reference without running code.

#### `Phase5_UNet_v2/`

Notes on the V2 U-Net — architecture, training runs, file locations, and slide update instructions.

#### `unet_v2/`

| File | What it is |
|------|-----------|
| `unet_v2_training_curves.png` | **Slide: show this** — train vs val loss and mIoU, best at epoch 22/27 |
| `unet_v2_predictions.png` | **Slide: show this** — qualitative predictions (original / GT / predicted / overlay) |
| `unet_v2_confusion_matrix.png` | **Slide: show this** — full 3×3 confusion matrix (BG / strawberry / peduncle) |
| `best_unet_v2.pt` | Trained model weights (tracked via Git LFS) |
| `confusion_val.csv` | Raw confusion counts |
| `val_split_v2.json` | Which crops were in the val set (reproducibility) |

#### `stem_angles_v3/`

| File | What it is |
|------|-----------|
| `stem_v3_examples.png` | **Slide: show this** — PCA axis overlays from V2 peduncle predictions |
| `stem_v3_dist.png` | **Slide: show this** — angle histogram after V2 (normalised to [−90°, +90°]) |
| `stem_angles_v3.json` | Per-crop angle data (all 192 val crops) |

---

## Run Order (if re-running from scratch)

1. `module1_yolov11_training.ipynb` — only if YOLO weights (`best.pt`) are missing
2. `module2_v2_multiclass_masks.ipynb` — regenerates all 3-class masks and per-fruit crops
3. `module3_v2_unet_multiclass.ipynb` — trains the 3-class U-Net
4. *(Module 4 notebook)* — runs PCA angle extraction on V2 predictions

If only the U-Net needs retraining, start at step 3. If only the angle extraction needs updating, start at step 4.

---

## Suggested Presentation Slide Order

1. **Title slide** — project name, team, course; add "V2: Multi-Class Segmentation on ROI Crops"
2. **Pipeline overview** — 5-stage flow diagram from `FINAL_REPORT_V2.md`
3. **Module 1: YOLO** — `training_curves.png` + `val_predictions_sample.png` (from `V1 Files/Phase2_YOLOv11_Training/`)
4. **Module 2: Annotation + V2 masks** — `annotation_sample.png` + `contributor_breakdown.png` (from `V1 Files/Phase4_Peduncle_Annotation/`)
5. **Module 3: V1 vs V2 U-Net** — show `unet_training_curves.png` (V1) alongside `unet_v2_training_curves.png` (V2); then `unet_v2_confusion_matrix.png`
6. **Module 3: Problems and fixes** — V1 identified 4 problems; V2 implemented all 4 fixes (see `PRESENTATION_UPDATE_GUIDE.md` Slide 15)
7. **Module 4: Stem angle** — `stem_v3_examples.png` → `stem_v3_dist.png`
8. **Results summary** — V1 vs V2 comparison table (see Key Numbers below)
9. **Limitations and future work** — peduncle precision (23.4%); larger annotated dataset; pretrained encoders

---

## Key Numbers

| Metric | V1 | V2 |
|--------|----|----|
| YOLOv11 mAP50 (box) | 0.927 | 0.927 (unchanged) |
| YOLOv11 mAP50 (mask) | 0.918 | 0.918 (unchanged) |
| Annotations | 357 pairs (4 contributors) | 357 pairs (4 contributors) |
| U-Net task | Binary (peduncle / not) | 3-class (BG / fruit / peduncle) |
| U-Net training input | Full 1008×756 images | Per-fruit 256×256 ROI crops |
| Best val metric | IoU 0.2291 (epoch 10/50) | mIoU 0.5843 (epoch 22/27) |
| Peduncle IoU | — | 0.195 |
| Peduncle recall | — | 54.0% (+20.5 ppts over unweighted) |
| Confusion matrix | 2×2 (near-empty) | 3×3 (interpretable) |
| Val set size | 57 images — 1 annotator | 192 crops — all 4 annotators |
| Valid angles produced | 50 / 57 (87.7%) | 178 / 192 (92.7%) |
| Angle range | 0°–180° (sign-ambiguous) | −43.85° to +81.89° (normalised) |
| Mean angle | 59.2° | 12.36° |
