# Dataset Inference Inventory and Run Plan

Date: 2026-03-21

This file documents the datasets currently prepared for segmentation inference, manifest files, modality selection strategy, run targets, and current progress.

## 1) Active manifests (`raw_data/nnUNet_raw_data/infer_jobs`)

### A) Legacy manifests (kept for reference)

- `brats_men_t2f.json`
  - Entries: **296**
  - Source: `/home/jason/autorg/dataset/BraTS-MEN-Train`
  - Modality used: `T2FLAIR`

- `brats2020_val_flair.json`
  - Entries: **125**
  - Source: `/home/jason/autorg/dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData`
  - Modality used: `T2FLAIR`

- `brats2021_train_flair.json`
  - Entries: **1251**
  - Source: `/home/jason/autorg/dataset/BraTS2021_Training_Data`
  - Modality used: `T2FLAIR`

- `isles2022_dwi.json`
  - Entries: **250**
  - Source: `/home/jason/autorg/dataset/ISLES-2022/ISLES-2022`
  - Modality used: `DWI`

- `mslesseg_flair.json`
  - Entries: **115**
  - Source: `/home/jason/autorg/dataset/MSLesSeg Dataset` (`train` + `test`)
  - Modality used: `T2FLAIR`

### B) New T1-priority manifests (current run set)

Selection rule used per case:

1. `T1WI`
2. `T2FLAIR`
3. `T2WI`
4. `DWI`

- `brats_men_t1_priority.json`
  - Entries: **296**
  - Modality mix: `T1WI=296`

- `brats2020_val_t1_priority.json`
  - Entries: **125**
  - Modality mix: `T1WI=125`

- `brats2021_train_t1_priority.json`
  - Entries: **1251**
  - Modality mix: `T1WI=1251`

- `isles2022_t1_priority.json`
  - Entries: **250**
  - Modality mix: `T2FLAIR=250` (no T1 available in ISLES-2022)

- `mslesseg_t1_priority.json`
  - Entries: **115**
  - Modality mix: `T1WI=115`

## 2) Inference output targets (`/home/jason/autorg/inference_output`)

### New T1-priority output folders

- `brats_men_t1_priority_run`
- `brats2020_val_t1_priority_run`
- `brats2021_train_t1_priority_run`
- `isles2022_t1_priority_run`
- `mslesseg_t1_priority_run`

### Background run logs and PID files

- Log/PID folder: `/home/jason/autorg/AutoRG-Brain/inference_output/logs_t1_priority`
- One `*.log` + one `*.pid` per run.

## 3) Current live snapshot (during running jobs)

Snapshot captured: 2026-03-21

- `brats_men_t1_priority_run`: `ab=26`, `ana=25`
- `brats2020_val_t1_priority_run`: `ab=27`, `ana=27`
- `brats2021_train_t1_priority_run`: `ab=25`, `ana=25`
- `isles2022_t1_priority_run`: `ab=14`, `ana=12`
- `mslesseg_t1_priority_run`: `ab=24`, `ana=24`

Process snapshot:

- `active_test_seg_processes=45`
  - includes worker subprocesses spawned by `test_seg.py`

## 4) Runtime behavior notes

- Inference is configured with `--save_output_nii`, producing:
  - `*_ab_mask.nii.gz`
  - `*_ana_mask.nii.gz`
- Existing outputs are skipped when both masks already exist.
- Checkpoint used:
  - `-model_folder /home/jason/autorg/checkpoints/SEG`
  - `-chk AutoRG_Brain_SEG`
- Modal argument used for launched runs:
  - `T1WI` for BraTS-MEN / BraTS2020 / BraTS2021 / MSLesSeg
  - `T2FLAIR` for ISLES-2022 fallback run
