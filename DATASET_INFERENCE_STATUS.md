# Dataset Inference Inventory and Run Plan

Date: 2026-03-18

This file documents the datasets currently prepared for segmentation inference, their manifest files, selected modality strategy, and output locations.

## 1) Active manifests (`raw_data/nnUNet_raw_data/infer_jobs`)

- `brats_men_t2f.json`  
  - Entries: **296**  
  - Source: `/home/jason/autorg/dataset/BraTS-MEN-Train`  
  - Modality used: `T2FLAIR` (from `*-t2f.nii`)

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
  - Case coverage: `sub-strokecase0001` to `sub-strokecase0250`  
  - Modality field used: `DWI`

- `mslesseg_flair.json`  
  - Entries: **115**  
  - Source: `/home/jason/autorg/dataset/MSLesSeg Dataset` (`train` + `test`)  
  - Modality policy: **single-modality FLAIR** (`T2FLAIR`) for pseudo-label consistency  
  - Includes all available train timepoints (`T1/T2/T3/T4` where present) and all test cases.

## 2) Selected pseudo-label strategy

- For MSLesSeg, pseudo labels (`ab` + `ana`) are generated from **FLAIR only**.
- Rationale: consistent single-modality pseudo labels across variable longitudinal timepoints.

## 3) Inference output targets (`/home/jason/autorg/inference_output`)

- `brats_men_run`
- `brats2020_val_run`
- `brats2021_train_run`
- `isles2022_dwi_run`
- `mslesseg_pseudolabel_flair`

## 4) Current snapshot (before/restart run)

- `brats_men_run`: `ab=182`, `ana=181`
- `brats2020_val_run`: `ab=0`, `ana=0`
- `brats2021_train_run`: `ab=0`, `ana=0`
- `isles2022_dwi_run`: `ab=0`, `ana=0`
- `mslesseg_pseudolabel_flair`: `ab=115`, `ana=115`

## 5) Runtime behavior notes

- Inference is configured with `--save_output_nii`, producing:
  - `*_ab_mask.nii.gz`
  - `*_ana_mask.nii.gz`
- Existing outputs are skipped by the predictor logic when both masks already exist.
- Checkpoint used for these runs:
  - `-model_folder /home/jason/autorg/checkpoints/SEG`
  - `-chk AutoRG_Brain_SEG`
