# AutoRG-Brain Combined Documentation

Last updated: 2026-03-21

This is the canonical operational documentation for this repository (except README and Supplementary).

---

## 1) Current Operational Status

- Local preprocessing command is validated and completes:

```bash
cd /home/jason/autorg/AutoRG-Brain/AutoRG_Brain
python -m experiment_planning.nnUNet_plan_and_preprocess_llm -t 1 --verify_dataset_integrity
```

- Bucket preprocessing command remains valid only when Petrel config exists:

```bash
python -m experiment_planning_bucket.nnUNet_plan_and_preprocess_llm_bucket -t 1 --verify_dataset_integrity
```

- If `~/petreloss.conf` is missing, use local preprocessing.

---

## 2) MedSAM2 Integration (Single Source Summary)

### What is integrated

- `MedSAM2SegAdapter` is integrated as a segmentation backbone adapter.
- Training and inference support `--network_type medsam2`.
- Existing AutoRG pipeline remains intact around the adapter.

### Important runtime notes

- MedSAM2 path selection depends on checkpoint + config env variables.
- Checkpoint selection should be consistent with `network_type` in `.model.pkl`.
- AMP path uses `torch.amp.autocast`.

### Typical training entry

```bash
python train_seg.py 3d_fullres nnUNetTrainerV2 001 0 \
  --network_type medsam2 \
  --abnormal_type intense \
  -train_batch 5 \
  -val_batch 1 \
  -train nnUNet_raw_data/Task001_seg_test/test_file.json
```

---

## 3) Training/Preflight Fixes Applied

The following issue classes were fixed in code (with debugging details preserved):

### 3.1 Checkpoint resume compatibility (`_pickle.UnpicklingError`)

- Symptom:
  - Resume/load failed after PyTorch upgrade when reading `model_latest.model`.
- Root cause:
  - PyTorch 2.6+ changed default behavior for `torch.load` toward safer loading (`weights_only=True` default path).
- Fix:
  - Explicit trusted-local checkpoint load path with `weights_only=False` plus compatibility fallback.

### 3.2 Augmentation worker crash (`KeyError: 1`)

- Symptom:
  - Worker crash in augmentation pipeline (`MaskTransform`) with channel map access.
- Root cause:
  - Direct dict indexing for a channel key that may not exist.
- Fix:
  - Replace direct indexing with safe lookup (`.get(c, False)`).

### 3.3 Input-channel mismatch (`expected 1 channel, got 2`)

- Symptom:
  - Network forward mismatch between plan-expected channels and dataloader output.
- Root cause:
  - Loader passed all non-label channels where plan expected a single modality channel.
- Fix:
  - Enforce one-channel training tensor path and first-modality extraction logic.

### 3.4 Effective batch-size behavior / OOM startup risk

- Symptom:
  - Startup OOM risk due to effective runtime batch handling not matching intended plan behavior.
- Root cause:
  - Runtime batch behavior could diverge from selected stage plan.
- Fix:
  - Align trainer initialization to stage plan batch behavior.

### 3.5 Metric container initialization (`AttributeError`)

- Symptom:
  - Missing metric containers on some execution paths (`all_train_eval_metrics_ana` style failures).
- Root cause:
  - Containers not initialized unconditionally.
- Fix:
  - Initialize defaults first, then override for custom validation layouts.

### 3.6 Invalid Dice aggregation warning

- Symptom:
  - `invalid value encountered in scalar divide` in online evaluation.
- Root cause:
  - Dice division attempted with zero denominator.
- Fix:
  - Compute per-class Dice only with denominator > 0 and fallback to `0.0` when no valid class exists.

### 3.7 Excessive checkpoint writes during validation

- Symptom:
  - Repeated best-checkpoint writes even without metric improvement.
- Root cause:
  - Save path triggered unconditionally in patience/selection path.
- Fix:
  - Save best checkpoint only when metric improves.

### 3.8 Preprocessing crash on segmentation stack (`ValueError` in `np.vstack`)

- Symptom:
  - Preprocessing crashed around P9 cases with shape mismatch similar to:
  - `ValueError: ... along dimension 1, array at index 0 has size 352 and index 1 has size 73`
- Root cause:
  - `seg_file1` and `seg_file2` were stacked directly even when geometry differed.
- Fix:
  - In `preprocess/cropping_llm.py`, align both segmentation masks to the image reference grid (size/spacing/origin/direction) using nearest-neighbor SimpleITK resampling before stacking.

Core files touched:

- `AutoRG_Brain/network_training/network_trainer.py`
- `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`
- `AutoRG_Brain/augmentation/custom_transforms.py`
- `AutoRG_Brain/dataset/dataset_loading.py`
- `AutoRG_Brain/preprocess/cropping_llm.py`

---

## 4) Segmentation JSON Generation and Split Behavior

Generator script:

```bash
cd /home/jason/autorg/AutoRG-Brain/AutoRG_Brain
python generate_seg_json.py
```

Outputs:

- `raw_data/nnUNet_raw_data/Task001_seg_test/case_dic.json`
- `raw_data/nnUNet_raw_data/Task001_seg_test/dataset.json`
- `raw_data/nnUNet_raw_data/Task001_seg_test/test_file.json`

Operational reminder:

- If split IDs exceed current preprocessed cache, rebuild preprocessing or use a filtered split aligned to available `.npz` IDs.

---

## 5) ADC Modality Integration Summary

ADC is integrated as a first-class modality across:

- Dataloader modality parsing and routing
- Shared-network modality branches
- Synthetic intensity logic
- MedSAM2-compatible flow

Modality map includes: `DWI`, `T1WI`, `T2WI`, `T2FLAIR`, `ADC`.

---

## 6) Warning Policy (What to Ignore vs Fix)

Non-blocking warnings (informational):

- CUDA Python deprecation message for `cuda.cudart` (upstream binding warning)
- SciPy namespace deprecations from third-party libraries

Blocking errors (must fix):

- Python `Traceback` with exception exit
- Dataset/key mismatches causing `KeyError`
- Missing file/config errors in active execution mode

---

## 7) Detailed Debugging Log (Chronological)

### 7.1 Environment and dependency blockers observed

- Missing modules during startup/preprocessing (resolved by installation):
  - `SimpleITK`
  - `environs`
  - `coloredlogs`

### 7.2 Bucket-vs-local preprocessing path confusion

- Bucket preprocessing error (expected if no Petrel config):
  - `ConfigFileNotFoundError(/home/jason/petreloss.conf)`
- Operational decision:
  - Use local preprocessing module when running on local dataset storage.

### 7.3 Local preprocessing command that completed

```bash
cd /home/jason/autorg/AutoRG-Brain/AutoRG_Brain
python -m experiment_planning.nnUNet_plan_and_preprocess_llm -t 1 --verify_dataset_integrity
```

### 7.4 CUDA Python warning analysis

- Warning seen:
  - `<frozen importlib._bootstrap_external>: ... The cuda.cudart module is deprecated ...`
- Determination:
  - Non-blocking deprecation warning from upstream binding/import path.
- Mitigation applied:
  - Local environment import path updated to prefer `cuda.bindings.runtime` before legacy `cuda.cudart` in MONAI TRT import path.

### 7.5 Non-blocking warning examples kept for context

- SciPy namespace deprecations from third-party packages:
  - `scipy.ndimage.filters`
  - `scipy.ndimage.measurements`
  - `scipy.ndimage.interpolation`
- SWIG deprecation notices at import time.

### 7.6 Verified post-fix behavior

- Preprocessing now progresses through previously failing P9 cases.
- Local preprocessing command exits successfully in validated run context.
- Training path remains operational with known non-blocking warnings.

---

## 8) Working-Tree Change Log (Uncommitted, 2026-03-23)

This section records the **current local uncommitted changes** shown by `git status`.

### 8.1 Data loader hardening and synthesis robustness

#### `AutoRG_Brain/dataset/dataset_loading.py`

- Added atomic `.npy` creation in `convert_to_npy` (write temp file then `os.replace`) to reduce truncated cache artifacts.
- Added `_load_case_array(...)` helper:
  - Prefer `.npy` memmap load.
  - On memmap failure (including `mmap length is greater than file size`), warn, remove corrupted `.npy`, and fallback to `.npz`.
- Replaced direct `np.load(... .npy)` call sites in `determine_shapes` and `generate_train_batch` with `_load_case_array(...)`.
- Improved modality bucket parsing:
  - Parse modality from tokenized ID parts (supports explicit tags like `dwi/adc/flair/t2/t1`).
  - Keep legacy prefix fallback (`a/b/c/d/e`) for compatibility.
- Added `_warned_no_fg_cases` guard to avoid repeated spam for foreground-missing cases.
- Wrapped abnormal synthesis retries with warning filtering for noisy runtime warnings.
- Added retry context reporting after repeated synthesis failures.

#### `AutoRG_Brain/dataset/dataset_loading_llm.py`

- `convert_to_npy` now writes `.npy` atomically via temp file + replace.

#### `AutoRG_Brain/dataset/utils.py`

- Added `_minimal_fallback_lesion(...)`:
  - Creates a small valid fallback lesion mask when synthesis fails to produce enough lesion voxels.
  - Uses anatomy foreground preference with intensity blending by modality.
- Updated `SynthesisTumor(...)` behavior:
  - Retry loop now continues on exception (no hard raise path on first failure).
  - If total abnormal mask is too small, uses fallback lesion synthesis.
  - Ensures `xyzs` is always non-empty with a center fallback.

### 8.2 Dataset JSON generation logic changes

#### `AutoRG_Brain/generate_seg_json.py`

- Added helpers for stricter/cleaner stem handling:
  - `strip_ana_suffix(...)`
  - `extract_case_id(...)`
  - `expand_lookup_stems(...)` with modality alias expansion.
- `build_pseudo_maps(...)` now builds extra modality-specific and case-id-specific lookup maps:
  - `ab_by_modal`, `ana_by_modal`, `ana_exact_by_modal`, `ab_by_modal_case`, `ana_by_modal_case`.
- Label lookup logic tightened:
  - `label2` (abnormal) now prefers modal-specific maps and can fallback through modal case-id map.
  - `label1` (anatomy) pseudo fallback now enforces exact basename matching (`<image_stem>_ana_mask` / legacy `_ana`) via `ana_exact_by_modal`.
- Removed broad patient-prefix fallback in generic pseudo lookup to reduce false matches.

### 8.3 Trainer stability/config behavior

#### `AutoRG_Brain/network_training/network_trainer.py`

- Moving-average aggregation now ignores `None` entries:
  - Computes MA using only valid values for anatomy/abnormal channels.
  - Falls back to `0.0` if no valid metric entries are available.

#### `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`

- Added `Path` usage for robust project-relative file loading.
- Safe handling for optional `train_file` path (checks only when not `None`).
- `val_choose_number.json` path resolved relative to trainer file location (instead of fragile cwd-relative path).
- For `network_type in {medsam2, sam2}`:
  - Batch size now comes from env var `AUTORG_MEDSAM2_BATCH_SIZE` (default `1`).
  - Invalid env var values log warning and fallback to `1`.

### 8.4 Integration notes updates

#### `MEDSAM2_INTEGRATION.md`

- Added 2026-03-22 notes about:
  - New ISLES DWI mask generation script.
  - `generate_seg_json.py` ISLES modal selection update.
- Added 2026-03-23 notes about:
  - Atomic unpack writes.
  - Corrupted `.npy` fallback handling for dataloader memmap failures.

### 8.5 Generated/updated dataset artifacts

#### `raw_data/nnUNet_raw_data/Task001_seg_test/case_dic.json`

- Modality list counts changed:
  - `ADC`: `199 -> 250`
  - `DWI`: `199 -> 250`
  - `T2FLAIR`: `1860 -> 2156`
  - `T2WI`: `1885 -> 2181`
  - (`T1WI` list unchanged in this diff summary script output)

#### `raw_data/nnUNet_raw_data/Task001_seg_test/dataset.json`

- Dataset structure remains standard nnU-Net style.
- Current values include:
  - `numTraining: 6968`
  - `numTest: 0`
  - `training_len: 6968`
  - `test_len: 0`

#### `raw_data/nnUNet_raw_data/Task001_seg_test/generation_audit.json`

- `global_summary` changed:
  - `total_images_scanned: 11540 -> 12648`
  - `eligible_modal_images: 6968 -> 6968` (unchanged)
  - `included: 6376 -> 6968`
  - `label2_gt: 5998 -> 6593`
  - `label2_pseudo: 378 -> 375`
  - `label1_local: 0 -> 1107`
  - `label1_pseudo: 6376 -> 5861`
  - `excluded_missing_label1: 592 -> 0`
  - `excluded_missing_label2: 0 -> 0` (unchanged)
- `dataset_entries` notable change:
  - `BraTS-MEN-Train: 296 -> 888`
  - Other listed dataset counts unchanged in this comparison.

#### `raw_data/nnUNet_raw_data/Task001_seg_test/test_file.json`

- Structure changed from prior format (`training` + `numTraining`) to:
  - `training`: list of 6968 case-id strings
  - `validation`: object with `{"test": []}`
- `numTraining` key removed in current file.

#### `raw_data/nnUNet_raw_data/infer_jobs/isles2022_dwi.json`

- Still 250 entries, all with `modal: "DWI"`.
- Image paths normalized/standardized for all entries:
  - `image_path_changes_at_same_index: 250 / 250`
  - Standardized basename pattern count: `250 / 250`.

### 8.6 New untracked files

#### `AutoRG_Brain/generate_isles2022_dwi_masks.py`

- New utility script to generate ISLES-2022 DWI `_ab` and `_ana` masks via `inferenceSdk.SegModel`.
- Supports both direct DWI NIfTI files and directory-wrapped `*_dwi.nii/<inner>.nii` layouts.
- Writes outputs to configurable output dir and emits manifest:
  - `isles2022_dwi_masks_manifest.json`.

#### `raw_data/nnUNet_raw_data/Task001_seg_test/modal_gap_report.json`

- New summary report with:
  - Included modal counts (`T1WI/T2FLAIR/T2WI/ADC/DWI`)
  - Eligible counts by dataset
  - Missing-label exclusion counts (all zero in current snapshot)
  - Empty missing-example maps.

#### `train_medsam2.log`

- New captured training log artifact (`312` lines).
- Log includes epoch progression and crash trace showing historical root cause:
  - `ValueError: mmap length is greater than file size`
  - followed by worker failure wrapper:
    - `RuntimeError: One or more background workers are no longer alive...`

### 8.7 Diffstat snapshot for modified tracked files

- `12 files changed, 28178 insertions(+), 14055 deletions(-)`
- Largest diffs are data artifacts:
  - `generation_audit.json`
  - `dataset.json`
  - `test_file.json`
  - `case_dic.json`

---
