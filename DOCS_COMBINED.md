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
