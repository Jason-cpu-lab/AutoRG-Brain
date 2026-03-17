# Training Fix Summary (Task001_seg_test)

## Current Status
- Training runs on CUDA and proceeds for multiple epochs.
- Loss decreases and Dice trends upward overall.
- The earlier invalid-Dice runtime warning is fixed.
- Last stop with `KeyboardInterrupt` happened during checkpoint save because training was manually interrupted (`Ctrl+C`).

## Total Code Changes
Yes. A total of **4 files** were updated.

## Fixes Applied

### 1) Checkpoint resume compatibility (`_pickle.UnpicklingError`)
- **Symptom**: loading `model_latest.model` failed after PyTorch upgrade.
- **Root cause**: PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`.
- **Fix**: load trusted local checkpoints with `weights_only=False` (with compatibility fallback).
- **File**: `AutoRG_Brain/network_training/network_trainer.py`

### 2) Augmentation worker crash (`KeyError: 1`)
- **Symptom**: `MaskTransform` worker crash.
- **Root cause**: direct dict indexing for channel flags when only channel `0` was configured.
- **Fix**: safe lookup via `.get(c, False)`.
- **File**: `AutoRG_Brain/augmentation/custom_transforms.py`

### 3) Input channel mismatch (`expected 1 channel, got 2`)
- **Symptom**: network forward mismatch between plans and dataloader output.
- **Root cause**: dataloader fed all non-label channels while plans expected one modality.
- **Fix**:
  - enforce one-channel data tensor shape
  - switch extraction to first modality channel
- **File**: `AutoRG_Brain/dataset/dataset_loading.py`

### 4) CUDA OOM due to effective batch-size override
- **Symptom**: out-of-memory at startup despite plan suggesting smaller batch size.
- **Root cause**: trainer runtime default effectively overrode plan batch size.
- **Fix**: align batch size to selected stage plan during initialization.
- **File**: `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`

### 5) Metric container initialization crash (`all_train_eval_metrics_ana` missing)
- **Symptom**: `AttributeError` during training/evaluation bookkeeping.
- **Root cause**: metric containers not always initialized on all execution paths.
- **Fix**: initialize default metric containers unconditionally, then override for custom validation layout.
- **File**: `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`

### 6) Invalid Dice warning (`invalid value encountered in scalar divide`)
- **Symptom**: warning in `finish_online_evaluation` during Dice aggregation.
- **Root cause**: Dice division attempted with zero denominator.
- **Fix**:
  - compute per-class Dice only when denominator `> 0`
  - use `0.0` fallback when no valid classes exist
- **File**: `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`

### 7) Excessive checkpoint saves during validation
- **Symptom**: multiple consecutive `saving checkpoint...` writes each validation epoch.
- **Root cause**: best-checkpoint save in `manage_patience_six_pub` executed even without improvement.
- **Fix**: save best checkpoint only when metric improves.
- **File**: `AutoRG_Brain/network_training/network_trainer.py`

## Files Updated
1. `AutoRG_Brain/augmentation/custom_transforms.py`
2. `AutoRG_Brain/dataset/dataset_loading.py`
3. `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`
4. `AutoRG_Brain/network_training/network_trainer.py`

## Runtime Notes
- These startup messages are warnings in your run, not blocking errors:
  - `nnUNet_raw_data_base is not defined`
  - `nnUNet_preprocessed is not defined`
  - `RESULTS_FOLDER is not defined`
- AMP deprecation warnings are non-fatal:
  - `torch.cuda.amp.autocast` → recommended `torch.amp.autocast('cuda', ...)`
  - `torch.cuda.amp.GradScaler` → recommended `torch.amp.GradScaler('cuda', ...)`

## Environment Note
To avoid local `libstdc++` mismatch in this environment, successful runs used:

```bash
LD_LIBRARY_PATH=/home/jason/miniforge3/envs/autorg/lib:$LD_LIBRARY_PATH \
/home/jason/miniforge3/envs/autorg/bin/python train_seg.py ...
```

## Last Verified Command Pattern

```bash
python train_seg.py 3d_fullres nnUNetTrainerV2 001 0 \
  --network_type share \
  --abnormal_type intense \
  -train_batch 5 \
  -val_batch 1 \
  -train nnUNet_raw_data/Task001_seg_test/test_file.json
```

## Conclusion
Training stability issues observed in this task have been addressed in code. Current remaining messages are mainly environmental/deprecation warnings, while core training now proceeds normally unless manually interrupted.
