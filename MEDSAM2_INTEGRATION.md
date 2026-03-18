# MedSAM2 Integration Guide (Beginner-Friendly)

This document explains:
1. What was changed in the codebase for MedSAM2 integration.
2. How to run training.
3. How to run inference.
4. Common errors and how to fix them.
5. March 2026 bug/warning fixes for `test_seg.py` inference.
6. March 2026 checkpoint selection fix (`model_best_*` vs `model_latest`) and direct use of `MedSAM2_latest.pt`.

The goal is to use a MedSAM2-based segmentation backbone **without changing AutoRG-Brain pipeline flow**.

---

## 0) March 2026 updates (important)

The inference/evaluation path was updated to fix warnings and a metric bug:

- ✅ AMP deprecation fixes:
  - migrated from `torch.cuda.amp` to `torch.amp`
  - updated both trainer and inference autocast/gradscaler usage
- ✅ Old-checkpoint warning noise reduced:
  - `self.epoch != len(self.all_tr_losses)` warning is now training-only
  - inference no longer prints this warning repeatedly
- ✅ Metric return bug fixed:
  - `inference/predict.py` no longer always returns `0`
  - returns real average Dice when labels exist
- ✅ Clearer output for unlabeled test JSON:
  - if your test JSON has no `label` field, output is now:
    - `avg metric not computed (no ground-truth labels in test file)`
- ✅ Checkpoint-type mismatch guidance added:
  - `-chk` loads an AutoRG checkpoint (`*.model` + `*.model.pkl`) that defines `network_type`
  - if a folder contains mixed history (`share` and `medsam2` checkpoints), choose a checkpoint whose `.model.pkl` has `network_type='medsam2'`
  - in the current validated workspace, `model_latest` is `medsam2`, while `model_best_ab/model_best_ana/model_best_both` are `share`
- ✅ Direct MedSAM2 checkpoint usage validated:
  - `AUTORG_MEDSAM2_CKPT` can point directly to `checkpoints/MedSAM2_latest.pt`
  - adapter initialization confirms `encoder=medsam2`

If you still see `avg metric 0` from older logs, that was pre-fix behavior.

---

## 1) What was changed in code

### A. New MedSAM2 adapter network
- File: `AutoRG_Brain/network/medsam2_adapter.py`
- Added class: `MedSAM2SegAdapter`
- This class is a drop-in replacement for segmentation backbone and keeps AutoRG output contract:
  - output 1: anatomy logits (96 classes)
  - output 2: abnormal logits (2 classes)

#### Important internal behavior
- Uses real MedSAM2 API: `sam2.build_sam.build_sam2`
- If MedSAM2 cannot be loaded (missing config/checkpoint/dependency), it safely falls back to an internal 3D encoder.
- Uses slice-wise 2D bridging for 3D MRI volumes:
  1. reshape `[B, C, D, H, W]` to 2D slices
  2. project channels to MedSAM2-compatible 3-channel input
  3. run MedSAM2 image encoder
  4. rebuild 3D feature tensor
- Memory optimizations added:
  - slice chunking (`AUTORG_MEDSAM2_SLICE_CHUNK`)
  - optional lower input size (`AUTORG_MEDSAM2_INPUT_SIZE`)
  - math-only SDPA kernel mode for compatibility

### B. Trainer integration
- File: `AutoRG_Brain/network_training/nnUNetTrainerV2_six_pub_seg.py`
- Added `network_type` support:
  - `normal`
  - `share`
  - `medsam2`
- For `medsam2`:
  - initializes `MedSAM2SegAdapter`
  - uses `AdamW` optimizer
  - enforces effective `batch_size <= 1` to reduce OOM risk

### C. CLI update
- File: `AutoRG_Brain/train_seg.py`
- `--network_type` now supports `medsam2` in argument choices.

---

## 2) Environment setup (recommended)

> Run these commands from your workspace root: `/home/jason/autorg/AutoRG-Brain`

```bash
source /home/jason/miniforge3/etc/profile.d/conda.sh
conda activate autorg

# Clone MedSAM2 once (if not already cloned)
mkdir -p external
[ -d external/MedSAM2/.git ] || git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2

# Install MedSAM2 in editable mode
cd external/MedSAM2
SAM2_BUILD_CUDA=0 pip install -e .

# Keep AutoRG dependencies compatible (elasticdeform requires NumPy 1.x ABI here)
pip install --force-reinstall "numpy<2"
pip install --force-reinstall --no-deps elasticdeform
```

### Download checkpoint

```bash
cd /home/jason/autorg/AutoRG-Brain/external/MedSAM2
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Optional/recommended MedSAM2 task checkpoint
wget -O checkpoints/MedSAM2_latest.pt \
  https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt
```

---

## 3) Required MedSAM2 environment variables

Set these before training or inference:

```bash
export AUTORG_USE_MEDSAM2=1
export AUTORG_MEDSAM2_REPO=/home/jason/autorg/AutoRG-Brain/external/MedSAM2
export AUTORG_MEDSAM2_CONFIG=configs/sam2.1_hiera_t512.yaml
# Option A: SAM2 tiny backbone checkpoint
export AUTORG_MEDSAM2_CKPT=/home/jason/autorg/AutoRG-Brain/external/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt
# Option B (validated): MedSAM2 latest checkpoint
# export AUTORG_MEDSAM2_CKPT=/home/jason/autorg/AutoRG-Brain/external/MedSAM2/checkpoints/MedSAM2_latest.pt
export AUTORG_MEDSAM2_FREEZE=1

# Memory controls (recommended for your GPU usage)
export AUTORG_MEDSAM2_SLICE_CHUNK=4
export AUTORG_MEDSAM2_INPUT_SIZE=256
```

---

## 4) How to run training

> Run from: `/home/jason/autorg/AutoRG-Brain/AutoRG_Brain`

```bash
source /home/jason/miniforge3/etc/profile.d/conda.sh
conda activate autorg
cd /home/jason/autorg/AutoRG-Brain/AutoRG_Brain

python train_seg.py 3d_fullres nnUNetTrainerV2 001 0 \
  --network_type medsam2 \
  --abnormal_type intense \
  -train_batch 1 \
  -val_batch 1 \
  -train /home/jason/autorg/AutoRG-Brain/raw_data/nnUNet_raw_data/Task001_seg_test/test_file.json \
  --no_resume
```

### What you should see in logs
- `Initialized MedSAM2 adapter (encoder=medsam2)`
- Epochs start and continue (loss printed each epoch)

---

## 5) How to run inference

Inference uses `test_seg.py` with your trained model folder.

> Run from: `/home/jason/autorg/AutoRG-Brain/AutoRG_Brain`

```bash
source /home/jason/miniforge3/etc/profile.d/conda.sh
conda activate autorg
cd /home/jason/autorg/AutoRG-Brain/AutoRG_Brain

python test_seg.py \
  -o /home/jason/autorg/AutoRG-Brain/inference_output/medsam2_run \
  -model_folder /home/jason/autorg/AutoRG-Brain/trained_model_output/nnUNet/3d_fullres/Task001_seg_test/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0 \
  -chk model_latest \
  -test /home/jason/autorg/AutoRG-Brain/raw_data/nnUNet_raw_data/brats_custom_format_local_3.json \
  --modal T2FLAIR \
  --dice_type abnormal \
  --save_output_nii
```

### Notes
- `--modal` is required by `test_seg.py`.
- `--dice_type` can be `abnormal` or `anatomy` depending on what you evaluate.
- Keep MedSAM2 environment variables exported during inference as well.
- `test_seg.py` expects a **list-style** JSON (`[{"image": ..., "modal": ...}, ...]`).
- To compute Dice/NSD/HD metrics, each item must also include `"label"`.
- If `"label"` is missing, segmentation still runs and NIfTI outputs are saved, but aggregate metric is intentionally not computed.
- `/home/jason/autorg/AutoRG-Brain/raw_data/nnUNet_raw_data/Task001_seg_test/test_file.json` is for training split metadata and is **not** valid for `test_seg.py`.
- `-chk` selects an AutoRG checkpoint file. Always ensure that checkpoint's `.model.pkl` has `network_type='medsam2'`.
- In this validated workspace: use `-chk model_latest` for MedSAM2 inference (`model_best_*` currently map to `share`).

---

## 6) Quick sanity check (without full training)

```bash
source /home/jason/miniforge3/etc/profile.d/conda.sh
conda activate autorg
cd /home/jason/autorg/AutoRG-Brain/AutoRG_Brain

CUDA_VISIBLE_DEVICES='' python - <<'PY'
import torch
from network.medsam2_adapter import MedSAM2SegAdapter

model = MedSAM2SegAdapter(in_channels=1, num_classes_anatomy=96, num_classes_abnormal=2,
                          use_medsam2_encoder=True, deep_supervision=False)
model.eval()
x = torch.randn(1, 1, 1, 32, 32)
with torch.no_grad():
    a, b = model(x)
print('encoder_name:', model.encoder_name)
print('anatomy:', tuple(a.shape), 'abnormal:', tuple(b.shape))
PY
```

Expected: `encoder_name: medsam2`.

---

## 7) Troubleshooting

### Error: `ImportError: numpy.core.multiarray failed to import`
Cause: NumPy ABI mismatch with compiled dependencies.

Fix:
```bash
conda activate autorg
pip install --force-reinstall "numpy<2"
pip install --force-reinstall --no-deps elasticdeform
```

### Error: CUDA OOM at first epoch
Fix options:
1. `export AUTORG_MEDSAM2_SLICE_CHUNK=2` (smaller chunk)
2. `export AUTORG_MEDSAM2_INPUT_SIZE=256` or `192`
3. keep `AUTORG_MEDSAM2_FREEZE=1`

### Error: fallback encoder is used (`encoder_name: fallback_3d`)
Check:
1. `AUTORG_USE_MEDSAM2=1`
2. `AUTORG_MEDSAM2_REPO` path exists
3. `AUTORG_MEDSAM2_CONFIG=configs/sam2.1_hiera_t512.yaml`
4. `AUTORG_MEDSAM2_CKPT` points to real `.pt` file

### Observation: model loads but MedSAM2 is not used
Cause: `-chk` points to a checkpoint whose `.model.pkl` encodes `network_type='share'` (or `normal`).

Quick check:
```bash
python - <<'PY'
from batchgenerators.utilities.file_and_folder_operations import load_pickle
p='/home/jason/autorg/AutoRG-Brain/trained_model_output/nnUNet/3d_fullres/Task001_seg_test/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_latest.model.pkl'
init = load_pickle(p)['init']
print('network_type:', init[14])
PY
```

Expected for MedSAM2 inference: `network_type: medsam2`.

### Observation: `avg metric not computed (no ground-truth labels in test file)`
This is expected for inference-only JSON files.

To enable metric computation, add `"label"` paths in your `-test` JSON entries:

```json
[
  {
    "image": "/path/to/BraTS20_Training_001_flair.nii.gz",
    "label": "/path/to/BraTS20_Training_001_seg.nii.gz",
    "modal": "T2FLAIR"
  }
]
```

---

## 8) Current status

- MedSAM2 integration is implemented in code and validated.
- Training entrypoint has been run successfully with real MedSAM2 encoder (multiple epochs observed).
- Inference can be executed through the existing `test_seg.py` pipeline using the same trained model folder.
