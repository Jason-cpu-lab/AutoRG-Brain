# MedSAM2 Integration Guide (Beginner-Friendly)

This document explains:
1. What was changed in the codebase for MedSAM2 integration.
2. How to run training.
3. How to run inference.
4. Common errors and how to fix them.
5. March 2026 bug/warning fixes for `test_seg.py` inference.
6. March 2026 checkpoint selection fix (`model_best_*` vs `model_latest`) and direct use of `MedSAM2_latest.pt`.
7. What files are needed after Phase 1 / Phase 2 training.

The goal is to use a MedSAM2-based segmentation backbone **without changing AutoRG-Brain pipeline flow**.

---

## 0) March 2026 updates (important)

The inference/evaluation path was updated to fix warnings and a metric bug:

- ✅ Full ADC modality integration (share + dataloader paths):
  - `ADC` is now supported as a first-class modality alongside `DWI/T1WI/T2WI/T2FLAIR`
  - dataloaders now recognize `_adc` suffix and can sample ADC during training
  - share-network variants now include a dedicated ADC encoder branch
  - synthetic lesion intensity generation has explicit ADC handling

- ✅ AMP updates (partial):
  - core inference/base trainer paths use `torch.amp`
  - some task-specific trainers (including segmentation trainer paths) still import `torch.cuda.amp.autocast`
  - this is currently functional, but may still show deprecation warnings depending on PyTorch version
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

### D. Current modality naming in data loader
- File: `AutoRG_Brain/dataset/dataset_loading.py`
- Current key mapping used during training:
  - `DWI` bucket: suffix `_dwi`
  - `T1` bucket: suffix `_t1wi`, `_t1ce`, or `_t1`
  - `T2` bucket: suffix `_t2wi` or `_t2`
  - `FLAIR` bucket: suffix `_flair` or `_t2flair`
  - `ADC` bucket: suffix `_adc`
- Runtime modal names passed into augmentation/training are:
  - `DWI`, `T1WI`, `T2WI`, `T2FLAIR`, `ADC`
- Practical note: in this pipeline, plain `T1` is treated as the `T1WI` branch.

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

### Important runtime note (after training)
- Even after Phase 2 (unfrozen fine-tuning), keep these available at inference:
  1. AutoRG checkpoint (for example `model_final_checkpoint.model` + `.pkl`)
  2. MedSAM2 repo/config environment (`AUTORG_MEDSAM2_REPO`, `AUTORG_MEDSAM2_CONFIG`)
  3. A valid `AUTORG_MEDSAM2_CKPT` path (recommended for robust constructor behavior)

Why: current runtime first builds MedSAM2 architecture from repo/config/checkpoint path, then loads AutoRG checkpoint weights.

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

### Recommended two-phase training plan

Phase 1 (adapter warm-up, encoder frozen):
- `export AUTORG_MEDSAM2_FREEZE=1`
- train until loss/validation stabilizes.

Phase 2 (joint fine-tuning):
- `export AUTORG_MEDSAM2_FREEZE=0`
- continue training from latest checkpoint (do **not** use `--no_resume`).

Artifacts you should expect:
- After Phase 1:
  - AutoRG checkpoint A (`model_latest.model`/`model_final_checkpoint.model`) + original MedSAM2 checkpoint file.
- After Phase 2:
  - AutoRG checkpoint B containing fine-tuned adapter + fine-tuned MedSAM2 encoder weights.
  - Keep original MedSAM2 checkpoint path available for runtime model construction in current code path.

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
- For modality naming consistency in JSON files, prefer:
  - `DWI`, `T1WI`, `T2WI`, `T2FLAIR`, `ADC`
  - If your source dataset uses `T1`, map it to `T1WI` in JSON generation.

### Task001 JSON generation reference

For detailed Task001 segmentation manifest generation logic (matching rules, GT-first policy, inclusion criteria, and outputs), see:

- `/home/jason/autorg/AutoRG-Brain/SEG_JSON_GENERATION_DETAILS.md`

Recent update (2026-03-21): Task001 generation explicitly detects GT lesion masks for BraTS-MEN hyphenated naming (for example, `BraTS-MEN-00008-000-seg.nii`) and ISLES-2022 derivatives masks (for example, `derivatives/sub-strokecase0032/ses-0001/sub-strokecase0032_ses-0001_msk.nii`), while preserving GT-first then pseudo fallback behavior. Verify outcomes in `raw_data/nnUNet_raw_data/Task001_seg_test/generation_audit.json`.

Generator script location:

- `/home/jason/autorg/AutoRG-Brain/AutoRG_Brain/generate_seg_json.py`

Run from workspace root (`/home/jason/autorg/AutoRG-Brain`):

```bash
python AutoRG_Brain/generate_seg_json.py
```

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

### March 2026 training pipeline blockers (found during startup preflight)

1. **Split key mismatch with preprocessed cache**
  - Symptom: `KeyError` during `do_split` (missing case id in loaded dataset).
  - Cause: generated `test_file.json` referenced IDs not yet available under `preprocessed_data/.../nnUNetData_plans_v2.1_stage0`.
  - Workaround used for immediate startup: create filtered split JSON including only preprocessed IDs.

2. **`--use_compressed_data` ignored in trainer**
  - Symptom: unpack behavior triggered even when compressed mode requested.
  - Cause: trainer hardcoded `self.unpack_data = True`.
  - Fix applied: trainer now respects argument (`self.unpack_data = unpack_data`) in `nnUNetTrainerV2_six_pub_seg.py`.

3. **Unpack validation format bug (`validation` dict vs list)**
  - Symptom: unpack attempted to read `.../test.npz` and crashed with `FileNotFoundError`.
  - Cause: `dataset_loading_llm.unpack_dataset` assumed `train_file['validation']` is a list.
  - Fix applied: unpack now supports both list/dict validation structures and skips missing files safely.

These fixes were applied to make startup behavior consistent and robust before long training runs.

---

## 8) Current status

- MedSAM2 integration is implemented in code and validated.
- Training entrypoint has been run successfully with real MedSAM2 encoder (multiple epochs observed).
- Inference can be executed through the existing `test_seg.py` pipeline using the same trained model folder.
- Current inference/runtime still depends on MedSAM2 repo/config environment for architecture construction.
