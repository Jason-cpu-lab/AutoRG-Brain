# MedSAM2 Integration (Archived Pointer)

This documentation has been consolidated into the canonical guide:

- `DOCS_COMBINED.md`

Use the combined guide for:

- MedSAM2 setup and environment
- Training and inference commands
- Checkpoint selection rules
- Runtime warning policy
- Preprocessing/troubleshooting notes

Recent implementation notes (2026-03-22):

- Added `AutoRG_Brain/generate_isles2022_dwi_masks.py` to batch-generate ISLES-2022 DWI `_ab` and `_ana` masks into:
	`/home/jason/autorg/inference_output/T1_priority/isles2022_t1_priority_run/DWI`.
- Updated `AutoRG_Brain/generate_seg_json.py` with `--isles-modals` (default: `DWI`) so ISLES-2022 entry selection aligns with current DWI-first inference outputs.

Recent implementation notes (2026-03-23):

- Hardened training data unpack/load path against intermittent worker crashes caused by truncated `.npy` cache files:
	- `AutoRG_Brain/dataset/dataset_loading_llm.py`: `convert_to_npy` now writes via atomic temp-file replace.
	- `AutoRG_Brain/dataset/dataset_loading.py`: added robust `.npy` load fallback that removes corrupted memmap files (`mmap length is greater than file size`) and transparently loads from `.npz`.
