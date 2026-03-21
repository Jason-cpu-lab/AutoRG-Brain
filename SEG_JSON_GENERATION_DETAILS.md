# Task001 Segmentation JSON Generation Details

This document explains how `AutoRG_Brain/generate_seg_json.py` builds the training JSON files for segmentation.

## Purpose

The script generates three files under:

- `raw_data/nnUNet_raw_data/Task001_seg_test/case_dic.json`
- `raw_data/nnUNet_raw_data/Task001_seg_test/dataset.json`
- `raw_data/nnUNet_raw_data/Task001_seg_test/test_file.json`

It is designed to avoid manual editing of very large JSON files.

## Input Sources

### Dataset root (images)

By default, the script scans all NIfTI files under:

- `/home/jason/autorg/dataset`

### Pseudo-label roots (fallback labels)

By default, the script scans pseudo-label outputs under:

- `/home/jason/autorg/inference_output`

## Output Format (README-compatible)

### `dataset.json`

Top-level fields:

- `description`
- `labels`
- `modality`
- `name`
- `numTest`
- `numTraining`
- `reference`
- `release`
- `tensorImageSize`
- `test`
- `training`

Each `training` entry has:

- `image`: path to image NIfTI
- `label1`: anatomy mask path
- `label2`: anomaly/abnormal mask path
- `modal`: one of `DWI`, `T1WI`, `T2WI`, `T2FLAIR`, `ADC`

### `case_dic.json`

Dictionary with modality keys:

- `DWI`
- `T1WI`
- `T2WI`
- `T2FLAIR`
- `ADC`

Each value is a case-id list (image stem without `.nii`/`.nii.gz`) filtered from final included training samples.

### `test_file.json`

Contains:

- `training`: unique list of included case IDs
- `validation`: `{ "test": [] }`

## Core Matching Logic

## 1) Candidate image selection

The script scans all `.nii` / `.nii.gz` files in `--data-root` and excludes files that already look like masks:

- names containing `_seg`, `_mask`, `_ab_mask`, `_ana_mask`
- names ending with `_ab` or `_ana`

## 2) Modality detection

Modality is inferred from filename tokens:

- `ADC` if `adc`
- `DWI` if token `dwi`
- `T2FLAIR` if `flair` or token `t2f`
- `T2WI` if token `t2` or `t2w`
- `T1WI` if token `t1` or `t1n`

Important guard:

- `t1ce` / `t1c` are **not** mapped to `T1WI`.

## 3) `label2` (abnormal) policy: GT first, pseudo fallback

For each candidate image, the script first searches for local GT near the image path.

Common local GT patterns include:

- BraTS-like: `<case>_seg.nii` / `<case>_seg.nii.gz`
- BraTS-MEN-like: `<case>-seg.nii` / `<case>-seg.nii.gz`
- Mask-like: `<case>_MASK.nii.gz` / `<case>_mask.nii.gz`
- MSLesSeg-like: `Pxx_MASK.nii.gz`

Dataset-specific GT support:

- **BraTS-MEN-Train**:
	- Example image: `.../BraTS-MEN-00008-000-t1n.nii`
	- GT lesion mask: `.../BraTS-MEN-00008-000-seg.nii`
- **ISLES-2022**:
	- Example image: `.../sub-strokecase0032_ses-0001_FLAIR.nii`
	- GT lesion mask (derivatives):
		- `.../ISLES-2022/ISLES-2022/derivatives/sub-strokecase0032/ses-0001/sub-strokecase0032_ses-0001_msk.nii`

If no local GT is found, it falls back to pseudo abnormal masks found in pseudo roots:

- `*_ab_mask.nii.gz`
- `*_ab.nii.gz` (or `.nii`)

## 4) `label1` (anatomy) policy

The script first checks local anatomy-style files:

- `<image_stem>_ana_mask.nii.gz`
- `<image_stem>_ana.nii.gz`

If missing, it falls back to pseudo anatomy masks:

- `*_ana_mask.nii.gz`
- `*_ana.nii.gz` (or `.nii`)

## 5) Inclusion rule

A sample is included only if **both** are found:

- `label1`
- `label2`

This is required by preprocessing/training pipeline expectations.

## 6) Deduplication and priority

Entries are deduplicated by `(image, modal)`.

If duplicates exist, entries with GT-like `label2` (for example paths containing `_seg` or dataset mask naming) are preferred over pseudo-label entries.

## 7) Final file writing

After filtering and deduplication, the script writes:

- `dataset.json` with full training entries and counts
- `case_dic.json` rebuilt from final entries
- `test_file.json` with final unique case IDs

## CLI Usage

Run with defaults:

- `python AutoRG_Brain/generate_seg_json.py`

Optional arguments:

- `--data-root <path>`
- `--pseudo-root <path>` (can be passed multiple times)
- `--output-dir <path>`

## Validation Checklist

After generation, verify:

- `dataset.json["numTraining"] == len(dataset.json["training"])`
- all `image`, `label1`, `label2` paths exist
- `case_dic.json` counts match training entries by modality
- `test_file.json["training"]` matches unique case IDs from `dataset.json`
- GT-first policy holds when local GT exists

## Notes

- Because file naming can vary by dataset, matching uses both exact and relaxed stem matching strategies.
- If pseudo-label folders are updated, rerun the script to refresh JSON outputs.

## Training startup issues found (2026-03-21) and fixes

During preflight/startup testing with `train_seg.py`, the following issues were found and resolved.

### Issue 1: split IDs did not match currently preprocessed cache

- Symptom:
	- `KeyError: 'BraTS-MEN-00008-000-t1n'` in trainer split step.
- Root cause:
	- Newly generated `test_file.json` referenced many IDs not present in current preprocessed stage cache.
	- At the time of check, `test_file.json` had 6819 training IDs, while `preprocessed_data/Task001_seg_test/nnUNetData_plans_v2.1_stage0` contained 368 `.npz` cases.
- Practical workaround used for startup validation:
	- Created filtered split files containing only available preprocessed IDs:
		- `test_file_preprocessed_only.json`
		- `test_file_preprocessed_split.json` (adds non-empty validation subset)

### Issue 2: `--use_compressed_data` was ignored

- Symptom:
	- Trainer still attempted unpack path even with `--use_compressed_data`.
- Root cause:
	- `nnUNetTrainerV2_six_pub_seg` overwrote `self.unpack_data` to `True`.
- Fix applied:
	- Updated trainer init to respect passed argument:
		- `self.unpack_data = unpack_data`

### Issue 3: unpack logic incompatible with validation dict format

- Symptom:
	- `FileNotFoundError ... /test.npz` during unpack.
- Root cause:
	- `dataset_loading_llm.unpack_dataset` assumed `train_file['validation']` is a list.
	- Current split format is dict (`{"validation": {"test": [...]}}`), causing incorrect filename construction.
- Fix applied:
	- Made unpack logic robust for both list/dict validation structures.
	- Added skip for missing `.npz` entries with informative message.

### Recommendation before full training

- If using full regenerated manifests, ensure preprocessing cache is rebuilt for all included IDs.
- If you need to start immediately, use a filtered split that matches currently available preprocessed IDs.
