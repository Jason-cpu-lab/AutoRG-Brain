# ADC Full Integration Update (2026-03-21)

## Summary

This document records the **full ADC modality integration** completed across the AutoRG-Brain codebase.

Goal: add `ADC` as a first-class modality in data loading, training-time routing, and shared-network modality-specific encoding, while keeping MedSAM2/AutoRG pipeline compatibility.

---

## What was updated

### 1) Dataloader modality support

#### `AutoRG_Brain/dataset/dataset_loading.py`
- Added `_adc` suffix recognition in modality key parsing.
- Added fallback prefix recognition for `e` modality bucket.
- Extended modality buckets from 4 to 5 (`a,b,c,d,e`).
- Updated runtime modality map:
  - `a -> DWI`
  - `b -> T1WI`
  - `c -> T2WI`
  - `d -> T2FLAIR`
  - `e -> ADC`
- Extended modality sampling to include ADC.
- Updated modal-bucket logging to print ADC bucket counts.

#### `AutoRG_Brain/dataset/dataset_loading_llm.py`
- Added `ADC` key in `list_of_keys_modal` construction.
- Replaced strict dictionary indexing with safe `.get(...)` access for modality lists.
- Replaced fixed 4-modality random selection with dynamic selection from available modalities.
- Added explicit runtime error when no modality keys are available.

#### `AutoRG_Brain/dataset/dataset_loading_bucket.py`
- Added `ADC` key in `list_of_keys_modal` construction.
- Replaced strict dictionary indexing with safe `.get(...)` access for modality lists.
- Replaced fixed 4-modality random selection with dynamic selection from available modalities.
- Added explicit runtime error when no modality keys are available.

---

### 2) Lesion synthesis / intensity logic

#### `AutoRG_Brain/dataset/utils.py`
- Updated intensity branch in `get_intensity(...)`:
  - from `if modality == "T2WI"`
  - to `if modality in ("T2WI", "ADC")`
- This ensures ADC receives explicit handling consistent with low-signal style logic used in this code path.

---

### 3) Shared UNet modality branches

#### `AutoRG_Brain/network/generic_UNet_share.py`
- Added a dedicated ADC encoder branch:
  - `self.conv_blocks_context_e`
- Added ADC stage stack construction through encoder levels + bottleneck.
- Registered branch as `nn.ModuleList`.
- Added forward routing for `modal == 'ADC'`.
- Added explicit unsupported-modality guard.

#### `AutoRG_Brain/network/generic_UNet_share_get_feature_patchwise.py`
- Same ADC integration pattern as above:
  - `conv_blocks_context_e`
  - encoder/bottleneck creation
  - module registration
  - ADC forward routing
  - unsupported-modality guard

#### `AutoRG_Brain/network/generic_UNet_share_get_feature_patchwise_region.py`
- Same ADC integration pattern as above:
  - `conv_blocks_context_e`
  - encoder/bottleneck creation
  - module registration
  - ADC forward routing
  - unsupported-modality guard

---

### 4) Integration documentation

#### `MEDSAM2_INTEGRATION.md`
- Added note that full ADC integration is implemented.
- Updated modality mapping guidance to include `_adc` and runtime `ADC` naming.
- Updated practical modality naming guidance for JSON usage.
- Added/clarified runtime notes around model/checkpoint environment expectations.

---

## Validation status

- Edited Python files were checked for problems after patching.
- Result: **no errors reported** in the modified Python files.

---

## Compatibility notes

1. **Checkpoint compatibility**
   - Older `share` checkpoints trained before ADC branch addition may be incompatible with the new 5-branch shared-encoder architecture.
   - If needed, use partial load strategies or retrain.

2. **Modality naming contract**
   - Use consistent modality names in JSON/training/inference metadata:
     - `DWI`, `T1WI`, `T2WI`, `T2FLAIR`, `ADC`

3. **Data metadata requirement**
   - Ensure split metadata (e.g., case modality dictionaries) includes ADC entries where expected.

---

## Recommended next checks

1. Run a small ADC-only smoke test (single case or mini-batch) to verify runtime routing logs show `modal=ADC`.
2. Start a short training run and confirm ADC appears in batch sampling distribution.
3. Verify inference JSON generation includes `ADC` where applicable.

---

## Change scope (quick index)

- `AutoRG_Brain/dataset/dataset_loading.py`
- `AutoRG_Brain/dataset/dataset_loading_llm.py`
- `AutoRG_Brain/dataset/dataset_loading_bucket.py`
- `AutoRG_Brain/dataset/utils.py`
- `AutoRG_Brain/network/generic_UNet_share.py`
- `AutoRG_Brain/network/generic_UNet_share_get_feature_patchwise.py`
- `AutoRG_Brain/network/generic_UNet_share_get_feature_patchwise_region.py`
- `MEDSAM2_INTEGRATION.md`
