"""
Generate ISLES-2022 DWI anomaly/anatomy masks using AutoRG segmentation model.

This script scans all DWI images under the ISLES-2022 dataset root and writes
predicted masks into one output directory (default:
/home/jason/autorg/inference_output/T1_priority/isles2022_t1_priority_run/DWI).

Outputs per image:
- <folder_order_case_id>_ab.nii.gz
- <folder_order_case_id>_ana.nii.gz

Where <folder_order_case_id> is derived from folder structure, typically:
- sub-strokecaseXXXX_ses-XXXX_dwi

A manifest JSON mapping each source image to generated masks is also written to:
- <output_dir>/isles2022_dwi_masks_manifest.json
"""

import argparse
import json
import os
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEG_FOLDER = (
    SCRIPT_DIR.parent
    / "trained_model_output"
    / "nnUNet"
    / "3d_fullres"
    / "Task001_seg_test"
    / "nnUNetTrainerV2__nnUNetPlansv2.1"
    / "fold_0"
)


def _is_nifti_file(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def collect_isles_dwi_images(dataset_root: Path):
    """Collect DWI NIfTI image paths from ISLES-2022.

    Handles both layouts:
    1) Direct files: .../*_dwi.nii or .../*_dwi.nii.gz
    2) Directory-wrapped NIfTI: .../*_dwi.nii/<inner>.nii
    """
    found = set()

    # Direct files
    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if (low.endswith("_dwi.nii") or low.endswith("_dwi.nii.gz")) and _is_nifti_file(p):
            found.add(str(p))

    # Directory-wrapped files, e.g. sub-..._dwi.nii/<actual>.nii
    for d in dataset_root.rglob("*_dwi.nii"):
        if not d.is_dir():
            continue
        for inner in d.rglob("*"):
            if inner.is_file() and _is_nifti_file(inner):
                found.add(str(inner))

    return sorted(found)


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def derive_case_id_from_path(image_path: Path) -> str:
    """Derive stable case id from ISLES folder hierarchy.

    Preferred format:
        sub-strokecaseXXXX_ses-XXXX_dwi
    """
    subj = None
    ses = None
    for part in image_path.parts:
        if subj is None and re.fullmatch(r"sub-strokecase[^/]*", part, flags=re.I):
            subj = part
        if ses is None and re.fullmatch(r"ses-[^/]+", part, flags=re.I):
            ses = part

    if subj and ses:
        return f"{subj}_{ses}_dwi"

    # Fallback to wrapper folder: sub-..._dwi.nii -> sub-..._dwi
    parent = image_path.parent.name
    if parent.lower().endswith("_dwi.nii"):
        return strip_nii(parent)

    # Last fallback to inner image stem
    return strip_nii(image_path.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ISLES-2022 DWI ana/ab masks")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/jason/autorg/dataset/ISLES-2022/ISLES-2022"),
        help="ISLES-2022 root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jason/autorg/inference_output/T1_priority/isles2022_t1_priority_run/DWI"),
        help="Directory to save generated ana/ab masks",
    )
    parser.add_argument(
        "--seg-folder",
        type=Path,
        default=DEFAULT_SEG_FOLDER,
        help=f"Folder containing segmentation checkpoint files (default: {DEFAULT_SEG_FOLDER})",
    )
    parser.add_argument(
        "--seg-chk",
        type=str,
        default="model_latest",
        help="Segmentation checkpoint name (without .model) (default: model_latest)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse args first so `--help` can exit cleanly before heavyweight imports.
    # Also normalize cwd for inferenceSdk relative paths (e.g. utils_file/*).
    os.chdir(str(SCRIPT_DIR))

    # Delayed import keeps CLI ergonomics and avoids import-time dependency issues.
    from inference.inferenceSdk import SegModel

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not args.seg_folder.exists():
        raise FileNotFoundError(f"Segmentation checkpoint folder not found: {args.seg_folder}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_isles_dwi_images(args.dataset_root)
    if len(images) == 0:
        raise RuntimeError(f"No DWI images found under: {args.dataset_root}")

    print(f"Found {len(images)} ISLES-2022 DWI images")

    config = {
        "seg_folder": str(args.seg_folder),
        "seg_chk": args.seg_chk,
        "output_dir": str(args.output_dir),
    }

    seg_model = SegModel(config)

    list_of_lists = [[p] for p in images]
    case_ids = [derive_case_id_from_path(Path(p)) for p in images]
    # IMPORTANT: keep these as None so SegModel actually performs prediction and writes outputs.
    # In inferenceSdk, non-None values are treated as pre-existing masks and skipped.
    list_of_ab_segs = [None] * len(images)
    list_of_ana_segs = [None] * len(images)
    modals = ["DWI"] * len(images)

    output_ab_filenames, output_ana_filenames = seg_model.seg(
        list_of_lists=list_of_lists,
        list_of_ab_segs=list_of_ab_segs,
        list_of_ana_segs=list_of_ana_segs,
        modals=modals,
    )

    manifest = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "num_images": len(images),
        "seg_folder": str(args.seg_folder),
        "seg_chk": args.seg_chk,
        "entries": [
            {
                "image": images[i],
                "case_id": case_ids[i],
                "modal": "DWI",
                "ab_mask": output_ab_filenames[i],
                "ana_mask": output_ana_filenames[i],
            }
            for i in range(len(images))
        ],
    }

    manifest_path = args.output_dir / "isles2022_dwi_masks_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Done. Generated/verified masks:")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
