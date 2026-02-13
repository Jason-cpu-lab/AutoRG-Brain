"""
Generate case_dic.json, test_file.json, and dataset.json for segmentation task
Works with BraTS2020 dataset structure (per-modality case identifiers)
"""
import os
import json
from pathlib import Path

# Get absolute paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Define BraTS2020 data directory and output paths
brats_dir = Path('/home/jason/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')


# Output paths (use Task001_seg_test as requested)
output_dir = PROJECT_ROOT / 'raw_data' / 'Task001_seg_test'
case_dic_path = output_dir / 'case_dic.json'
dataset_path = output_dir / 'dataset.json'
test_file_path = output_dir / 'test_file.json'


# Only include flair modality
MODALITIES = ["flair"]


# Initialize case dictionary for BraTS2020 (flair only)
case_dic = {"flair": []}

# Initialize dataset structure
# Note: this codebase often treats images as single-modality inputs and uses an extra per-sample "modal" field.
dataset = {
    "description": "BraTS2020 Brain Tumor Segmentation Dataset",
    "labels": {
        "0": "background",
        "1": "NCR/NET",
        "2": "ED",
        "4": "ET"
    },
    "modality": {
        "0": "MRI"
    },
    "name": "BraTS2020_Segmentation",
    "numTest": 0,
    "numTraining": 0,
    "reference": "https://www.med.upenn.edu/cbica/brats2020/data.html",
    "release": "1.0",
    "tensorImageSize": "4D",
    "test": [],
    "training": []
}

# Cases to skip
SKIP_CASES = set()


def parse_case_id_and_modal(stem: str):
    """Split '<case_id>_<modal>' safely (case_id may contain underscores)."""
    for m in MODALITIES:
        suffix = f"_{m}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], m
    return None, None


def populate_case_dic_brats():
    """Scan BraTS2020 dataset and populate case_dic with per-modality stems."""
    if not brats_dir.is_dir():
        print(f"WARNING: Dataset directory not found: {brats_dir}")
        return

    print(f"Scanning BraTS2020 dataset: {brats_dir}")
    for case_id in sorted(os.listdir(brats_dir)):
        if case_id in SKIP_CASES:
            continue

        case_folder = brats_dir / case_id
        if not case_folder.is_dir():
            continue

        for modal in MODALITIES:
            img_path = case_folder / f"{case_id}_{modal}.nii"
            if img_path.exists():
                # store identifier WITH suffix
                case_dic[modal].append(f"{case_id}_{modal}")


def write_json(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated: {filepath}")


def main():
    print("\n" + "=" * 70)
    print("BraTS2020 Segmentation JSON Generator")
    print("=" * 70 + "\n")

    # Step 1: Populate case_dic
    print("Step 1: Populating case_dic.json...")
    populate_case_dic_brats()
    write_json(case_dic_path, case_dic)


    all_stems = sorted(set(case_dic["flair"]))
    unique_case_ids = sorted(set(parse_case_id_and_modal(s)[0] for s in all_stems if parse_case_id_and_modal(s)[0] is not None))

    print(f"  Total stems: {len(all_stems)} (flair only)")
    print(f"  Unique base cases: {len(unique_case_ids)}\n")

    if len(all_stems) == 0:
        print("ERROR: case_dic.json is empty. Check data directories and filenames.")
        return


    # Step 2: Assign all stems to training set (flair only)
    print("Step 2: Assigning all stems to training set (flair only)...")
    training_stems = all_stems
    print(f"  Training stems: {len(training_stems)}")
    print("  Test stems: 0\n")


    # Step 3: Populate dataset.json (one entry per flair stem)
    print("Step 3: Generating dataset.json (flair only)...")
    dataset["training"].clear()

    for stem in training_stems:
        case_id, modal = parse_case_id_and_modal(stem)
        if case_id is None:
            continue

        case_folder = brats_dir / case_id
        img_path = case_folder / f"{case_id}_flair.nii"
        seg_path = case_folder / f"{case_id}_seg.nii"

        entry = {
            "image": str(img_path),
            "label": str(seg_path),
            "modal": "flair"
        }

        if img_path.exists() and seg_path.exists():
            dataset["training"].append(entry)

    dataset["test"] = []
    dataset["numTraining"] = len(dataset["training"])
    dataset["numTest"] = 0

    write_json(dataset_path, dataset)
    print(f"  Training entries written: {dataset['numTraining']}\n")

    # Step 4: Generate test_file.json
    print("Step 4: Generating test_file.json...")
    test_file = {
        "training": training_stems,
        "validation": {
            "test": []
        }
    }
    write_json(test_file_path, test_file)
    print(f"  Training stems: {len(test_file['training'])}")
    print("  Test stems: 0\n")

    print("=" * 70)
    print("All JSON files generated successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()