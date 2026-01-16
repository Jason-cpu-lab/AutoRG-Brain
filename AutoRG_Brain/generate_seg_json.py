"""
Generate case_dic.json, test_file.json, and dataset.json for segmentation task
Works with ISLES-2022 dataset structure
"""
import os
import json
from collections import defaultdict
from pathlib import Path

# Get absolute paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

# Define data directories and output paths
data_dirs = {
    'ISLES-2022': PROJECT_ROOT / 'dataset' / 'ISLES-2022' / 'ISLES-2022'
}

# Mask directory for anatomy masks (inference output)
mask_dir = PROJECT_ROOT / 'dataset' / 'masks'

# Output paths
output_dir = PROJECT_ROOT / 'raw_data' / 'Task001_seg_test'
case_dic_path = output_dir / 'case_dic.json'
dataset_path = output_dir / 'dataset.json'
test_file_path = output_dir / 'test_file.json'

# Initialize case dictionary
case_dic = {
    "DWI": [],
    "T1WI": [],
    "T2WI": [],
    "T2FLAIR": []
}

# Initialize dataset structure
dataset = {
    "description": "ISLES-2022 Brain Stroke Segmentation Dataset",
    "labels": {
        "0": "background",
        "1": "stroke_core",
        "2": "stroke_penumbra",
        "3": "label_3",
        "4": "label_4"
    },
    "modality": {
        "0": "MRI"
    },
    "name": "ISLES2022_Segmentation",
    "numTest": 0,
    "numTraining": 0,
    "reference": "https://isles.virtual-imaging.org/",
    "release": "1.0",
    "tensorImageSize": "4D",
    "test": [],
    "training": []
}

# Cases to skip
SKIP_CASES = set()

def get_dataset_name(stem):
    """Determine dataset name from stem"""
    if stem.startswith('sub-strokecase'):
        return 'ISLES-2022'
    return None

def get_case_id(stem, modal, dataset_name):
    """Extract case_id from stem based on modality and dataset"""
    if dataset_name == 'ISLES-2022':
        suffix_map = {'DWI': '_ses-0001_dwi'}
        suffix = suffix_map.get(modal, '')
        if suffix and stem.endswith(suffix):
            return stem[:-len(suffix)]
    return None

def get_image_path(stem, modal, dataset_name, case_id):
    """Construct image path based on stem, modality, dataset, and case_id"""
    if case_id is None or dataset_name != 'ISLES-2022':
        return ""
    
    # ISLES-2022 structure: ISLES-2022/sub-strokecaseXXXX/ses-0001/dwi/sub-strokecaseXXXX_ses-0001_dwi.nii
    case_path = str(data_dirs[dataset_name] / case_id / 'ses-0001' / 'dwi')
    return os.path.join(case_path, f"{stem}.nii")

def get_mask_paths(case_id, dataset_name, modal):
    """Construct mask paths based on case_id, dataset, and modality"""
    if dataset_name != 'ISLES-2022':
        return "", ""
    
    # Determine modality suffix for anatomy mask
    suffix_map = {'DWI': '_ses-0001_dwi'}
    suffix = suffix_map.get(modal, '')
    
    # Construct anatomy mask path (from inference output)
    anatomy_mask = str(mask_dir / f"{case_id}{suffix}_ana.nii.gz") if suffix else ""
    
    # Construct anomaly (ground truth) mask path
    # ISLES-2022 structure: derivatives/sub-strokecaseXXXX/ses-0001/
    anomaly_mask = str(
        data_dirs['ISLES-2022'] / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"
    )
    
    return anatomy_mask, anomaly_mask

def populate_case_dic():
    """Scan ISLES-2022 dataset and populate case_dic"""
    isles_dir = str(data_dirs['ISLES-2022'])
    
    if not os.path.isdir(isles_dir):
        print(f"⚠ Warning: Dataset directory not found: {isles_dir}")
        return
    
    print(f"Scanning ISLES-2022 dataset: {isles_dir}")
    
    # Scan ISLES-2022 directory
    for case in sorted(os.listdir(isles_dir)):
        if not case.startswith("sub-strokecase"):
            continue
        
        case_dir = os.path.join(isles_dir, case, "ses-0001", "dwi")
        if not os.path.isdir(case_dir):
            continue
        
        # Look for DWI images
        for filename in os.listdir(case_dir):
            if filename.endswith('_dwi.nii') and not filename.endswith('.json'):
                # Extract stem (remove .nii or .nii.gz)
                stem = filename.replace('.nii', '')
                case_dic['DWI'].append(stem)
                print(f"  Found: {stem}")
                break

def write_json(filepath, data):
    """Write data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Generated: {filepath}")

def main():
    """Main function to generate all JSON files"""
    print("\n" + "="*70)
    print("Segmentation JSON Generator")
    print("="*70 + "\n")
    
    # Step 1: Populate case_dic
    print("Step 1: Populating case_dic.json...")
    populate_case_dic()
    
    # Write case_dic.json
    write_json(case_dic_path, case_dic)
    
    total_cases = sum(len(v) for v in case_dic.values())
    print(f"  Total cases: {total_cases} entries across {len(case_dic)} modalities\n")
    
    if not case_dic or total_cases == 0:
        print("⚠ Error: case_dic.json is empty. Check data directories.")
        return
    
    # Step 2: Collect unique case_ids and stems
    print("Step 2: Processing case IDs...")
    dataset_to_cases = defaultdict(list)
    case_to_stems = defaultdict(list)
    
    for modal, stems in case_dic.items():
        for stem in stems:
            dataset_name = get_dataset_name(stem)
            if dataset_name is None:
                continue
            case_id = get_case_id(stem, modal, dataset_name)
            if case_id is None or case_id in SKIP_CASES:
                continue
            case_to_stems[case_id].append((stem, modal, dataset_name))
            dataset_to_cases[dataset_name].append(case_id)
    
    # Ensure case IDs are unique and sorted
    for dataset_name in dataset_to_cases:
        dataset_to_cases[dataset_name] = sorted(list(set(dataset_to_cases[dataset_name])))
    
    print(f"  Processed {len(case_to_stems)} unique cases\n")
    
    # Step 3: Assign all cases to training set
    print("Step 3: Assigning all cases to training set...")
    training_case_ids = set()
    for dataset_name in sorted(dataset_to_cases.keys()):
        case_ids = dataset_to_cases[dataset_name]
        training_case_ids.update(case_ids)
    print(f"  Training: {len(training_case_ids)} cases")
    print(f"  Test: 0 cases\n")

    # Step 4: Populate dataset.json
    print("Step 4: Generating dataset.json...")
    for dataset_name in sorted(dataset_to_cases.keys()):
        for case_id in dataset_to_cases[dataset_name]:
            for stem, modal, ds_name in sorted(case_to_stems[case_id], key=lambda x: x[1]):
                if ds_name != dataset_name:
                    continue
                image_path = get_image_path(stem, modal, dataset_name, case_id)
                if not os.path.exists(image_path):
                    print(f"  ⚠ Warning: Image not found: {image_path}")
                    continue
                anatomy_mask, anomaly_mask = get_mask_paths(case_id, dataset_name, modal)
                entry = {
                    "image": image_path,
                    "label1": anatomy_mask,
                    "label2": anomaly_mask if os.path.exists(anomaly_mask) else "",
                    "modal": modal
                }
                if os.path.exists(anatomy_mask):
                    dataset["training"].append(entry)
    dataset["test"] = []
    dataset["numTraining"] = len(dataset["training"])
    dataset["numTest"] = 0
    write_json(dataset_path, dataset)
    print(f"  Training entries: {dataset['numTraining']}")
    print(f"  Test entries: {dataset['numTest']}\n")

    # Step 5: Generate test_file.json
    print("Step 5: Generating test_file.json...")
    test_file = {
        "training": [
            stem for dataset_name in sorted(dataset_to_cases.keys())
            for case_id in sorted(dataset_to_cases[dataset_name])
            for stem, _, _ in sorted(case_to_stems[case_id], key=lambda x: x[1])
        ],
        "validation": {
            "test": []
        }
    }
    write_json(test_file_path, test_file)
    print(f"  Training cases: {len(test_file['training'])}")
    print(f"  Test cases: 0\n")
    
    print("="*70)
    print("✓ All JSON files generated successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()