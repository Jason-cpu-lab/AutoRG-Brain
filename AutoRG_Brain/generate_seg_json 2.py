import os
import json
from collections import defaultdict

# Define data directories and output paths
data_dirs = {
    'BraTS2020': '/data/cwang/mri_data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
    'BraTS2021': '/data/cwang/mri_data/BraTS2021',
    'BraTS-MEN': '/data/cwang/mri_data/BraTS-MEN',
    'ISLES-2022': '/data/cwang/mri_data/ISLES-2022'
}
# Mask directory remains the updated V2 path
mask_dir = '/data/cwang/mri_data/inference_output.v2'
case_dic_path = '/data/cwang/mri_data/raw_data/nnUNet_raw_data/Task001_seg_test/case_dic.json'
dataset_path = '/data/cwang/mri_data/raw_data/nnUNet_raw_data/Task001_seg_test/dataset.json'
test_file_path = '/data/cwang/mri_data/raw_data/nnUNet_raw_data/Task001_seg_test/test_file.json'

# Initialize case dictionary
case_dic = {
    "DWI": [],
    "T1WI": [],
    "T2WI": [],
    "T2FLAIR": []
}

# Initialize dataset structure
dataset = {
    "description": "the training dataset",
    "labels": {
        "0": "background",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4"
    },
    "modality": {
        "0": "MRI"
    },
    "name": "dataset_name",
    "numTest": 0,
    "numTraining": 0,
    "reference": "no",
    "release": "0.0",
    "tensorImageSize": "4D",
    "test": [],
    "training": []
}

# Cases to skip
SKIP_CASES = {'BraTS20_Training_355'}

# Function to determine dataset name from stem
def get_dataset_name(stem):
    if stem.startswith('BraTS20_Training_'):
        return 'BraTS2020'
    elif stem.startswith('BraTS2021_'):
        return 'BraTS2021'
    elif stem.startswith('BraTS-MEN-'):
        return 'BraTS-MEN'
    elif stem.startswith('sub-strokecase'):
        return 'ISLES-2022'
    return None

# Function to extract case_id from stem based on modality and dataset
def get_case_id(stem, modal, dataset_name):
    suffix = ''
    if dataset_name in ['BraTS2020', 'BraTS2021']:
        suffix_map = {'T1WI': '_t1', 'T1C': '_t1ce', 'T2WI': '_t2', 'T2FLAIR': '_flair', 'DWI': '_dwi'}
    elif dataset_name == 'BraTS-MEN':
        suffix_map = {'T1WI': '-t1n', 'T1C': '-t1c', 'T2WI': '-t2w', 'T2FLAIR': '-t2f'}
    elif dataset_name == 'ISLES-2022':
        # Only check for DWI suffix as per user request
        suffix_map = {'DWI': '_ses-0001_dwi'}
    
    suffix = suffix_map.get(modal, '')
    
    if suffix and stem.endswith(suffix):
        return stem[:-len(suffix)]
    return None

# Function to construct image path based on stem, modality, dataset, and case_id
def get_image_path(stem, modal, dataset_name, case_id):
    if case_id is None:
        return ""
    ext = '.nii.gz' if dataset_name in ['BraTS2021', 'ISLES-2022'] else '.nii'
    sub_dir = ''
    if dataset_name == 'ISLES-2022':
        # Only DWI is supported here
        sub_dir = 'dwi' if modal == 'DWI' else ''
    case_dir = os.path.join(data_dirs[dataset_name], case_id)
    if dataset_name == 'ISLES-2022':
        case_dir = os.path.join(case_dir, 'ses-0001')
    if sub_dir:
        case_dir = os.path.join(case_dir, sub_dir)
    return os.path.join(case_dir, f"{stem}{ext}")

# Function to construct mask paths based on case_id, dataset, and modality
def get_mask_paths(case_id, dataset_name, modal):
    anatomy_mask = ""
    anomaly_mask = ""

    # 1. Determine the correct modality suffix for the ANATOMY mask filename (for v2)
    suffix = ''
    if dataset_name in ['BraTS2020', 'BraTS2021']:
        suffix_map = {'T1WI': '_t1', 'T1C': '_t1ce', 'T2WI': '_t2', 'T2FLAIR': '_flair', 'DWI': '_dwi'}
        suffix = suffix_map.get(modal, '')
    elif dataset_name == 'BraTS-MEN':
        suffix_map = {'T1WI': '-t1n', 'T1C': '-t1c', 'T2WI': '-t2w', 'T2FLAIR': '-t2f'}
        suffix = suffix_map.get(modal, '')
    elif dataset_name == 'ISLES-2022':
        # Only DWI mask path is constructed for ISLES-2022
        suffix_map = {'DWI': '_ses-0001_dwi'}
        suffix = suffix_map.get(modal, '')
    
    # Construct the anatomy mask path using the dynamic suffix and the new mask_dir
    if suffix:
        anatomy_mask = os.path.join(mask_dir, f"{case_id}{suffix}_ana.nii.gz")
    
    # 2. Determine the ANOMALY (Ground Truth) mask path (unchanged from original logic)
    if dataset_name == 'BraTS2020':
        anomaly_mask = os.path.join(data_dirs['BraTS2020'], case_id, f"{case_id}_seg.nii")
    elif dataset_name == 'BraTS2021':
        anomaly_mask = os.path.join(data_dirs['BraTS2021'], case_id, f"{case_id}_seg.nii.gz")
    elif dataset_name == 'BraTS-MEN':
        anomaly_mask = os.path.join(data_dirs['BraTS-MEN'], case_id, f"{case_id}-seg.nii")
    elif dataset_name == 'ISLES-2022':
        anomaly_mask = os.path.join(data_dirs['ISLES-2022'], "derivatives", case_id, "ses-0001", f"{case_id}_ses-0001_msk.nii.gz")
        
    return anatomy_mask, anomaly_mask

# Generate case_dic.json
# BraTS2020
for case in sorted(os.listdir(data_dirs['BraTS2020'])):
    if case in SKIP_CASES:
        continue
    case_dir = os.path.join(data_dirs['BraTS2020'], case)
    if not os.path.isdir(case_dir):
        continue
    for modal, suffix in [('T1WI', '_t1'), ('T1C', '_t1ce'), ('T2WI', '_t2'), ('T2FLAIR', '_flair')]:
        image_path = os.path.join(case_dir, f"{case}{suffix}.nii")
        if os.path.exists(image_path) and modal in case_dic:
            stem = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')
            case_dic[modal].append(stem)

# BraTS2021
for case in sorted(os.listdir(data_dirs['BraTS2021'])):
    case_dir = os.path.join(data_dirs['BraTS2021'], case)
    if not os.path.isdir(case_dir):
        continue
    for modal, suffix in [('T1WI', '_t1'), ('T1C', '_t1ce'), ('T2WI', '_t2'), ('T2FLAIR', '_flair')]:
        image_path = os.path.join(case_dir, f"{case}{suffix}.nii.gz")
        if os.path.exists(image_path) and modal in case_dic:
            stem = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')
            case_dic[modal].append(stem)

# BraTS-MEN
for case in sorted(os.listdir(data_dirs['BraTS-MEN'])):
    case_dir = os.path.join(data_dirs['BraTS-MEN'], case)
    if not os.path.isdir(case_dir):
        continue
    for modal, suffix in [('T1WI', '-t1n'), ('T1C', '-t1c'), ('T2WI', '-t2w'), ('T2FLAIR', '-t2f')]:
        image_path = os.path.join(case_dir, f"{case}{suffix}.nii")
        if os.path.exists(image_path) and modal in case_dic:
            stem = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')
            case_dic[modal].append(stem)

# ISLES-2022
for case in sorted(os.listdir(data_dirs['ISLES-2022'])):
    if not case.startswith("sub-strokecase"):
        continue
    case_dir = os.path.join(data_dirs['ISLES-2022'], case, "ses-0001")
    if not os.path.isdir(case_dir):
        continue
    # --- IMPORTANT FIX: Only include DWI, exclude T2FLAIR ---
    for modal, sub_dir, suffix in [('DWI', 'dwi', '_dwi')]: 
        image_path = os.path.join(case_dir, sub_dir, f"{case}_ses-0001{suffix}.nii.gz")
        if os.path.exists(image_path) and modal in case_dic:
            stem = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')
            case_dic[modal].append(stem)

# Write case_dic.json
os.makedirs(os.path.dirname(case_dic_path), exist_ok=True)
with open(case_dic_path, 'w') as f:
    json.dump(case_dic, f, indent=2)
print(f"case_dic.json generated at {case_dic_path} with total {sum(len(v) for v in case_dic.values())} entries across {len(case_dic)} modalities.")

# Verify case_dic is not empty
if not case_dic:
    print("Error: case_dic.json is empty.")
    exit(1)

# Collect unique case_ids and associated stems, grouped by dataset
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

# Ensure case IDs are unique and sorted within each dataset
for dataset_name in dataset_to_cases:
    dataset_to_cases[dataset_name] = sorted(list(set(dataset_to_cases[dataset_name])))

# Split case_ids for each dataset (80% training, 20% test)
training_case_ids = set()
test_case_ids = set()
for dataset_name in sorted(dataset_to_cases.keys()): 
    case_ids = dataset_to_cases[dataset_name]
    total_cases = len(case_ids)
    split_point = int(0.8 * total_cases)
    training_case_ids.update(case_ids[:split_point])
    test_case_ids.update(case_ids[split_point:])

# Populate dataset.json (training and test)
for dataset_name in sorted(dataset_to_cases.keys()): 
    for case_id in dataset_to_cases[dataset_name]:
        for stem, modal, ds_name in sorted(case_to_stems[case_id], key=lambda x: x[1]): 
            if ds_name != dataset_name:
                continue
            image_path = get_image_path(stem, modal, dataset_name, case_id)
            if not os.path.exists(image_path):
                continue
            
            anatomy_mask, anomaly_mask = get_mask_paths(case_id, dataset_name, modal) 
            
            entry = {
                "image": image_path,
                "label1": anatomy_mask,
                "label2": anomaly_mask if os.path.exists(anomaly_mask) else "",
                "modal": modal
            }
            
            # Training logic: must check for mask existence
            if case_id in training_case_ids:
                if os.path.exists(anatomy_mask):
                    dataset["training"].append(entry)
                # Note: No printout for missing masks, as per user's final requirement (exclusion is now intended)
            
            # Test logic: no check for mask existence
            elif case_id in test_case_ids:
                dataset["test"].append(entry)

# Update numTraining and numTest
dataset["numTraining"] = len(dataset["training"])
dataset["numTest"] = len(dataset["test"])

# Write dataset.json
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
with open(dataset_path, 'w') as f:
    json.dump(dataset, f, indent=2)
print(f"dataset.json generated at {dataset_path} with {dataset['numTraining']} training entries and {dataset['numTest']} test entries.")

# Create test_file.json structure with stems (including modality suffixes)
test_file = {
    "training": [
        stem for dataset_name in sorted(dataset_to_cases.keys())
        for case_id in sorted(dataset_to_cases[dataset_name])
        if case_id in training_case_ids
        for stem, _, _ in sorted(case_to_stems[case_id], key=lambda x: x[1]) 
    ],
    "validation": {
        "test": [
            stem for dataset_name in sorted(dataset_to_cases.keys())
            for case_id in sorted(dataset_to_cases[dataset_name])
            if case_id in test_case_ids
            for stem, _, _ in sorted(case_to_stems[case_id], key=lambda x: x[1])
        ]
    }
}

# Write test_file.json
os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
with open(test_file_path, 'w') as f:
    json.dump(test_file, f, indent=2)
print(f"test_file.json generated at {test_file_path} with {len(test_file['training'])} training cases and {len(test_file['validation']['test'])} validation cases.")