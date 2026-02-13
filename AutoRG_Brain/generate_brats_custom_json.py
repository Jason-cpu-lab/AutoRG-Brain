"""
Generate custom JSON format for BraTS2020 dataset
This script creates a JSON array with entries for each modality of each case
Format: [{"image": "path", "modal": "modality", "label": "seg_path", "label2": "optional"}]
"""
import os
import json
from pathlib import Path

# Dataset configuration
BRATS_DIR = Path('/home/jason/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
OUTPUT_DIR = Path('/home/jason/autorg/AutoRG-Brain/raw_data/Task001_seg_test')
OUTPUT_JSON = OUTPUT_DIR / 'brats_custom_format.json'

# Modality mapping from BraTS to custom format
MODALITY_MAPPING = {
    'flair': 'T2FLAIR'
}

def scan_brats_dataset():
    """Scan BraTS2020 dataset and generate JSON entries for each modality"""
    dataset_entries = []
    
    if not BRATS_DIR.exists():
        print(f"ERROR: Dataset directory not found: {BRATS_DIR}")
        return dataset_entries
    
    print(f"Scanning BraTS2020 dataset: {BRATS_DIR}")
    
    # Get all case directories
    case_dirs = [d for d in BRATS_DIR.iterdir() if d.is_dir() and d.name.startswith('BraTS20_Training')]
    case_dirs.sort()
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"Processing case: {case_id}")
        
        # Process each modality (FLAIR only)
        for brats_modal, custom_modal in MODALITY_MAPPING.items():
            image_file = case_dir / f"{case_id}_{brats_modal}.nii"
            
            if image_file.exists():
                entry = {
                    "image": str(image_file),
                    "modal": custom_modal
                }
                
                dataset_entries.append(entry)
    
    return dataset_entries

def generate_variations(dataset_entries):
    """
    Return entries as-is since we only want image and modal fields
    """
    return dataset_entries

def write_json(filepath, data):
    """Write JSON data to file with proper formatting"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated: {filepath}")
    print(f"Total entries: {len(data)}")

def main():
    print("\n" + "=" * 70)
    print("BraTS2020 Custom JSON Format Generator")
    print("=" * 70 + "\n")
    
    # Scan dataset and generate entries
    print("Step 1: Scanning BraTS2020 dataset...")
    dataset_entries = scan_brats_dataset()
    
    if not dataset_entries:
        print("ERROR: No dataset entries found. Check the dataset path.")
        return
    
    print(f"Found {len(dataset_entries)} FLAIR entries\n")
    
    # Use entries as-is (no variations needed)
    print("Step 2: Preparing final entries...")
    final_entries = generate_variations(dataset_entries)
    
    # Write JSON file
    print("Step 3: Writing JSON file...")
    write_json(OUTPUT_JSON, final_entries)
    
    print("\n" + "=" * 70)
    print("Custom JSON file generated successfully!")
    print("=" * 70 + "\n")
    
    # Show first few entries as preview
    print("Preview of first 4 entries:")
    for i, entry in enumerate(final_entries[:4]):
        print(f"  Entry {i+1}: {json.dumps(entry, indent=4)}")

if __name__ == "__main__":
    main()