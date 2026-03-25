import nibabel as nib
import numpy as np

nii_path = "/home/jason/autorg/AutoRG-Brain/inference_output/medsam2_phase1/T1/P1_T1_T1_ana_mask.nii.gz"
img = nib.load(nii_path)
data = img.get_fdata()
unique_labels = np.unique(data)
print(f"Number of unique classes: {len(unique_labels)}")
print(f"Unique class labels: {unique_labels}")
if len(unique_labels) == 96:
    print("✅ There are exactly 96 classes.")
else:
    print("❌ There are NOT 96 classes.")