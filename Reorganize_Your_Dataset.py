import os
import shutil
from pathlib import Path

def reorganize_dataset(source_dir, dest_dir):
    """
    Reorganize images into proper class folders
    Assumes images are named like: Normal-001.jpg, Cyst-045.jpg, etc.
    """
    
    # Create destination directories
    classes = ['Normal', 'Cyst', 'Tumor', 'Stone']
    for class_name in classes:
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)
    
    # Move images to appropriate folders
    for image_file in os.listdir(source_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Extract class name from filename
            for class_name in classes:
                if class_name.lower() in image_file.lower():
                    src = os.path.join(source_dir, image_file)
                    dst = os.path.join(dest_dir, class_name, image_file)
                    shutil.copy2(src, dst)
                    break
    
    print("Dataset reorganization complete!")
    for class_name in classes:
        count = len(os.listdir(os.path.join(dest_dir, class_name)))
        print(f"{class_name}: {count} images")

# Usage
source = "path/to/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
destination = "path/to/CT-KIDNEY-DATASET-ORGANIZED"
reorganize_dataset(source, destination)