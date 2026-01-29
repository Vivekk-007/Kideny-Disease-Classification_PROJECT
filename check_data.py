import os
from pathlib import Path

# Check multiple possible locations
possible_paths = [
    Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"),
    Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"),
    Path("artifacts/data_ingestion"),
]

print("=" * 50)
print("SEARCHING FOR DATASET")
print("=" * 50)

data_dir = None
for path in possible_paths:
    print(f"Checking: {path}")
    if path.exists():
        contents = list(path.iterdir())
        print(f"  Found {len(contents)} items")
        for item in contents[:5]:  # Show first 5 items
            print(f"    - {item.name}")
        
        # Check if this looks like the dataset root
        subdirs = [d for d in contents if d.is_dir()]
        if any(name in ['Cyst', 'Normal', 'Stone', 'Tumor'] for d in subdirs for name in [d.name]):
            data_dir = path
            print(f"  ✓ This is the dataset root!")
            break
    else:
        print(f"  ✗ Does not exist")

if data_dir:
    print("\n" + "=" * 50)
    print("DATA DISTRIBUTION")
    print("=" * 50)
    
    total_images = 0
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            num_images = len(images)
            total_images += num_images
            print(f"{class_dir.name}: {num_images} images")
    
    print(f"\nTotal: {total_images} images")
    print("=" * 50)
else:
    print("\n❌ Dataset not found! Please extract the zip file.")