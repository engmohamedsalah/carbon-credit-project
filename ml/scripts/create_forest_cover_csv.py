# ml/scripts/create_forest_cover_csv.py

import pandas as pd
from pathlib import Path

def create_forest_cover_csv(
    image_dir='ml/data/sentinel2_downloads/patches_128',
    label_dir='ml/data/hansen_labels/patches_128',
    output_csv='ml/data/forest_cover_patches.csv'
):
    """
    Creates a CSV file mapping Sentinel-2 image patches to their corresponding
    Hansen forest cover label patches.

    Args:
        image_dir (str): Directory containing the Sentinel-2 image patches (.npy).
        label_dir (str): Directory containing the Hansen label patches (.npy).
        output_csv (str): Path to the output CSV file.
    """
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_csv_path = Path(output_csv)

    image_files = sorted(list(image_path.glob("*.npy")))
    print(f"Found {len(image_files)} image patches.")

    if not image_files:
        print(f"Error: No image files found in {image_dir}. Please check the path.")
        return

    records = []
    for img_file in image_files:
        # Construct the expected label filename
        label_filename = img_file.name.replace('_patch_', '_label_patch_')
        label_file = label_path / label_filename
        
        if label_file.exists():
            records.append({
                "image_path": str(img_file.resolve()),
                "label_path": str(label_file.resolve())
            })
        else:
            print(f"Warning: No matching label found for image {img_file.name}")

    if not records:
        print("Error: No matching image-label pairs were found. Aborting.")
        return

    df = pd.DataFrame(records)
    
    # Ensure the output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully created CSV with {len(df)} entries at {output_csv_path}")

if __name__ == '__main__':
    create_forest_cover_csv() 