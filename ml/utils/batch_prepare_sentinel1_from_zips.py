import os
import glob
import zipfile
import shutil
import argparse
from data_preparation import prepare_sentinel1_stacks

def batch_prepare_sentinel1_from_zips(s1_zip_dir, s2_stacks_dir, out_dir, bands=["VV", "VH"]):
    zip_files = sorted(glob.glob(os.path.join(s1_zip_dir, "*.zip")))
    print(f"Found {len(zip_files)} Sentinel-1 zip files.")
    processed_safe_folders = set()
    for zip_path in zip_files:
        base = os.path.basename(zip_path)
        scene_name = base.replace(".SAFE_PRODUCT.zip", "")
        safe_folder = os.path.join(s1_zip_dir, scene_name + ".SAFE")
        if os.path.exists(safe_folder):
            print(f"{safe_folder} already exists, skipping unzip.")
        else:
            print(f"Unzipping {base}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(s1_zip_dir)
            except zipfile.BadZipFile:
                print(f"Warning: {base} is not a valid zip file or is corrupted. Skipping.")
                continue
        try:
            print(f"Processing {safe_folder}...")
            prepare_sentinel1_stacks(s1_zip_dir, s2_stacks_dir, out_dir, bands=bands)
            processed_safe_folders.add(safe_folder)
        except Exception as e:
            print(f"Error processing {safe_folder}: {e}")
        print(f"Deleting {safe_folder}...")
        shutil.rmtree(safe_folder, ignore_errors=True)
    # After all zips, process any .SAFE folders not just unzipped
    all_safe_folders = [os.path.join(s1_zip_dir, f) for f in os.listdir(s1_zip_dir) if os.path.isdir(os.path.join(s1_zip_dir, f)) and f.endswith('.SAFE')]
    for safe_folder in all_safe_folders:
        if safe_folder not in processed_safe_folders:
            try:
                print(f"Processing existing {safe_folder}...")
                prepare_sentinel1_stacks(s1_zip_dir, s2_stacks_dir, out_dir, bands=bands)
            except Exception as e:
                print(f"Error processing {safe_folder}: {e}")
            print(f"Deleting {safe_folder}...")
            shutil.rmtree(safe_folder, ignore_errors=True)
    print("Batch Sentinel-1 preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch unzip and prepare Sentinel-1 scenes from zips.")
    parser.add_argument("--s1_zip_dir", type=str, required=True, help="Directory with Sentinel-1 zip files.")
    parser.add_argument("--s2_stacks_dir", type=str, required=True, help="Directory with prepared Sentinel-2 stacks.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for S1 stacks.")
    args = parser.parse_args()
    batch_prepare_sentinel1_from_zips(args.s1_zip_dir, args.s2_stacks_dir, args.out_dir) 