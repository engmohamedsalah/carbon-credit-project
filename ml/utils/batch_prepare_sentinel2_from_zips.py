import os
import glob
import zipfile
import shutil
import argparse
from data_preparation import prepare_data_for_ml, prepare_sentinel1_stacks

def batch_prepare_sentinel2_from_zips(s2_zip_dir, hansen_dir, out_dir, s2_bands=["B02", "B03", "B04", "B08"], s1_zip_dir=None):
    # Collect all .SAFE folders before starting
    all_safe_folders = [os.path.join(s2_zip_dir, f) for f in os.listdir(s2_zip_dir) if os.path.isdir(os.path.join(s2_zip_dir, f)) and f.lower().endswith('.safe')]
    zip_files = sorted(glob.glob(os.path.join(s2_zip_dir, "*.zip")))
    print(f"Found {len(zip_files)} Sentinel-2 zip files.")
    corrupted_files = []
    for zip_path in zip_files:
        base = os.path.basename(zip_path)
        # Expected .SAFE folder name (replace .SAFE_PRODUCT.zip or .zip with .SAFE)
        expected_safe = base.replace('.SAFE_PRODUCT.zip', '.SAFE').replace('.zip', '.SAFE')
        expected_safe_path = os.path.join(s2_zip_dir, expected_safe)
        if os.path.isdir(expected_safe_path):
            print(f"{expected_safe_path} already exists, skipping unzip.")
            if expected_safe_path not in all_safe_folders:
                all_safe_folders.append(expected_safe_path)
            continue
        print(f"Unzipping {base}...")
        # Record .SAFE folders before extraction
        safe_folders_before = set([f for f in os.listdir(s2_zip_dir) if os.path.isdir(os.path.join(s2_zip_dir, f)) and f.lower().endswith('.safe')])
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(s2_zip_dir)
        except zipfile.BadZipFile:
            print(f"Warning: {base} is not a valid zip file or is corrupted. Skipping.")
            corrupted_files.append(base)
            continue
        # Record .SAFE folders after extraction
        safe_folders_after = set([f for f in os.listdir(s2_zip_dir) if os.path.isdir(os.path.join(s2_zip_dir, f)) and f.lower().endswith('.safe')])
        new_safe_folders = safe_folders_after - safe_folders_before
        if os.path.isdir(expected_safe_path):
            all_safe_folders.append(expected_safe_path)
            print(f"Found expected .SAFE folder: {expected_safe_path}")
        elif new_safe_folders:
            for f in new_safe_folders:
                folder_path = os.path.join(s2_zip_dir, f)
                if folder_path not in all_safe_folders:
                    all_safe_folders.append(folder_path)
            print(f"Found new .SAFE folder(s) after extraction: {[os.path.join(s2_zip_dir, f) for f in new_safe_folders]}")
        else:
            print(f"Error: No .SAFE folder found after unzipping {base}.")
    # Remove duplicates
    all_safe_folders = list(dict.fromkeys(all_safe_folders))
    # Find Hansen files
    hansen_files = glob.glob(os.path.join(hansen_dir, '*.tif'))
    hansen_treecover = next((f for f in hansen_files if 'treecover' in f), None)
    hansen_lossyear = next((f for f in hansen_files if 'lossyear' in f), None)
    hansen_datamask = next((f for f in hansen_files if 'datamask' in f), None)
    if not (hansen_treecover and hansen_lossyear and hansen_datamask):
        print("Error: Missing Hansen files in", hansen_dir)
        return
    # Prepare all S2 scenes
    print(f"Preparing {len(all_safe_folders)} Sentinel-2 scenes...")
    prepare_data_for_ml(all_safe_folders, hansen_treecover, hansen_lossyear, hansen_datamask, out_dir, s2_bands=s2_bands)
    # Delete all extracted .SAFE folders (optional, comment out if you want to keep them)
    # for safe_folder in all_safe_folders:
    #     print(f"Deleting {safe_folder}...")
    #     shutil.rmtree(safe_folder, ignore_errors=True)
    print("Batch Sentinel-2 preparation complete.")
    if corrupted_files:
        print("\nThe following files are corrupted or not valid zips and should be re-downloaded:")
        for f in corrupted_files:
            print(f"  - {f}")
    # Optionally process Sentinel-1 if s1_zip_dir is provided
    if s1_zip_dir:
        print("Starting batch Sentinel-1 preparation...")
        s2_stacks_dir = os.path.join(out_dir, "s2_stacks")
        s1_out_dir = os.path.join(out_dir, "s1_stacks")
        from batch_prepare_sentinel1_from_zips import batch_prepare_sentinel1_from_zips
        batch_prepare_sentinel1_from_zips(s1_zip_dir, s2_stacks_dir, s1_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch unzip and prepare Sentinel-2 scenes from zips, then optionally Sentinel-1.")
    parser.add_argument("--s2_zip_dir", type=str, required=True, help="Directory with Sentinel-2 zip files.")
    parser.add_argument("--hansen_dir", type=str, required=True, default="ml/data/hansen_downloads", help="Directory with Hansen files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for prepared data.")
    parser.add_argument("--s1_zip_dir", type=str, help="Directory with Sentinel-1 zip files (optional).")
    args = parser.parse_args()
    batch_prepare_sentinel2_from_zips(args.s2_zip_dir, args.hansen_dir, args.out_dir, s1_zip_dir=args.s1_zip_dir) 