import os
import re
import glob
import rasterio
import numpy as np
import pandas as pd
from rasterio.windows import Window
from collections import defaultdict
import random

# Configuration
S2_DIR = 'ml/data/sentinel2_downloads'
PATCH_SIZE = 128
STRIDE = 128  # For sliding window
RANDOM_SAMPLES = 100  # Number of random patches per scene
BANDS = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
PATCHES_DIR = 'ml/data/sentinel2_patches'
PAIRS_CSV = 'ml/data/sentinel2_annual_pairs.csv'

os.makedirs(PATCHES_DIR, exist_ok=True)

def extract_date_from_name(name):
    # Example: S2A_MSIL2A_20200312T140051_...
    m = re.search(r'_(\d{8})T', name)
    if m:
        return m.group(1)  # YYYYMMDD
    return None

def extract_year(date_str):
    return date_str[:4]

def find_band_file(safe_dir, band):
    # Looks for the JP2 file for the given band in the SAFE structure (R10m)
    pattern = os.path.join(safe_dir, 'GRANULE', '*', 'IMG_DATA', 'R10m', f'*_{band}_10m.jp2')
    files = glob.glob(pattern)
    return files[0] if files else None

def read_bands(safe_dir, bands=BANDS):
    band_arrays = []
    for band in bands:
        band_file = find_band_file(safe_dir, band)
        if not band_file:
            raise FileNotFoundError(f'Band {band} not found in {safe_dir}')
        with rasterio.open(band_file) as src:
            band_arrays.append(src.read(1))
    arr = np.stack(band_arrays, axis=0)  # Shape: (C, H, W)
    return arr

def save_patch(arr, out_path):
    np.save(out_path, arr)

def extract_patches(arr, year, scene_id, mode='sliding', stride=128, n_random=100):
    C, H, W = arr.shape
    patches = {}
    locations = set()
    # Sliding window
    if mode in ['sliding', 'both']:
        for y in range(0, H - PATCH_SIZE + 1, stride):
            for x in range(0, W - PATCH_SIZE + 1, stride):
                patch = arr[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                loc = (y, x)
                patch_name = f'{year}_{scene_id}_Y{y}_X{x}_slide.npy'
                patch_path = os.path.join(PATCHES_DIR, patch_name)
                save_patch(patch, patch_path)
                patches[loc] = patch_path
                locations.add(loc)
    # Random sampling
    if mode in ['random', 'both']:
        for _ in range(n_random):
            y = random.randint(0, H - PATCH_SIZE)
            x = random.randint(0, W - PATCH_SIZE)
            loc = (y, x)
            if loc in locations:
                continue  # Avoid duplicates
            patch = arr[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_name = f'{year}_{scene_id}_Y{y}_X{x}_rand.npy'
            patch_path = os.path.join(PATCHES_DIR, patch_name)
            save_patch(patch, patch_path)
            patches[loc] = patch_path
            locations.add(loc)
    return patches  # {location: patch_path}

def main():
    # 1. Scan all scenes and group by year
    scenes = defaultdict(list)
    for entry in os.listdir(S2_DIR):
        if entry.endswith('.SAFE'):
            date_str = extract_date_from_name(entry)
            if date_str:
                year = extract_year(date_str)
                scenes[year].append((date_str, entry))
    # 2. For each year, select the earliest scene
    selected = {}
    for year, items in scenes.items():
        items.sort()  # Sort by date
        selected[year] = items[0][1]  # Earliest
    # 3. For each selected scene, extract bands and patches
    year_patches = {}
    for year, scene_dir in selected.items():
        safe_path = os.path.join(S2_DIR, scene_dir)
        arr = read_bands(safe_path)
        scene_id = scene_dir.split('_')[1] if '_' in scene_dir else scene_dir
        patches = extract_patches(arr, year, scene_id, mode='both', stride=STRIDE, n_random=RANDOM_SAMPLES)
        year_patches[year] = patches  # {location: patch_path}
    # 4. Create annual pairs for matching locations
    years = sorted(year_patches.keys())
    pairs = []
    for i in range(len(years)-1):
        year1, year2 = years[i], years[i+1]
        locs1 = set(year_patches[year1].keys())
        locs2 = set(year_patches[year2].keys())
        common_locs = locs1 & locs2
        for loc in common_locs:
            pairs.append({'year1': year1, 'year2': year2,
                          'patch1': year_patches[year1][loc],
                          'patch2': year_patches[year2][loc],
                          'y': loc[0], 'x': loc[1]})
    df = pd.DataFrame(pairs)
    df.to_csv(PAIRS_CSV, index=False)
    print(f'[DONE] Annual patch pairs saved to {PAIRS_CSV}. Total pairs: {len(df)}')

if __name__ == '__main__':
    main() 