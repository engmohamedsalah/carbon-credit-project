import os
import glob
import rasterio
import shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

S2_DOWNLOADS = 'ml/data/sentinel2_downloads'
HANSEN_DIR = 'ml/data/hansen_downloads'
T1_OUT = 'ml/data/sentinel_t1'
T2_OUT = 'ml/data/sentinel_t2'
LABELS_OUT = 'ml/data/change_labels'
BANDS = ['B02', 'B03', 'B04', 'B08']

os.makedirs(T1_OUT, exist_ok=True)
os.makedirs(T2_OUT, exist_ok=True)
os.makedirs(LABELS_OUT, exist_ok=True)

def find_safe_folders():
    return sorted(glob.glob(os.path.join(S2_DOWNLOADS, '*.SAFE')))

def extract_tile_and_date(safe_name):
    # Example: S2A_MSIL2A_20201217T140051_N0500_R067_T21MYN_20230312T114956.SAFE
    parts = safe_name.split('_')
    date = parts[2][:8]
    tile = None
    for p in parts:
        if p.startswith('T') and len(p) == 6:
            tile = p
    return tile, date

def get_band_path(safe_folder, band):
    # Find the 10m band .jp2 file for the given band
    pattern = os.path.join(safe_folder, 'GRANULE', '*', 'IMG_DATA', 'R10m', f'*_{band}_10m.jp2')
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def copy_bands(safe_folder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for band in BANDS:
        src = get_band_path(safe_folder, band)
        if src:
            dst = os.path.join(out_dir, f'{band}.tif')
            with rasterio.open(src) as src_ds:
                profile = src_ds.profile.copy()
                profile.update(driver='GTiff')
                with rasterio.open(dst, 'w', **profile) as dst_ds:
                    dst_ds.write(src_ds.read(1), 1)
        else:
            print(f'[WARN] Band {band} not found in {safe_folder}')

def align_and_crop_hansen(s2_band_path, hansen_path, out_path):
    with rasterio.open(s2_band_path) as src_s2, rasterio.open(hansen_path) as src_hansen:
        dst_crs = src_s2.crs
        dst_transform, width, height = calculate_default_transform(
            src_hansen.crs, dst_crs, src_s2.width, src_s2.height, *src_s2.bounds)
        kwargs = src_hansen.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'count': 1
        })
        data = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src_hansen, 1),
            destination=data,
            src_transform=src_hansen.transform,
            src_crs=src_hansen.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        # Binarize: forest loss = 1, else 0 (assuming Hansen loss band)
        data = (data > 0).astype(np.uint8)
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            dst.write(data, 1)

def main():
    safe_folders = find_safe_folders()
    # Group by tile
    tile_dict = {}
    for safe in safe_folders:
        tile, date = extract_tile_and_date(os.path.basename(safe))
        if tile and date:
            tile_dict.setdefault(tile, []).append((date, safe))
    # For each tile, generate all possible t1/t2 pairs (t1_date < t2_date)
    for tile, date_safe_list in tile_dict.items():
        if len(date_safe_list) < 2:
            continue
        date_safe_list.sort()
        n = len(date_safe_list)
        for i in range(n):
            for j in range(i+1, n):
                t1_date, t1_safe = date_safe_list[i]
                t2_date, t2_safe = date_safe_list[j]
                scene_id = f'{tile}_{t1_date}_{t2_date}'
                print(f'[INFO] Preparing scene {scene_id}')
                t1_dir = os.path.join(T1_OUT, scene_id)
                t2_dir = os.path.join(T2_OUT, scene_id)
                copy_bands(t1_safe, t1_dir)
                copy_bands(t2_safe, t2_dir)
                # Use B02.tif as reference for alignment
                s2_band_path = os.path.join(t1_dir, 'B02.tif')
                hansen_path = os.path.join(HANSEN_DIR, 'hansen_clipped_datamask_32721.tif')
                if os.path.exists(hansen_path):
                    label_out = os.path.join(LABELS_OUT, f'{scene_id}_change.tif')
                    align_and_crop_hansen(s2_band_path, hansen_path, label_out)
                    print(f'[INFO] Saved label: {label_out}')
                else:
                    print('[WARN] Reprojected Hansen data not found!')
    print('[DONE] Data preparation complete.')

if __name__ == '__main__':
    main() 