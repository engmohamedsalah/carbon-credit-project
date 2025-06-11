# ml/scripts/extract_sentinel2_patches.py

import os
import glob
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rasterio.windows import Window
from itertools import product
import logging
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_band_file(granule_dir, band):
    """Find a band file in a granule directory, searching all resolution folders."""
    # Search for files matching the pattern like T21MYN_20240724T135709_B02_10m.jp2
    search_pattern = f"**/T*_{band}_*.jp2"
    files = list(Path(granule_dir).glob(search_pattern))
    if files:
        return files[0]

    # Fallback for slightly different naming conventions
    search_pattern_alt = f"**/*_{band}.jp2"
    files = list(Path(granule_dir).glob(search_pattern_alt))
    if files:
        return files[0]

    logger.warning(f"No file found for band {band} in {granule_dir}")
    return None

def tile_raster(raster_path, label_raster_path, out_dir, label_out_dir, patch_size=128):
    """
    Tiles a raster and its corresponding label raster into smaller patches
    and saves them as .npy files.
    """
    path = Path(raster_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = Path(label_raster_path)
    label_out_dir = Path(label_out_dir)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path) as src:
        # Reproject label raster to match the source raster's CRS, transform, and dimensions
        with rasterio.open(label_raster_path) as label_src:
            with WarpedVRT(label_src, crs=src.crs, transform=src.transform, 
                         width=src.width, height=src.height,
                         resampling=Resampling.nearest) as vrt:

                width = src.width
                height = src.height

                # Create a list of tile windows
                offsets = product(range(0, width, patch_size), range(0, height, patch_size))
                windows = [Window(x_off, y_off, patch_size, patch_size) for x_off, y_off in offsets]

                logger.info(f"Tiling {path.name} and reprojected {label_path.name} into {len(windows)} patches...")
                patch_count = 0
                for i, window in enumerate(tqdm(windows, desc=f"Tiling {path.stem}")):
                    # Ensure the window does not go beyond the raster boundaries
                    width_val = window.width
                    height_val = window.height
                    if window.col_off + window.width > width:
                        width_val = width - window.col_off
                    if window.row_off + window.height > height:
                        height_val = height - window.row_off
                    
                    window = Window(window.col_off, window.row_off, width_val, height_val)

                    # Skip incomplete edge patches
                    if window.width != patch_size or window.height != patch_size:
                        continue
                    
                    try:
                        # Read the data from the window for both rasters
                        patch = src.read(window=window)
                        label_patch = vrt.read(window=window)
                    except Exception as e:
                        logger.warning(f"Error reading window {i} for {path.name}: {e}. Skipping patch.")
                        continue

                    # Save the patches as .npy files
                    patch_filename = f"{path.stem}_patch_{patch_count}.npy"
                    label_filename = f"{path.stem}_label_patch_{patch_count}.npy"
                    np.save(out_dir / patch_filename, patch)
                    np.save(label_out_dir / label_filename, label_patch)
                    patch_count += 1

def process_safe_directory(safe_dir, out_dir, bands_to_find=['B02', 'B03', 'B04', 'B08'], hansen_datamask_path=None):
    """
    Processes a single .SAFE directory to extract, stack, and tile specified bands.
    Also tiles the corresponding area from the Hansen datamask.
    """
    if not hansen_datamask_path:
        logger.error("Hansen datamask path not provided. Cannot create label patches.")
        return
        
    safe_path = Path(safe_dir)
    logger.info(f"Processing directory: {safe_path.name}")
    
    granule_dir = next(safe_path.glob('GRANULE/*'), None)
    if not granule_dir:
        logger.error(f"No GRANULE folder found in {safe_path}")
        return

    band_paths = [find_band_file(granule_dir, band) for band in bands_to_find]

    if not all(band_paths):
        logger.error(f"Could not find all required bands in {granule_dir}. Skipping.")
        return

    # Read bands and stack them
    try:
        # Use the first band to get metadata, then upsample all bands to 10m
        with rasterio.open(band_paths[0]) as src:
            target_profile = src.meta.copy()
            target_height = src.height
            target_width = src.width

        stack = []
        for p in band_paths:
            with rasterio.open(p) as src:
                if src.height != target_height or src.width != target_width:
                    # Resample to 10m resolution if needed
                    data = src.read(
                        out_shape=(src.count, target_height, target_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                else:
                    data = src.read()
                stack.append(data[0]) # Read the first band of the array
        
        stacked_data = np.stack(stack)

    except Exception as e:
        logger.error(f"Error reading, resampling, or stacking bands for {safe_path.name}: {e}")
        return

    # Save the temporary stacked raster
    target_profile['count'] = len(bands_to_find)
    target_profile['driver'] = 'GTiff'
    
    temp_stack_dir = Path(out_dir) / 'temp_stacks'
    temp_stack_dir.mkdir(exist_ok=True, parents=True)
    stack_filename = temp_stack_dir / f"{safe_path.stem}_stack.tif"

    with rasterio.open(stack_filename, 'w', **target_profile) as dst:
        dst.write(stacked_data)

    # Tile the stacked raster and the corresponding label raster
    patches_out_dir = Path(out_dir) / 'patches_128'
    labels_out_dir = Path('ml/data/hansen_labels/patches_128')
    tile_raster(stack_filename, hansen_datamask_path, patches_out_dir, labels_out_dir)
    
    # Clean up the temporary stack file
    os.remove(stack_filename)

def main():
    s2_dir = 'ml/data/sentinel2_downloads'
    out_dir = 'ml/data/sentinel2_downloads' # Save stacks and patches inside
    hansen_dir = 'ml/data/hansen_downloads'

    # The 12 essential bands for land cover analysis
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    # Find the Hansen treecover file
    try:
        hansen_treecover_path = next(Path(hansen_dir).glob('*treecover*.tif'))
    except StopIteration:
        logger.error(f"Could not find the Hansen treecover GeoTIFF in {hansen_dir}. Exiting.")
        return

    safe_directories = [d for d in Path(s2_dir).iterdir() if d.is_dir() and d.name.endswith('.SAFE')]
    
    if not safe_directories:
        logger.info(f"No .SAFE directories found in {s2_dir}. Exiting.")
        return
        
    logger.info(f"Found {len(safe_directories)} .SAFE directories to process.")
    
    for safe_dir in safe_directories:
        process_safe_directory(str(safe_dir), out_dir, bands_to_find=bands, hansen_datamask_path=hansen_treecover_path)

if __name__ == '__main__':
    main() 