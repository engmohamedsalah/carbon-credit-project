# ml/utils/data_preparation.py

import os
import sys
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import subprocess
import glob
import shutil
import requests
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from rasterio.enums import Resampling

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def download_file(url, destination_path, chunk_size=8192):
    """Download a file from a URL to a destination path with progress."""
    try:
        response = requests.get(url, stream=True, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination_path, 'wb') as f:
            if total_size == 0: # No content-length header
                logger.info(f"Downloading {os.path.basename(destination_path)} (size unknown)")
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            else:
                logger.info(f"Downloading {os.path.basename(destination_path)} ({total_size / (1024*1024):.2f} MB)")
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=os.path.basename(destination_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        logger.info(f"Successfully downloaded {destination_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(destination_path): # Clean up partially downloaded file
            os.remove(destination_path)
        return False
    except IOError as e: # Handle file system errors
        logger.error(f"Failed to write to {destination_path}: {e}")
        return False

def calculate_ndvi(red, nir):
    """Calculate Normalized Difference Vegetation Index."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1, 1)

# --- Sentinel-2 Data Functions ---

S2_BAND_RESOLUTIONS = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20,
    "B06": 20, "B07": 20, "B08": 10, "B8A": 20, "B09": 60,
    "B10": 60, "B11": 20, "B12": 20
}

def download_sentinel2_data(aoi_geojson, start_date, end_date, output_dir):
    """
    Download Sentinel-2 L2A data for AOI and time range using sentinelsat.
    If .SAFE directories are already present in output_dir, skip download and return their paths.
    Requires Copernicus Data Space Ecosystem credentials (SENTINEL_USER, SENTINEL_PASSWORD env vars).
    """
    import glob
    # Check for existing .SAFE directories
    safe_dirs = glob.glob(os.path.join(output_dir, "*.SAFE"))
    if safe_dirs:
        logger.info(f"Found {len(safe_dirs)} existing .SAFE directories in {output_dir}. Skipping download.")
        return safe_dirs

    try:
        from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    except ImportError:
        logger.error("sentinelsat package not installed. Install with: pip install sentinelsat")
        return []

    os.makedirs(output_dir, exist_ok=True)
    user = os.getenv("SENTINEL_USER")
    password = os.getenv("SENTINEL_PASSWORD")
    if not user or not password:
        logger.error("SENTINEL_USER and SENTINEL_PASSWORD environment variables must be set.")
        return []

    # Use the Copernicus Data Space Ecosystem API endpoint (without instance_id)
    api = SentinelAPI(user, password, 'https://catalogue.dataspace.copernicus.eu/odata/v1/')
    footprint = geojson_to_wkt(read_geojson(aoi_geojson))

    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname="Sentinel-2",
        producttype="S2MSI2A",
        cloudcoverpercentage=(0, 20)
    )

    if not products:
        logger.warning(f"No Sentinel-2 scenes found for the given parameters")
        return []
    logger.info(f"Found {len(products)} Sentinel-2 scenes")

    downloaded_scenes = []
    for product_id, product_info in products.items():
        scene_dir_safe = os.path.join(output_dir, product_info["title"] + ".SAFE")
        zip_path = os.path.join(output_dir, f"{product_info['title']}.zip")

        if os.path.exists(scene_dir_safe):
            logger.info(f"Scene {product_info['title']} already exists (extracted)")
            downloaded_scenes.append(scene_dir_safe)
            continue
        if not os.path.exists(zip_path):
            logger.info(f"Downloading {product_info['title']}...")
            try:
                api.download(product_id, directory_path=output_dir)
            except Exception as e:
                logger.error(f"Failed to download {product_info['title']}: {e}")
                continue

        if os.path.exists(zip_path):
            import zipfile
            logger.info(f"Extracting {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(output_dir)
                if os.path.isdir(scene_dir_safe):
                    downloaded_scenes.append(scene_dir_safe)
                    logger.info(f"Extracted to {scene_dir_safe}")
                    os.remove(zip_path)
                else:
                    logger.error(f"Extraction did not create expected directory: {scene_dir_safe}")
            except Exception as e:
                logger.error(f"Failed to extract {zip_path}: {e}")
                if os.path.exists(zip_path): os.remove(zip_path) # Clean up bad zip

    return downloaded_scenes

def prepare_scene_bands(scene_path, bands, output_dir):
    """
    Extracts and stacks specified bands from a Sentinel-2 .SAFE scene.
    Applies a cloud mask using the SCL band (8, 9, 10 = cloud/cirrus).
    Saves the stack and a cloud mask raster (1=cloud, 0=clear) at 10m.
    """
    import glob
    import rasterio
    import numpy as np
    import os
    import logging
    from rasterio.enums import Resampling
    logger = logging.getLogger(__name__)

    granule_dir = os.path.join(scene_path, "GRANULE")
    granule_subdirs = [os.path.join(granule_dir, d) for d in os.listdir(granule_dir) if os.path.isdir(os.path.join(granule_dir, d))]
    if not granule_subdirs:
        logger.error(f"No GRANULE subdirectories found in {scene_path}")
        return None
    granule = granule_subdirs[0]
    img_data_dir_10m = os.path.join(granule, "IMG_DATA", "R10m")
    img_data_dir_20m = os.path.join(granule, "IMG_DATA", "R20m")
    if not os.path.isdir(img_data_dir_10m) or not os.path.isdir(img_data_dir_20m):
        logger.error(f"R10m or R20m directory not found in {granule}/IMG_DATA")
        return None

    # 1. Read bands at 10m
    band_arrays = []
    meta = None
    for band in bands:
        pattern = f"*_{band}_10m.jp2"
        band_files = glob.glob(os.path.join(img_data_dir_10m, pattern))
        if not band_files:
            logger.error(f"Band {band} not found in {img_data_dir_10m} with pattern {pattern}")
            return None
        band_file = band_files[0]
        with rasterio.open(band_file) as src:
            arr = src.read(1)
            band_arrays.append(arr)
            if meta is None:
                meta = src.meta.copy()

    # 2. Find and read SCL band at 20m
    scl_files = glob.glob(os.path.join(img_data_dir_20m, "*_SCL_20m.jp2"))
    if not scl_files:
        logger.error(f"SCL band not found in {img_data_dir_20m}")
        return None
    scl_file = scl_files[0]
    with rasterio.open(scl_file) as scl_src:
        scl = scl_src.read(1)
        scl_profile = scl_src.profile
        # Resample SCL to 10m using nearest neighbor
        scl_10m = scl_src.read(
            1,
            out_shape=(
                scl_src.count,
                int(scl_src.height * scl_src.res[0] / meta['transform'][0]),
                int(scl_src.width * scl_src.res[1] / abs(meta['transform'][4]))
            ),
            resampling=Resampling.nearest
        )

    # 3. Create cloud mask (1=cloud, 0=clear)
    cloud_mask = np.isin(scl_10m, [8, 9, 10]).astype(np.uint8)

    # 4. Apply cloud mask to bands (set cloudy pixels to 0)
    band_arrays_masked = [np.where(cloud_mask == 1, 0, arr) for arr in band_arrays]

    # 5. Save stack and cloud mask
    stack = np.stack(band_arrays_masked)
    meta.update({"count": len(bands)})
    out_path = os.path.join(output_dir, os.path.basename(scene_path) + "_stack.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        for i in range(len(bands)):
            dst.write(stack[i], i + 1)
    logger.info(f"Wrote stacked bands to {out_path} (cloud-masked)")

    # Save cloud mask as GeoTIFF
    cloud_mask_path = os.path.join(output_dir, os.path.basename(scene_path) + "_cloudmask.tif")
    cloud_mask_meta = meta.copy()
    cloud_mask_meta.update({"count": 1, "dtype": rasterio.uint8})
    with rasterio.open(cloud_mask_path, "w", **cloud_mask_meta) as dst:
        dst.write(cloud_mask, 1)
    logger.info(f"Wrote cloud mask to {cloud_mask_path}")

    return out_path, cloud_mask_path

# --- Hansen Data Functions ---

def download_hansen_data(aoi_geojson, output_dir):
    """
    Download Hansen GFC data (treecover2000, lossyear) for AOI, merge tiles, clip.
    """
    try:
        import geopandas as gpd
        import requests
        from shapely.geometry import box
        import subprocess
        import glob
    except ImportError:
        logger.error("Required packages not installed. Install: pip install geopandas requests gdal")
        return None

    os.makedirs(output_dir, exist_ok=True)
    raw_tile_dir = os.path.join(output_dir, "raw_tiles")
    os.makedirs(raw_tile_dir, exist_ok=True)

    aoi = gpd.read_file(aoi_geojson)
    minx, miny, maxx, maxy = aoi.total_bounds

    # Ensure longitude is within -180 to 180 and latitude within -90 to 90
    minx = max(-180, min(180, minx))
    maxx = max(-180, min(180, maxx))
    miny = max(-90, min(90, miny))
    maxy = max(-90, min(90, maxy))

    # Determine the tile coordinates. Tiles are 10x10 degrees.
    # Longitude: Origin is 180W. Tiles are 000W, 010W, ..., 170W, 170E, ..., 000E
    # Latitude: Origin is 90N. Tiles are 90N, 80N, ..., 00N, 00S, ..., 80S
    
    # Calculate the bottom-left corner of the tiles
    # For longitude, tiles are named by their westernmost edge.
    # For latitude, tiles are named by their northernmost edge.
    start_lon_tile = int(np.floor(minx / 10.0)) * 10
    end_lon_tile = int(np.floor(maxx / 10.0)) * 10
    
    start_lat_tile = int(np.ceil(maxy / 10.0)) * 10 
    end_lat_tile = int(np.ceil(miny / 10.0)) * 10 # This will go towards south

    # Hansen data versioning - use a recent, fixed version for reproducibility
    # As of early 2024, GFC-2022-v1.10 is the latest. Check for updates if needed.
    # https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change
    hansen_year_str = "2023" # Data is GFC-YYYY-vX.Y - this refers to the last year of the period
    hansen_version = "1.11" # Example: "v1.11" corresponds to GFC 2000-2023
    
    base_url = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-{hansen_year_str}-v{hansen_version}/"
    logger.info(f"Using Hansen GFC base URL: {base_url}")

    # Ensure tqdm is available for the download_file function
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm package not found. Progress bars will not be shown. Install with: pip install tqdm")
        # Define a dummy tqdm if not found so download_file doesn't break
        class tqdm:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def update(self, n=1): pass


    required_tile_files = {'treecover2000': [], 'lossyear': [], 'datamask': []}
    all_downloads_successful = True

    # Iterate over longitude from West to East, and latitude from North to South
    for lon_base in range(start_lon_tile, end_lon_tile + 10, 10):
        for lat_base in range(start_lat_tile, end_lat_tile -10 , -10): # Iterate from North towards South
            # Format latitude string (e.g., 50N, 10S)
            lat_str = f"{abs(lat_base):02d}{'N' if lat_base >= 0 else 'S'}"
            # Format longitude string (e.g., 100W, 020E)
            lon_str = f"{abs(lon_base):03d}{'W' if lon_base < 0 else 'E'}" # Corrected lon_base >=0 to lon_base < 0 for W
            
            tile_id = f"{lat_str}_{lon_str}"
            # logger.debug(f"Processing tile ID: {tile_id} for lon_base: {lon_base}, lat_base: {lat_base}")

            # Define filenames for different Hansen products
            # Note: "treecover2000" is the baseline tree cover for the year 2000.
            # "lossyear" indicates the year of loss (0 for no loss, 1-23 for year 2001-2023 for v1.11).
            # "datamask" indicates areas with no data, water, etc. (0: No data, 1: Mapped land surface, 2: Water)
            
            filenames_data = {
                "treecover2000": f"Hansen_GFC-{hansen_year_str}-v{hansen_version}_treecover2000_{tile_id}.tif",
                "lossyear": f"Hansen_GFC-{hansen_year_str}-v{hansen_version}_lossyear_{tile_id}.tif",
                "datamask": f"Hansen_GFC-{hansen_year_str}-v{hansen_version}_datamask_{tile_id}.tif"
                # Other potentially useful layers: "gain", "first", "last"
            }

            for layer_name, filename in filenames_data.items():
                file_url = base_url + filename
                file_path = os.path.join(raw_tile_dir, filename)
                
                if not os.path.exists(file_path):
                    logger.info(f"Attempting to download {filename} for tile {tile_id}...")
                    if not download_file(file_url, file_path):
                        logger.warning(f"Failed to download {filename}. It might not exist or an error occurred.")
                        all_downloads_successful = False # Mark that at least one download failed
                        # Depending on policy, we might want to break or continue
                    else:
                        required_tile_files[layer_name].append(file_path)
                else:
                    logger.info(f"File {filename} already exists, skipping download.")
                    required_tile_files[layer_name].append(file_path)
    
    if not all_downloads_successful:
        logger.warning("One or more Hansen GFC tiles failed to download. Subsequent processing might be affected.")
    
    if not required_tile_files['treecover2000'] and not required_tile_files['lossyear']: # Check if any essential tiles were found/downloaded
        logger.error("No Hansen GFC treecover or lossyear tiles were successfully downloaded or found for the AOI.")
        return None, None, None # Return None for all expected outputs

    # --- Merging and Clipping ---
    # For each layer type (treecover, lossyear, datamask), merge the downloaded tiles
    # Then clip the merged raster to the AOI.
    
    merged_files = {}
    clipped_files = {}

    for layer_name in ['treecover2000', 'lossyear', 'datamask']:
        tiles_to_merge = required_tile_files[layer_name]
        if not tiles_to_merge:
            logger.warning(f"No tiles found/downloaded for layer: {layer_name}. Skipping merge and clip for this layer.")
            merged_files[layer_name] = None
            clipped_files[layer_name] = None
            continue

        # 1. Merge tiles for the current layer
        merged_vrt_path = os.path.join(output_dir, f"hansen_merged_{layer_name}.vrt")
        merged_tif_path = os.path.join(output_dir, f"hansen_merged_{layer_name}.tif")

        # Use gdalbuildvrt for merging (creates a virtual raster)
        cmd_buildvrt = ["gdalbuildvrt", merged_vrt_path] + tiles_to_merge
        try:
            logger.info(f"Merging {len(tiles_to_merge)} tiles for {layer_name} into {merged_vrt_path}...")
            subprocess.run(cmd_buildvrt, check=True, capture_output=True, text=True) # Added text=True for better error messages
            logger.info(f"Successfully created VRT: {merged_vrt_path}")
            merged_files[layer_name] = merged_vrt_path # Store VRT, can be used directly by GDAL
            
            # Optionally convert VRT to a single TIF if preferred (can be large)
            # logger.info(f"Converting VRT {merged_vrt_path} to TIF {merged_tif_path}...")
            # cmd_translate = ["gdal_translate", merged_vrt_path, merged_tif_path, "-co", "COMPRESS=LZW", "-co", "TILED=YES"]
            # subprocess.run(cmd_translate, check=True, capture_output=True, text=True)
            # logger.info(f"Successfully created merged TIF: {merged_tif_path}")
            # merged_files[layer_name] = merged_tif_path # If converted to TIF
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error merging {layer_name} tiles with gdalbuildvrt: {e.stderr}")
            # Clean up failed VRT
            if os.path.exists(merged_vrt_path): os.remove(merged_vrt_path)
            return None, None, None # Critical failure

        # 2. Clip the merged raster to the AOI
        clipped_path = os.path.join(output_dir, f"hansen_clipped_{layer_name}.tif")
        # Use gdalwarp for clipping. Ensure AOI CRS matches raster CRS or specify transformations.
        # Hansen data is typically EPSG:4326.
        # We need the bounds of the AOI in minx, miny, maxx, maxy format for gdalwarp's -te option.
        # The aoi_geojson file itself can also be used as a cutline.
        cmd_clip = [
            "gdalwarp",
            "-cutline", aoi_geojson,  # Use the GeoJSON file as the cutline
            "-crop_to_cutline",      # Crop the raster to the extent of the cutline
            "-dstalpha",             # Add an alpha band for no-data areas outside the cutline
            merged_files[layer_name], # Input merged raster (VRT or TIF)
            clipped_path,            # Output clipped raster
            "-overwrite",            # Overwrite if exists
            "-co", "COMPRESS=LZW",   # Apply compression
            "-co", "TILED=YES"       # Use tiled format for efficiency
        ]
        try:
            logger.info(f"Clipping {merged_files[layer_name]} to AOI {aoi_geojson} for {layer_name}...")
            subprocess.run(cmd_clip, check=True, capture_output=True, text=True)
            logger.info(f"Successfully clipped {layer_name} raster: {clipped_path}")
            clipped_files[layer_name] = clipped_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clipping {layer_name} raster: {e.stderr}")
            # Clean up failed clipped file
            if os.path.exists(clipped_path): os.remove(clipped_path)
            return None, None, None # Critical failure
        finally:
            # Clean up the merged VRT/TIF if it's different from the final clipped path and still exists
            if merged_files[layer_name] and merged_files[layer_name] != clipped_path and os.path.exists(merged_files[layer_name]):
                # Only remove VRT if we successfully created a TIF from it or if clipping was successful from VRT
                # If we decided to keep merged_files[layer_name] as a TIF, then it might be the same as clipped_path if AOI covers everything
                 if merged_files[layer_name].endswith(".vrt"): # Typically remove VRTs after use
                    logger.debug(f"Removing temporary merged VRT: {merged_files[layer_name]}")
                    os.remove(merged_files[layer_name])
                 # elif os.path.exists(clipped_path): # If we converted VRT to TIF and then clipped TIF
                      # os.remove(merged_files[layer_name])


    logger.info("Hansen GFC data download, merge, and clip process completed.")
    # Return paths to the clipped treecover2000, lossyear, and datamask files
    return clipped_files.get('treecover2000'), clipped_files.get('lossyear'), clipped_files.get('datamask')

# --- Sentinel-1 (SAR) Data Functions ---

# Section for Sentinel-1 removed as per user request to simplify the pipeline.
# download_sentinel1_data function removed.
# preprocess_sentinel1_scene function removed.

# --- Alignment and Labeling Functions ---

def align_rasters(reference_raster_path, raster_to_align_path, output_path, resampling_method="near"):
    """Aligns a raster to match the grid, extent, and CRS of a reference raster using GDAL."""
    try:
        logger.info(f"Aligning {os.path.basename(raster_to_align_path)} to {os.path.basename(reference_raster_path)}...")
        with rasterio.open(reference_raster_path) as ref_src:
            crs = ref_src.crs.to_string()
            bounds = ref_src.bounds
            res = ref_src.res
            width = ref_src.width
            height = ref_src.height

        cmd = [
            "gdalwarp",
            "-t_srs", crs,
            "-tr", str(res[0]), str(abs(res[1])), # Use positive resolution for -tr
            "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
            "-ts", str(width), str(height), # Explicitly set target size
            "-r", resampling_method,
            "-overwrite",
            raster_to_align_path,
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Aligned raster saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"GDAL alignment failed for {raster_to_align_path}: {e}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        return None
    except Exception as e:
        logger.error(f"Error during alignment: {e}")
        return None

def create_change_label(hansen_treecover_path, hansen_lossyear_path, target_year, output_path, reference_raster_path=None, tree_cover_threshold=30, hansen_datamask_path=None):
    """
    Creates a binary change label (1=loss, 0=no loss/not forest) aligned to reference.
    Optionally uses a Hansen datamask to exclude non-valid areas.
    """
    aligned_tc_path = hansen_treecover_path
    aligned_ly_path = hansen_lossyear_path
    aligned_dm_path = hansen_datamask_path
    temp_files = []

    if reference_raster_path:
        logger.info(f"Aligning Hansen data to reference: {os.path.basename(reference_raster_path)} for label creation.")
        aligned_tc_path_temp = os.path.join(os.path.dirname(output_path), f"temp_aligned_tc_{os.path.basename(output_path)}")
        aligned_ly_path_temp = os.path.join(os.path.dirname(output_path), f"temp_aligned_ly_{os.path.basename(output_path)}")
        
        aligned_tc_path = align_rasters(reference_raster_path, hansen_treecover_path, aligned_tc_path_temp, resampling_method="near")
        aligned_ly_path = align_rasters(reference_raster_path, hansen_lossyear_path, aligned_ly_path_temp, resampling_method="near")
        
        if not aligned_tc_path or not aligned_ly_path:
            logger.error("Failed to align Hansen treecover or lossyear data."); return None
        temp_files.extend([f for f in [aligned_tc_path, aligned_ly_path] if f not in [hansen_treecover_path, hansen_lossyear_path]])

        if hansen_datamask_path:
            aligned_dm_path_temp = os.path.join(os.path.dirname(output_path), f"temp_aligned_dm_{os.path.basename(output_path)}")
            aligned_dm_path = align_rasters(reference_raster_path, hansen_datamask_path, aligned_dm_path_temp, resampling_method="near")
            if not aligned_dm_path:
                logger.warning("Failed to align Hansen datamask. Proceeding without it.")
                aligned_dm_path = None # Ensure it's None if alignment fails
            elif aligned_dm_path != hansen_datamask_path: # It was a temporary file
                 temp_files.append(aligned_dm_path)

    try:
        with rasterio.open(aligned_tc_path) as tc_src, \
             rasterio.open(aligned_ly_path) as ly_src:
            
            treecover = tc_src.read(1)
            lossyear = ly_src.read(1)
            profile = tc_src.profile # Use profile from treecover, should be same grid as others

            # Initialize label based on forest cover and loss year
            is_forest = treecover >= tree_cover_threshold
            # Loss year: 1-X for 2001-200X. target_year is e.g. 2020. (target_year - 2000) gives the value for 2020 loss.
            is_loss_up_to_target_year = (lossyear > 0) & (lossyear <= (target_year - 2000))
            
            change_label_raw = (is_forest & is_loss_up_to_target_year).astype(rasterio.uint8)

            # Apply datamask if available and successfully aligned
            if aligned_dm_path and os.path.exists(aligned_dm_path):
                logger.info(f"Applying Hansen datamask: {aligned_dm_path}")
                with rasterio.open(aligned_dm_path) as dm_src:
                    datamask = dm_src.read(1)
                    # Hansen GFC datamask: 1 for mapped land surface, 0 for no data, 2 for water.
                    # We only want areas that are mapped land surface.
                    valid_data_mask = (datamask == 1)
                    change_label_final = change_label_raw * valid_data_mask
                    logger.info("Applied datamask to change label.")
            else:
                logger.info("No valid Hansen datamask applied to change label.")
                change_label_final = change_label_raw

            profile.update(dtype=rasterio.uint8, count=1, nodata=0) # Explicitly set nodata to 0 for labels
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(change_label_final, 1)
            logger.info(f"Change label created: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Error creating change label: {e}"); return None
    finally:
        for f in temp_files:
            if os.path.exists(f): os.remove(f)

# --- Main Preparation Function --- #

def prepare_data_for_ml(s2_scene_dirs, 
                        hansen_treecover_path, hansen_lossyear_path, hansen_datamask_path,
                        output_dir, s2_bands=["B02", "B03", "B04", "B08"]):
    """
    Prepares aligned Sentinel-2 stacks and change labels for ML training.
    Sentinel-1 processing has been removed.
    """
    prepared_s2_stacks_dir = os.path.join(output_dir, "s2_stacks")
    # prepared_s1_stacks_dir = os.path.join(output_dir, "s1_stacks") # Removed S1
    prepared_labels_dir = os.path.join(output_dir, "change_labels")
    os.makedirs(prepared_s2_stacks_dir, exist_ok=True)
    # os.makedirs(prepared_s1_stacks_dir, exist_ok=True) # Removed S1
    os.makedirs(prepared_labels_dir, exist_ok=True)

    processed_s2_scenes = {}
    # processed_s1_scenes = {} # Removed S1
    scene_dates = set()

    # 1. Process Sentinel-2 scenes
    logger.info("--- Processing Sentinel-2 Scenes ---")
    for scene_dir in s2_scene_dirs:
        scene_name = os.path.basename(scene_dir).replace(".SAFE", "")
        try:
            date_str = scene_name.split("_")[2].split("T")[0]
            scene_date = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception: logger.warning(f"Could not parse date from S2 {scene_name}. Skipping."); continue

        logger.info(f"Processing S2: {scene_name}")
        stacked_s2_path, cloud_mask_path = prepare_scene_bands(scene_dir, s2_bands, prepared_s2_stacks_dir)
        if stacked_s2_path:
            processed_s2_scenes[scene_date] = stacked_s2_path
            scene_dates.add(scene_date)
        else: logger.warning(f"Failed to process S2 bands for {scene_name}.")

    if not processed_s2_scenes:
        logger.error("No Sentinel-2 scenes processed successfully. Cannot proceed.")
        return None

    # 2. Process Sentinel-1 scenes (if requested) # Section Removed
    # if include_s1 and s1_scene_dirs:
    #    ... (S1 processing logic removed)

    # 3. Align S1 data to S2 grid and create labels
    logger.info("--- Creating Labels ---") # Simplified title
    sorted_dates = sorted(list(scene_dates))
    # aligned_s1_scenes = {} # Removed S1
    scene_labels = {}

    # Use the first S2 scene as the reference grid (still relevant for consistency if multiple S2 scenes processed, though labels are aligned per-scene)
    if not sorted_dates: # Should be caught by processed_s2_scenes check, but defensive
        logger.error("No dates found from processed S2 scenes.")
        return None
    ref_s2_date = min(processed_s2_scenes.keys()) #This is min date, so first S2 scene chronologically
    # reference_grid_path = processed_s2_scenes[ref_s2_date] # This was for S1 alignment, less critical now but good for a common grid concept.

    for scene_date in sorted_dates:
        scene_name_base = scene_date.strftime("%Y%m%d") # Base name for aligned files

        # Align S1 if available for this date (or nearest) # Section removed
        # if include_s1 and scene_date in processed_s1_scenes:
        #    ... (S1 alignment logic removed)

        # Create change label aligned to the S2 grid for this date
        if scene_date in processed_s2_scenes:
            s2_path = processed_s2_scenes[scene_date]
            label_output_path = os.path.join(prepared_labels_dir, f"{scene_name_base}_change_label.tif")
            created_label_path = create_change_label(
                hansen_treecover_path,
                hansen_lossyear_path,
                target_year=scene_date.year,
                output_path=label_output_path,
                reference_raster_path=s2_path, # Align label to this S2 scene
                hansen_datamask_path=hansen_datamask_path # Pass datamask path
            )
            if created_label_path:
                scene_labels[scene_date] = created_label_path
            else:
                logger.warning(f"Failed to create label for {scene_date}")

    logger.info(f"Prepared {len(processed_s2_scenes)} S2 stacks.")
    # logger.info(f"Prepared {len(aligned_s1_scenes)} aligned S1 stacks.") # Removed S1
    logger.info(f"Prepared {len(scene_labels)} change labels.")

    if not processed_s2_scenes or not scene_labels:
        logger.error("Failed to prepare necessary S2 stacks or labels.")
        return None

    return output_dir # Return base directory

# --- Main Execution Logic --- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare S2 & Hansen data for ML.") # Updated description
    parser.add_argument("--aoi", type=str, required=True, help="Path to GeoJSON AOI file.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory.")
    parser.add_argument("--s2_bands", nargs="+", default=["B02", "B03", "B04", "B08"])
    # parser.add_argument("--include_s1", action="store_true", help="Include Sentinel-1 data processing.") # Removed S1 arg
    # parser.add_argument("--s1_orbit", default="ASCENDING") # Removed S1 arg
    # parser.add_argument("--s1_polarization", default="VV VH") # Removed S1 arg
    args = parser.parse_args()

    raw_s2_dir = os.path.join(args.output_dir, "raw", "sentinel2")
    # raw_s1_dir = os.path.join(args.output_dir, "raw", "sentinel1") # Removed S1
    raw_hansen_dir = os.path.join(args.output_dir, "raw", "hansen")
    prepared_data_dir = os.path.join(args.output_dir, "prepared")

    # --- Download --- #
    logger.info("--- Starting Data Download ---")
    s2_scene_dirs = download_sentinel2_data(args.aoi, args.start_date, args.end_date, raw_s2_dir)
    hansen_paths = download_hansen_data(args.aoi, raw_hansen_dir)
    # s1_scene_dirs = [] # Removed S1
    # if args.include_s1: # Removed S1
        # s1_scene_dirs = download_sentinel1_data(args.aoi, args.start_date, args.end_date, raw_s1_dir, args.s1_orbit, args.s1_polarization)

    if not s2_scene_dirs or not hansen_paths or len(hansen_paths) < 3 or not all(hansen_paths):
        logger.error("S2 or Hansen data (treecover, lossyear, datamask) download failed or incomplete. Exiting.")
        sys.exit(1)
    # if args.include_s1 and not s1_scene_dirs: # Removed S1
        # logger.warning("S1 download specified but no scenes found/downloaded.")

    # --- Prepare --- #
    logger.info("--- Starting Data Preparation ---")
    hansen_tc_path, hansen_ly_path, hansen_dm_path = hansen_paths
    result_dir = prepare_data_for_ml(
        s2_scene_dirs,
        # s1_scene_dirs, # Removed S1
        hansen_tc_path, 
        hansen_ly_path,
        hansen_dm_path, # Pass the datamask path
        prepared_data_dir,
        s2_bands=args.s2_bands
        # include_s1=args.include_s1 # Removed S1
    )

    if result_dir:
        logger.info(f"Data preparation completed. Results in {result_dir}")
    else:
        logger.error("Data preparation failed.")
        sys.exit(1)

if __name__ == "__main__":
    # Example Usage:
    # python ml/utils/data_preparation.py --aoi path/to/aoi.geojson --start_date 2020-01-01 --end_date 2021-12-31 --output_dir /data/ml_project --include_s1
    main()

