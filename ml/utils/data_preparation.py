import os
import sys
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_ndvi(red, nir):
    """Calculate Normalized Difference Vegetation Index."""
    return (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero

def download_sentinel2_data(aoi_geojson, start_date, end_date, output_dir):
    """
    Download Sentinel-2 data for a given area of interest and time range.
    
    Args:
        aoi_geojson (str): Path to GeoJSON file defining the area of interest
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        output_dir (str): Directory to save downloaded data
    
    Returns:
        list: Paths to downloaded Sentinel-2 scenes
    """
    try:
        # Try to import sentinelsat
        from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    except ImportError:
        logger.error("sentinelsat package not installed. Install with: pip install sentinelsat")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to Copernicus Open Access Hub
    # Note: Users need to register at https://scihub.copernicus.eu/dhus/#/self-registration
    # and replace 'user' and 'password' with their credentials
    api = SentinelAPI('user', 'password', 'https://scihub.copernicus.eu/dhus')
    
    # Read area of interest from GeoJSON file
    footprint = geojson_to_wkt(read_geojson(aoi_geojson))
    
    # Search for Sentinel-2 scenes
    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname='Sentinel-2',
        producttype='S2MSI2A',  # Level-2A product (atmospherically corrected)
        cloudcoverpercentage=(0, 20)  # Max 20% cloud cover
    )
    
    if not products:
        logger.warning(f"No Sentinel-2 scenes found for the given parameters")
        return []
    
    logger.info(f"Found {len(products)} Sentinel-2 scenes")
    
    # Download scenes
    downloaded_scenes = []
    for product_id, product_info in products.items():
        scene_dir = os.path.join(output_dir, product_info['title'])
        
        # Skip if already downloaded
        if os.path.exists(scene_dir):
            logger.info(f"Scene {product_info['title']} already downloaded")
            downloaded_scenes.append(scene_dir)
            continue
        
        # Download scene
        logger.info(f"Downloading {product_info['title']}...")
        api.download(product_id, directory_path=output_dir)
        
        # Extract bands from downloaded ZIP file
        import zipfile
        zip_path = os.path.join(output_dir, f"{product_info['title']}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(scene_dir)
            
            # Clean up ZIP file
            os.remove(zip_path)
            
            downloaded_scenes.append(scene_dir)
    
    return downloaded_scenes

def download_hansen_data(aoi_geojson, output_dir):
    """
    Download Hansen Global Forest Change data for a given area of interest.
    
    Args:
        aoi_geojson (str): Path to GeoJSON file defining the area of interest
        output_dir (str): Directory to save downloaded data
    
    Returns:
        str: Path to downloaded Hansen data
    """
    try:
        # Try to import required packages
        import geopandas as gpd
        import requests
        from shapely.geometry import box
    except ImportError:
        logger.error("Required packages not installed. Install with: pip install geopandas requests")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read area of interest from GeoJSON file
    aoi = gpd.read_file(aoi_geojson)
    
    # Get bounding box
    minx, miny, maxx, maxy = aoi.total_bounds
    
    # Hansen data is available in 10x10 degree tiles
    # Calculate which tiles we need
    min_lon_tile = int(minx // 10) * 10
    max_lon_tile = int(maxx // 10) * 10
    min_lat_tile = int(miny // 10) * 10
    max_lat_tile = int(maxy // 10) * 10
    
    # Hansen data URL template
    # Latest year available is typically the previous year
    current_year = datetime.now().year
    hansen_year = current_year - 1
    hansen_url_template = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-{hansen_year}-v1.8/Hansen_GFC-{hansen_year}-v1.8_treecover2000_{{}}_{{}}_{{}}.tif"
    
    # Download tiles
    downloaded_tiles = []
    for lon in range(min_lon_tile, max_lon_tile + 10, 10):
        for lat in range(min_lat_tile, max_lat_tile + 10, 10):
            # Adjust latitude for Hansen data format
            hansen_lat = lat
            if hansen_lat < 0:
                hansen_lat = f"S{abs(hansen_lat):02d}"
            else:
                hansen_lat = f"N{hansen_lat:02d}"
            
            # Adjust longitude for Hansen data format
            hansen_lon = lon
            if hansen_lon < 0:
                hansen_lon = f"W{abs(hansen_lon):03d}"
            else:
                hansen_lon = f"E{hansen_lon:03d}"
            
            # Download tree cover for 2000
            treecover_url = hansen_url_template.format("treecover2000", hansen_lat, hansen_lon)
            treecover_path = os.path.join(output_dir, f"treecover2000_{hansen_lat}_{hansen_lon}.tif")
            
            # Download loss year data
            lossyear_url = hansen_url_template.format("lossyear", hansen_lat, hansen_lon)
            lossyear_path = os.path.join(output_dir, f"lossyear_{hansen_lat}_{hansen_lon}.tif")
            
            # Download files if they don't exist
            for url, path in [(treecover_url, treecover_path), (lossyear_url, lossyear_path)]:
                if not os.path.exists(path):
                    logger.info(f"Downloading {url}...")
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logger.info(f"Downloaded {path}")
                    else:
                        logger.warning(f"Failed to download {url}")
                else:
                    logger.info(f"File {path} already exists")
            
            downloaded_tiles.append((treecover_path, lossyear_path))
    
    # Merge tiles if more than one
    if len(downloaded_tiles) > 1:
        # Merge tree cover tiles
        merged_treecover_path = os.path.join(output_dir, "merged_treecover2000.tif")
        merged_lossyear_path = os.path.join(output_dir, "merged_lossyear.tif")
        
        # Use gdal_merge.py to merge tiles
        import subprocess
        
        # Merge tree cover tiles
        treecover_tiles = [tile[0] for tile in downloaded_tiles]
        cmd = ["gdal_merge.py", "-o", merged_treecover_path] + treecover_tiles
        subprocess.run(cmd, check=True)
        
        # Merge loss year tiles
        lossyear_tiles = [tile[1] for tile in downloaded_tiles]
        cmd = ["gdal_merge.py", "-o", merged_lossyear_path] + lossyear_tiles
        subprocess.run(cmd, check=True)
        
        # Clip to AOI
        clipped_treecover_path = os.path.join(output_dir, "treecover2000.tif")
        clipped_lossyear_path = os.path.join(output_dir, "lossyear.tif")
        
        # Use gdalwarp to clip to AOI
        cmd = ["gdalwarp", "-cutline", aoi_geojson, "-crop_to_cutline", merged_treecover_path, clipped_treecover_path]
        subprocess.run(cmd, check=True)
        
        cmd = ["gdalwarp", "-cutline", aoi_geojson, "-crop_to_cutline", merged_lossyear_path, clipped_lossyear_path]
        subprocess.run(cmd, check=True)
        
        return (clipped_treecover_path, clipped_lossyear_path)
    elif len(downloaded_tiles) == 1:
        # Just clip the single tile to AOI
        clipped_treecover_path = os.path.join(output_dir, "treecover2000.tif")
        clipped_lossyear_path = os.path.join(output_dir, "lossyear.tif")
        
        # Use gdalwarp to clip to AOI
        import subprocess
        
        cmd = ["gdalwarp", "-cutline", aoi_geojson, "-crop_to_cutline", downloaded_tiles[0][0], clipped_treecover_path]
        subprocess.run(cmd, check=True)
        
        cmd = ["gdalwarp", "-cutline", aoi_geojson, "-crop_to_cutline", downloaded_tiles[0][1], clipped_lossyear_path]
        subprocess.run(cmd, check=True)
        
        return (clipped_treecover_path, clipped_lossyear_path)
    else:
        logger.warning("No Hansen data downloaded")
        return None

def prepare_training_data(sentinel_scenes, hansen_data, output_dir):
    """
    Prepare training data by matching Sentinel-2 scenes with Hansen data.
    
    Args:
        sentinel_scenes (list): Paths to Sentinel-2 scenes
        hansen_data (tuple): Paths to Hansen tree cover and loss year data
        output_dir (str): Directory to save prepared data
    
    Returns:
        str: Path to prepared data directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for Sentinel-2 and Hansen data
    sentinel_dir = os.path.join(output_dir, 'sentinel')
    hansen_dir = os.path.join(output_dir, 'hansen')
    os.makedirs(sentinel_dir, exist_ok=True)
    os.makedirs(hansen_dir, exist_ok=True)
    
    # Process each Sentinel-2 scene
    for scene_path in sentinel_scenes:
        scene_id = os.path.basename(scene_path)
        scene_output_dir = os.path.join(sentinel_dir, scene_id)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        # Extract required bands (B02, B03, B04, B08)
        # The exact paths depend on the Sentinel-2 product structure
        # This is a simplified example
        for band in ['B02', 'B03', 'B04', 'B08']:
            # Find band file in scene directory (recursive search)
            import glob
            band_files = glob.glob(os.path.join(scene_path, f"**/*{band}*.jp2"), recursive=True)
            
            if band_files:
                band_file = band_files[0]
                output_file = os.path.join(scene_output_dir, f"{band}.tif")
                
                # Convert JP2 to GeoTIFF
                import subprocess
                cmd = ["gdal_translate", band_file, output_file]
                subprocess.run(cmd, check=True)
                
                logger.info(f"Extracted {band} from {scene_id}")
            else:
                logger.warning(f"Could not find {band} in {scene_id}")
        
        # Create forest change label from Hansen data
        treecover_path, lossyear_path = hansen_data
        
        # Create a binary forest change mask for this scene
        # 1: forest loss, 0: no change
        forest_change_path = os.path.join(hansen_dir, f"{scene_id}_forest_change.tif")
        
        # Get scene date
        # This is a simplified example - in reality, you would extract the date from scene metadata
        scene_date = datetime.now()  # Placeholder
        scene_year = scene_date.year
        
        # Create forest change mask
        with rasterio.open(treecover_path) as treecover_src, rasterio.open(lossyear_path) as lossyear_src:
            treecover = treecover_src.read(1)
            lossyear = lossyear_src.read(1)
            
            # Forest is defined as areas with tree cover > 30%
            forest_mask = treecover > 30
            
            # Forest loss is defined as areas that were forest and lost between 2000 and scene_year
            # Loss year is encoded as year - 2000, so 1 = 2001, 2 = 2002, etc.
            forest_loss = forest_mask & (lossyear > 0) & (lossyear <= scene_year - 2000)
            
            # Save forest change mask
            with rasterio.open(
                forest_change_path,
                'w',
                driver='GTiff',
                height=forest_loss.shape[0],
                width=forest_loss.shape[1],
                count=1,
                dtype=rasterio.uint8,
                crs=treecover_src.crs,
                transform=treecover_src.transform
            ) as dst:
                dst.write(forest_loss.astype(rasterio.uint8), 1)
            
            logger.info(f"Created forest change mask for {scene_id}")
    
    return output_dir

def main():
    """Main function to download and prepare data for forest change detection."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download and prepare data for forest change detection')
    parser.add_argument('--aoi', type=str, required=True, help='Path to GeoJSON file defining the area of interest')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in format YYYY-MM-DD')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save data')
    args = parser.parse_args()
    
    # Download Sentinel-2 data
    sentinel_scenes = download_sentinel2_data(args.aoi, args.start_date, args.end_date, 
                                             os.path.join(args.output_dir, 'raw', 'sentinel'))
    
    # Download Hansen data
    hansen_data = download_hansen_data(args.aoi, os.path.join(args.output_dir, 'raw', 'hansen'))
    
    # Prepare training data
    if sentinel_scenes and hansen_data:
        prepared_data_dir = prepare_training_data(sentinel_scenes, hansen_data, 
                                                 os.path.join(args.output_dir, 'prepared'))
        logger.info(f"Data preparation completed. Results saved to {prepared_data_dir}")
    else:
        logger.error("Data download failed. Cannot prepare training data.")

if __name__ == "__main__":
    main()
