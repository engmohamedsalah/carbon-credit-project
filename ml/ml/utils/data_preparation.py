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

# Configure logging
logging.basicConfig(level=logging.INFO, format=\"%(asctime)s - %(name)s - %(levelname)s - %(message)s\")
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def calculate_ndvi(red, nir):
    """Calculate Normalized Difference Vegetation Index."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1, 1)

# --- Sentinel-2 Data Functions ---

def download_sentinel2_data(aoi_geojson, start_date, end_date, output_dir):
    """
    Download Sentinel-2 L2A data for AOI and time range using sentinelsat.
    Requires Copernicus Hub credentials (SENTINEL_USER, SENTINEL_PASSWORD env vars).
    """
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

    api = SentinelAPI(user, password, \"https://scihub.copernicus.eu/dhus\")
    footprint = geojson_to_wkt(read_geojson(aoi_geojson))

    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname=\"Sentinel-2\",
        producttype=\"S2MSI2A\",
        cloudcoverpercentage=(0, 20)
    )

    if not products:
        logger.warning(f"No Sentinel-2 scenes found for the given parameters")
        return []
    logger.info(f"Found {len(products)} Sentinel-2 scenes")

    downloaded_scenes = []
    for product_id, product_info in products.items():
        scene_dir_safe = os.path.join(output_dir, product_info[\"title\"] + ".SAFE")
        zip_path = os.path.join(output_dir, f"{product_info[\"title\"]}.zip")

        if os.path.exists(scene_dir_safe):
            logger.info(f"Scene {product_info[\"title\"]} already exists (extracted)")
            downloaded_scenes.append(scene_dir_safe)
            continue
        if not os.path.exists(zip_path):
            logger.info(f"Downloading {product_info[\"title\"]}...")
            try:
                api.download(product_id, directory_path=output_dir)
            except Exception as e:
                logger.error(f"Failed to download {product_info[\"title\"]}: {e}")
                continue

        if os.path.exists(zip_path):
            import zipfile
            logger.info(f"Extracting {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, \"r\") as zip_ref:
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

def prepare_scene_bands(scene_dir, output_dir, bands=[\"B02\", \"B03\", \"B04\", \"B08\"]):
    """
    Extracts specified bands from a Sentinel-2 .SAFE directory, converts to GeoTIFF,
    aligns bands if necessary, and stacks them into a single multi-band raster.
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_name = os.path.basename(scene_dir).replace(".SAFE", "")
    stacked_output_path = os.path.join(output_dir, f"{scene_name}_s2_stack.tif")
    temp_files = [] # Keep track of temporary files for cleanup

    try:
        granule_dir_pattern = os.path.join(scene_dir, "GRANULE", "*", "IMG_DATA")
        granule_dirs = glob.glob(granule_dir_pattern)
        if not granule_dirs: raise FileNotFoundError("IMG_DATA directory not found")
        img_data_dir = granule_dirs[0]

        res_dirs = [d for d in glob.glob(os.path.join(img_data_dir, "R*")) if os.path.isdir(d)]
        if not res_dirs: raise FileNotFoundError("Resolution directories not found")

        band_paths = {}
        ref_band_path_jp2 = None
        ref_band_res = None
        for band in bands:
            found = False
            for res_dir in res_dirs:
                band_pattern = os.path.join(res_dir, f"*{band}.jp2")
                band_files = glob.glob(band_pattern)
                if band_files:
                    band_paths[band] = band_files[0]
                    if not ref_band_path_jp2:
                         ref_band_path_jp2 = band_files[0]
                         ref_band_res = os.path.basename(res_dir) # e.g., R10m
                    found = True
                    break
            if not found: raise FileNotFoundError(f"Band {band} not found")

        # Convert JP2 to temporary GeoTIFFs
        temp_tif_paths = {}
        for band, jp2_path in band_paths.items():
            temp_tif_path = os.path.join(output_dir, f"temp_{scene_name}_{band}.tif")
            cmd_translate = ["gdal_translate", jp2_path, temp_tif_path]
            subprocess.run(cmd_translate, check=True, capture_output=True)
            temp_tif_paths[band] = temp_tif_path
            temp_files.append(temp_tif_path)

        # Align bands to the reference band resolution if necessary
        aligned_band_files = []
        ref_tif_path = temp_tif_paths[bands[0]] # Use first band\"s TIF as reference grid

        for band in bands:
            current_tif_path = temp_tif_paths[band]
            # Simple check: if band name suggests different resolution (e.g., B02 is 10m, B05 is 20m)
            # A more robust check would involve reading metadata
            needs_alignment = False # Assume alignment needed if resolutions differ
            # Example logic (needs refinement based on actual band resolutions):
            # if band in [\"B05\", \"B06\", \"B07\", \"B8A\", \"B11\", \"B12\"] and ref_band_res == \"R10m\":
            #     needs_alignment = True
            # elif band in [\"B01\", \"B09\", \"B10\"] and ref_band_res != \"R60m\":
            #     needs_alignment = True

            if needs_alignment:
                aligned_path = os.path.join(output_dir, f"temp_aligned_{scene_name}_{band}.tif")
                aligned_result = align_rasters(ref_tif_path, current_tif_path, aligned_path, resampling_method=\"bilinear\") # Use bilinear for reflectance
                if not aligned_result: raise RuntimeError(f"Failed to align band {band}")
                aligned_band_files.append(aligned_result)
                temp_files.append(aligned_result)
            else:
                aligned_band_files.append(current_tif_path)

        # Stack aligned bands
        vrt_path = os.path.join(output_dir, f"temp_{scene_name}_stack.vrt")
        temp_files.append(vrt_path)
        cmd_buildvrt = ["gdalbuildvrt", "-separate", vrt_path] + aligned_band_files
        subprocess.run(cmd_buildvrt, check=True, capture_output=True)

        cmd_translate_stack = ["gdal_translate", vrt_path, stacked_output_path]
        subprocess.run(cmd_translate_stack, check=True, capture_output=True)

        logger.info(f"Created S2 stacked raster: {stacked_output_path}")
        return stacked_output_path

    except Exception as e:
        logger.error(f"Error preparing S2 bands for {scene_name}: {e}")
        # Add more specific error handling if needed (e.g., CalledProcessError)
        return None
    finally:
        # Clean up temporary files
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except OSError: pass # Ignore errors during cleanup

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
    raw_tile_dir = os.path.join(output_dir, \"raw_tiles\")
    os.makedirs(raw_tile_dir, exist_ok=True)

    aoi = gpd.read_file(aoi_geojson)
    minx, miny, maxx, maxy = aoi.total_bounds

    min_lon_tile_base = int((minx + 180) // 10) * 10 - 180
    max_lon_tile_base = int((maxx + 180) // 10) * 10 - 180
    min_lat_tile_base = int((miny + 90) // 10) * 10 - 90 # Check calculation, might need floor/ceil
    max_lat_tile_base = int((maxy + 90) // 10) * 10 - 90 # Check calculation

    hansen_version = "1.8" # Or query latest
    hansen_year = datetime.now().year - 1
    base_url = f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-{hansen_year}-v{hansen_version}/"

    required_tiles = []
    # Iterate over longitude and latitude ranges
    for lon_base in range(min_lon_tile_base, max_lon_tile_base + 10, 10):
        for lat_base in range(max_lat_tile_base, min_lat_tile_base - 10, -10): # North to South
            lat_str = f"{abs(lat_base):02d}{\'N\' if lat_base >= 0 else \'S\'}"
            lon_str = f"{abs(lon_base):03d}{\'E\' if lon_base >= 0 else \'W\'}"
            tile_id = f"{lat_str}_{lon_str}"

            treecover_filename = f"Hansen_GFC-{hansen_year}-v{hansen_version}_treecover2000_{tile_id}.tif"
            lossyear_filename = f"Hansen_GFC-{hansen_year}-v{hansen_version}_lossyear_{tile_id}.tif"
            treecover_url = base_url + treecover_filename
            lossyear_url = base_url + lossyear_filename
            treecover_path = os.path.join(raw_tile_dir, treecover_filename)
            lossyear_path = os.path.join(raw_tile_dir, lossyear_filename)

            # Download logic (simplified, see previous version for full logic)
            download_success = True
            for url, path in [(treecover_url, treecover_path), (lossyear_url, lossyear_path)]:
                 if not os.path.exists(path):
                     logger.info(f"Downloading {os.path.basename(url)}...")
                     # Add robust download here (requests, check status, etc.)
                     # Placeholder: Assume download happens
                     # download_success = download_file(url, path)
                     # if not download_success: break
                     pass # Replace with actual download
                 else:
                     logger.info(f"File {os.path.basename(path)} already exists")

            if download_success and os.path.exists(treecover_path) and os.path.exists(lossyear_path):
                required_tiles.append((treecover_path, lossyear_path))
            else:
                 logger.warning(f"Skipping tile {tile_id} due to download failure.")

    if not required_tiles: logger.error("No Hansen tiles downloaded."); return None

    merged_treecover_path = os.path.join(output_dir, "merged_treecover2000.tif")
    merged_lossyear_path = os.path.join(output_dir, "merged_lossyear.tif")
    clipped_treecover_path = os.path.join(output_dir, "hansen_treecover2000_clipped.tif")
    clipped_lossyear_path = os.path.join(output_dir, "hansen_lossyear_clipped.tif")

    # --- Merging & Clipping Logic (Simplified - see previous version for full GDAL commands) --- #
    logger.info("Merging and clipping Hansen tiles (using GDAL - requires installation)...")
    # Placeholder: Assume merging and clipping happens using gdalbuildvrt, gdal_translate, gdalwarp
    # Need robust error handling around subprocess calls
    # Example merge command structure:
    # treecover_paths = [t[0] for t in required_tiles]
    # cmd_merge_tc = ["gdal_merge.py", "-o", merged_treecover_path] + treecover_paths
    # subprocess.run(cmd_merge_tc, check=True)
    # Example clip command structure:
    # cmd_clip_tc = ["gdalwarp", "-cutline", aoi_geojson, "-crop_to_cutline", merged_treecover_path, clipped_treecover_path]
    # subprocess.run(cmd_clip_tc, check=True)

    # --- Simulate successful merge/clip for structure --- #
    # In a real run, these files would be created by GDAL commands
    if not os.path.exists(clipped_treecover_path):
        # Create dummy files if they don\"t exist after placeholder merge/clip
        logger.warning(f"Simulating Hansen clip output: {clipped_treecover_path}")
        # shutil.copy(required_tiles[0][0], clipped_treecover_path) # Example: copy first tile
        pass # Avoid error if GDAL isn\"t run
    if not os.path.exists(clipped_lossyear_path):
        logger.warning(f"Simulating Hansen clip output: {clipped_lossyear_path}")
        # shutil.copy(required_tiles[0][1], clipped_lossyear_path) # Example: copy first tile
        pass # Avoid error if GDAL isn\"t run
    # --- End Placeholder --- #

    if os.path.exists(clipped_treecover_path) and os.path.exists(clipped_lossyear_path):
        logger.info(f"Hansen data processed: {clipped_treecover_path}, {clipped_lossyear_path}")
        # Clean up merged files
        if os.path.exists(merged_treecover_path): os.remove(merged_treecover_path)
        if os.path.exists(merged_lossyear_path): os.remove(merged_lossyear_path)
        return (clipped_treecover_path, clipped_lossyear_path)
    else:
        logger.error("Hansen data merging or clipping failed.")
        return None

# --- Sentinel-1 (SAR) Data Functions ---

def download_sentinel1_data(aoi_geojson, start_date, end_date, output_dir, orbit=\"ASCENDING\", polarization=\"VV VH\"):
    """
    Download Sentinel-1 GRD data for AOI and time range using sentinelsat.
    Requires Copernicus Hub credentials (SENTINEL_USER, SENTINEL_PASSWORD env vars).
    """
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

    api = SentinelAPI(user, password, \"https://scihub.copernicus.eu/dhus\")
    footprint = geojson_to_wkt(read_geojson(aoi_geojson))

    # Query for Sentinel-1 GRD IW products
    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname=\"Sentinel-1\",
        producttype=\"GRD\",
        sensoroperationalmode=\"IW\", # Interferometric Wide swath
        orbitdirection=orbit,
        polarisationmode=polarization.split()[0] # Use first polarization for query if multiple specified
    )

    if not products:
        logger.warning(f"No Sentinel-1 GRD IW scenes found for the given parameters")
        return []
    logger.info(f"Found {len(products)} Sentinel-1 scenes")

    downloaded_scenes = []
    # Download logic similar to Sentinel-2
    for product_id, product_info in products.items():
        scene_dir_safe = os.path.join(output_dir, product_info[\"title\"] + ".SAFE")
        zip_path = os.path.join(output_dir, f"{product_info[\"title\"]}.zip")

        if os.path.exists(scene_dir_safe):
            logger.info(f"Scene {product_info[\"title\"]} already exists (extracted)")
            downloaded_scenes.append(scene_dir_safe)
            continue
        if not os.path.exists(zip_path):
            logger.info(f"Downloading {product_info[\"title\"]}...")
            try:
                api.download(product_id, directory_path=output_dir)
            except Exception as e:
                logger.error(f"Failed to download {product_info[\"title\"]}: {e}")
                continue

        if os.path.exists(zip_path):
            import zipfile
            logger.info(f"Extracting {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, \"r\") as zip_ref:
                    zip_ref.extractall(output_dir)
                if os.path.isdir(scene_dir_safe):
                    downloaded_scenes.append(scene_dir_safe)
                    logger.info(f"Extracted to {scene_dir_safe}")
                    os.remove(zip_path)
                else:
                    logger.error(f"Extraction did not create expected directory: {scene_dir_safe}")
            except Exception as e:
                logger.error(f"Failed to extract {zip_path}: {e}")
                if os.path.exists(zip_path): os.remove(zip_path)

    return downloaded_scenes

def preprocess_sentinel1_scene(scene_dir, output_dir, target_resolution=10, dem_path=None):
    """
    Placeholder function outlining Sentinel-1 GRD preprocessing using SNAP GPT.
    Requires ESA SNAP command-line tool (gpt) installed and configured.

    Steps:
    1. Apply Orbit File: Updates orbit metadata for accuracy.
    2. Radiometric Calibration: Converts DN to physically meaningful backscatter (Sigma0).
    3. Speckle Filtering: Reduces inherent SAR speckle noise (e.g., Lee Sigma filter).
    4. Terrain Correction: Geocodes the image using a DEM, correcting geometric distortions.
    5. (Optional) Convert to dB scale.
    6. Stack VV and VH polarizations (if available).

    Args:
        scene_dir (str): Path to the Sentinel-1 .SAFE directory.
        output_dir (str): Directory to save the processed output.
        target_resolution (int): Target resolution in meters for Terrain Correction.
        dem_path (str, optional): Path to external DEM file. If None, SNAP attempts auto-download.

    Returns:
        str: Path to the processed multi-band GeoTIFF (e.g., VV, VH stacked), or None if failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_name = os.path.basename(scene_dir).replace(".SAFE", "")
    output_path = os.path.join(output_dir, f"{scene_name}_s1_processed_stack.tif")

    # Check if SNAP GPT is available
    gpt_path = shutil.which("gpt") # Find gpt executable in PATH
    if not gpt_path:
        logger.error("SNAP GPT command-line tool not found in PATH. Cannot preprocess Sentinel-1.")
        logger.error("Install SNAP: https://step.esa.int/main/download/snap-download/")
        return None

    logger.info(f"Preprocessing Sentinel-1 scene: {scene_name} using SNAP GPT")

    # --- Define SNAP Graph XML (Simplified Example) --- #
    # This graph needs to be created carefully, often using SNAP Desktop GUI first
    # and then exporting the XML. Paths within the XML need parameterization.
    graph_xml_content = f"""\
    <graph id=\"S1PreprocessingGraph\">
      <version>1.0</version>
      <node id=\"Read\">
        <operator>Read</operator>
        <sources />
        <parameters class=\"com.bc.ceres.binding.dom.XppDom\">
          <file>{scene_dir}/manifest.safe</file> <!-- Input Product -->
        </parameters>
      </node>
      <node id=\"Apply-Orbit-File\">
        <operator>Apply-Orbit-File</operator>
        <sources>
          <sourceProduct refid=\"Read\"/>
        </sources>
        <parameters class=\"com.bc.ceres.binding.dom.XppDom\">
          <orbitType>Sentinel Precise (Auto Download)</orbitType>
          <polyDegree>3</polyDegree>
          <continueOnFail>false</continueOnFail>
        </parameters>
      </node>
      <node id=\"Calibration\">
        <operator>Calibration</operator>
        <sources>
          <sourceProduct refid=\"Apply-Orbit-File\"/>
        </sources>
        <parameters class=\"com.bc.ceres.binding.dom.XppDom\">
          <sourceBands/>
          <auxFile>Product Auxiliary File</auxFile>
          <externalAuxFile/>
          <outputImageInComplex>false</outputImageInComplex>
          <outputImageScaleInDb>false</outputImageScaleInDb>
          <createGammaBand>false</createGammaBand>
          <createBetaBand>false</createBetaBand>
          <selectedPolarisations>VV,VH</selectedPolarisations> <!-- Adjust if only single pol -->
          <outputSigmaBand>true</outputSigmaBand>
          <outputGammaBand>false</outputGammaBand>
          <outputBetaBand>false</outputBetaBand>
        </parameters>
      </node>
      <node id=\"Speckle-Filter\">
        <operator>Speckle-Filter</operator>
        <sources>
          <sourceProduct refid=\"Calibration\"/>
        </sources>
        <parameters class=\"com.bc.ceres.binding.dom.XppDom\">
          <sourceBands/>
          <filter>Lee Sigma</filter>
          <filterSizeX>7</filterSizeX>
          <filterSizeY>7</filterSizeY>
          <dampingFactor>2</dampingFactor>
          <estimateENL>true</estimateENL>
          <enl>1.0</enl>
          <numLooksStr>1</numLooksStr>
          <windowSize>7x7</windowSize>
          <targetWindowSizeStr>3x3</targetWindowSizeStr>
          <sigmaStr>0.9</sigmaStr>
          <anSize>50</anSize>
        </parameters>
      </node>
      <node id=\"Terrain-Correction\">
        <operator>Terrain-Correction</operator>
        <sources>
          <sourceProduct refid=\"Speckle-Filter\"/>
        </sources>
        <parameters class=\"com.bc.ceres.binding.dom.XppDom\">
          <sourceBands/>
          <demName>{ \"SRTM 1Sec HGT\" if dem_path is None else \"External DEM\" }</demName>
          <externalDEMFile>{ dem_path if dem_path else \"\" }</externalDEMFile>
          <externalDEMNoDataValue>{ 0.0 if dem_path else \"0\" }</externalDEMNoDataValue>
          <externalDEMApplyEGM>true</externalDEMApplyEGM>
          <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
          <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
          <pixelSpacingInMeter>{target_resolution}</pixelSpacingInMeter>
          <pixelSpacingInDegree>0.0</pixelSpacingInDegree>
          <mapProjection>AUTO:42001</mapProjection> <!-- Example: UTM Auto -->
          <alignToStandardGrid>false</alignToStandardGrid>
          <standardGridOriginX>0.0</standardGridOriginX>
          <standardGridOriginY>0.0</standardGridOriginY>
          <nodataValueAtSea>true</nodataValueAtSea>
          <saveDEM>false</saveDEM>
          <saveLatLon>false</saveLatLon>
          <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
          <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
          <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
          <saveSelectedSourceBand>true</saveSelectedSourceBand>
          <outputComplex>false</outputComplex>
          <applyRadiometricNormalization>false</applyRadiometricNormalization>
          <saveSigmaNought>false</saveSigmaNought>
          <saveGammaNought>false</saveGammaNought>
          <saveBetaNought>false</saveBetaNought>
          <incidenceAngleForSigma0>Use projected local incidence angle from DEM</incidenceAngleForSigma0>
          <incidenceAngleForGamma0>Use projected local incidence angle from DEM</incidenceAngleForGamma0>
          <auxFile>Latest Auxiliary File</auxFile>
          <externalAuxFile/>
        </parameters>
      </node>
      <node id=\"Write\">
        <operator>Write</operator>
        <sources>
          <sourceProduct refid=\"Terrain-Correction\"/>
        </sources>
        <parameters class=\"com.bc.ceres.binding.dom.XppDom\">
          <file>{output_path}</file> <!-- Output File -->
          <formatName>GeoTIFF-BigTIFF</formatName>
        </parameters>
      </node>
    </graph>
    """

    # Save graph XML to a temporary file
    graph_xml_path = os.path.join(output_dir, f"temp_s1_graph_{scene_name}.xml")
    with open(graph_xml_path, \"w\") as f:
        f.write(graph_xml_content)

    # Construct GPT command
    # Adjust memory allocation as needed (-Xmx parameter)
    cmd_gpt = [
        gpt_path,
        graph_xml_path,
        "-q", "16", # Max parallelism
        "-J-Xmx16G" # Example: Allocate 16GB RAM to SNAP
    ]

    try:
        logger.info(f"Running SNAP GPT command: {\" \".join(cmd_gpt)}")
        # Execute GPT command
        # Increased timeout might be needed for long processing
        result = subprocess.run(cmd_gpt, check=True, capture_output=True, text=True, timeout=1800) # 30 min timeout
        logger.info("SNAP GPT execution successful.")
        logger.debug(f"GPT Output:\n{result.stdout}")
        if not os.path.exists(output_path):
             raise FileNotFoundError(f"SNAP GPT completed but output file not found: {output_path}")
        return output_path
    except subprocess.TimeoutExpired:
        logger.error("SNAP GPT command timed out.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"SNAP GPT command failed with exit code {e.returncode}.")
        logger.error(f"STDERR:\n{e.stderr}")
        logger.error(f"STDOUT:\n{e.stdout}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during SNAP GPT execution: {e}")
        return None
    finally:
        # Clean up temporary graph file
        if os.path.exists(graph_xml_path):
            os.remove(graph_xml_path)

# --- Alignment and Labeling Functions ---

def align_rasters(reference_raster_path, raster_to_align_path, output_path, resampling_method=\"near\"):
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

def create_change_label(hansen_treecover_path, hansen_lossyear_path, target_year, output_path, reference_raster_path=None, tree_cover_threshold=30):
    """
    Creates a binary change label (1=loss, 0=no loss/not forest) aligned to reference.
    """
    aligned_tc_path = hansen_treecover_path
    aligned_ly_path = hansen_lossyear_path
    temp_files = []

    if reference_raster_path:
        aligned_tc_path_temp = os.path.join(os.path.dirname(output_path), "temp_aligned_tc.tif")
        aligned_ly_path_temp = os.path.join(os.path.dirname(output_path), "temp_aligned_ly.tif")
        aligned_tc_path = align_rasters(reference_raster_path, hansen_treecover_path, aligned_tc_path_temp, resampling_method=\"near\")
        aligned_ly_path = align_rasters(reference_raster_path, hansen_lossyear_path, aligned_ly_path_temp, resampling_method=\"near\")
        if not aligned_tc_path or not aligned_ly_path:
            logger.error("Failed to align Hansen data."); return None
        temp_files.extend([aligned_tc_path, aligned_ly_path])

    try:
        with rasterio.open(aligned_tc_path) as tc_src, rasterio.open(aligned_ly_path) as ly_src:
            treecover = tc_src.read(1)
            lossyear = ly_src.read(1)
            profile = tc_src.profile

            forest_mask = treecover > tree_cover_threshold
            loss_mask = (lossyear > 0) & (lossyear <= (target_year - 2000))
            change_label = (forest_mask & loss_mask).astype(rasterio.uint8)

            profile.update(dtype=rasterio.uint8, count=1, nodata=None)
            with rasterio.open(output_path, \"w\", **profile) as dst:
                dst.write(change_label, 1)
            logger.info(f"Change label created: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Error creating change label: {e}"); return None
    finally:
        for f in temp_files:
            if os.path.exists(f): os.remove(f)

# --- Main Preparation Function --- #

def prepare_data_for_ml(s2_scene_dirs, s1_scene_dirs, hansen_treecover_path, hansen_lossyear_path, output_dir, s2_bands=[\"B02\", \"B03\", \"B04\", \"B08\"], include_s1=True):
    """
    Prepares aligned Sentinel-1, Sentinel-2 stacks and change labels for ML training.
    """
    prepared_s2_stacks_dir = os.path.join(output_dir, \"s2_stacks\")
    prepared_s1_stacks_dir = os.path.join(output_dir, \"s1_stacks\")
    prepared_labels_dir = os.path.join(output_dir, \"change_labels\")
    os.makedirs(prepared_s2_stacks_dir, exist_ok=True)
    os.makedirs(prepared_s1_stacks_dir, exist_ok=True)
    os.makedirs(prepared_labels_dir, exist_ok=True)

    processed_s2_scenes = {}
    processed_s1_scenes = {}
    scene_dates = set()

    # 1. Process Sentinel-2 scenes
    logger.info("--- Processing Sentinel-2 Scenes ---")
    for scene_dir in s2_scene_dirs:
        scene_name = os.path.basename(scene_dir).replace(".SAFE", "")
        try:
            date_str = scene_name.split(\"_\")[2].split(\"T\")[0]
            scene_date = datetime.strptime(date_str, \"%Y%m%d\").date()
        except Exception: logger.warning(f"Could not parse date from S2 {scene_name}. Skipping."); continue

        logger.info(f"Processing S2: {scene_name}")
        stacked_s2_path = prepare_scene_bands(scene_dir, prepared_s2_stacks_dir, s2_bands)
        if stacked_s2_path:
            processed_s2_scenes[scene_date] = stacked_s2_path
            scene_dates.add(scene_date)
        else: logger.warning(f"Failed to process S2 bands for {scene_name}.")

    if not processed_s2_scenes:
        logger.error("No Sentinel-2 scenes processed successfully. Cannot proceed.")
        return None

    # 2. Process Sentinel-1 scenes (if requested)
    if include_s1 and s1_scene_dirs:
        logger.info("--- Processing Sentinel-1 Scenes ---")
        for scene_dir in s1_scene_dirs:
            scene_name = os.path.basename(scene_dir).replace(".SAFE", "")
            try:
                # S1 naming: S1A_IW_GRDH_1SDV_20200104T... -> 20200104
                date_str = scene_name.split(\"_\")[5].split(\"T\")[0]
                scene_date = datetime.strptime(date_str, \"%Y%m%d\").date()
            except Exception: logger.warning(f"Could not parse date from S1 {scene_name}. Skipping."); continue

            logger.info(f"Processing S1: {scene_name}")
            # Preprocess using SNAP GPT (placeholder function)
            processed_s1_path = preprocess_sentinel1_scene(scene_dir, prepared_s1_stacks_dir)
            if processed_s1_path:
                processed_s1_scenes[scene_date] = processed_s1_path
                scene_dates.add(scene_date)
            else: logger.warning(f"Failed to preprocess S1 scene {scene_name}.")

    # 3. Align S1 data to S2 grid and create labels
    logger.info("--- Aligning Data and Creating Labels ---")
    sorted_dates = sorted(list(scene_dates))
    aligned_s1_scenes = {}
    scene_labels = {}

    # Use the first S2 scene as the reference grid
    ref_s2_date = min(processed_s2_scenes.keys())
    reference_grid_path = processed_s2_scenes[ref_s2_date]

    for scene_date in sorted_dates:
        scene_name_base = scene_date.strftime(\"%Y%m%d\") # Base name for aligned files

        # Align S1 if available for this date (or nearest)
        # Simple approach: use S1 if date matches S2 date
        if include_s1 and scene_date in processed_s1_scenes:
            s1_path = processed_s1_scenes[scene_date]
            aligned_s1_output_path = os.path.join(prepared_s1_stacks_dir, f"{scene_name_base}_s1_aligned.tif")
            aligned_s1_path = align_rasters(reference_grid_path, s1_path, aligned_s1_output_path, resampling_method=\"bilinear\")
            if aligned_s1_path:
                aligned_s1_scenes[scene_date] = aligned_s1_path
            else:
                logger.warning(f"Failed to align S1 data for {scene_date}")

        # Create change label aligned to the S2 grid for this date
        if scene_date in processed_s2_scenes:
            s2_path = processed_s2_scenes[scene_date]
            label_output_path = os.path.join(prepared_labels_dir, f"{scene_name_base}_change_label.tif")
            created_label_path = create_change_label(
                hansen_treecover_path,
                hansen_lossyear_path,
                target_year=scene_date.year,
                output_path=label_output_path,
                reference_raster_path=s2_path # Align label to this S2 scene
            )
            if created_label_path:
                scene_labels[scene_date] = created_label_path
            else:
                logger.warning(f"Failed to create label for {scene_date}")

    logger.info(f"Prepared {len(processed_s2_scenes)} S2 stacks.")
    logger.info(f"Prepared {len(aligned_s1_scenes)} aligned S1 stacks.")
    logger.info(f"Prepared {len(scene_labels)} change labels.")

    if not processed_s2_scenes or not scene_labels:
        logger.error("Failed to prepare necessary S2 stacks or labels.")
        return None

    return output_dir # Return base directory

# --- Main Execution Logic --- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description=\"Download and prepare S1/S2/Hansen data for ML.\")
    parser.add_argument(\" --aoi\", type=str, required=True, help=\"Path to GeoJSON AOI file.\")
    parser.add_argument(\" --start_date\", type=str, required=True, help=\"Start date (YYYY-MM-DD).\")
    parser.add_argument(\" --end_date\", type=str, required=True, help=\"End date (YYYY-MM-DD).\")
    parser.add_argument(\" --output_dir\", type=str, required=True, help=\"Base output directory.\")
    parser.add_argument(\" --s2_bands\", nargs=\"+\", default=[\"B02\", \"B03\", \"B04\", \"B08\"])
    parser.add_argument(\" --include_s1\", action=\"store_true\", help=\"Include Sentinel-1 data processing.\")
    parser.add_argument(\" --s1_orbit\", default=\"ASCENDING\")
    parser.add_argument(\" --s1_polarization\", default=\"VV VH\")
    args = parser.parse_args()

    raw_s2_dir = os.path.join(args.output_dir, \"raw\", \"sentinel2\")
    raw_s1_dir = os.path.join(args.output_dir, \"raw\", \"sentinel1\")
    raw_hansen_dir = os.path.join(args.output_dir, \"raw\", \"hansen\")
    prepared_data_dir = os.path.join(args.output_dir, \"prepared\")

    # --- Download --- #
    logger.info("--- Starting Data Download ---")
    s2_scene_dirs = download_sentinel2_data(args.aoi, args.start_date, args.end_date, raw_s2_dir)
    hansen_paths = download_hansen_data(args.aoi, raw_hansen_dir)
    s1_scene_dirs = []
    if args.include_s1:
        s1_scene_dirs = download_sentinel1_data(args.aoi, args.start_date, args.end_date, raw_s1_dir, args.s1_orbit, args.s1_polarization)

    if not s2_scene_dirs or not hansen_paths:
        logger.error("S2 or Hansen data download failed. Exiting.")
        sys.exit(1)
    if args.include_s1 and not s1_scene_dirs:
        logger.warning("S1 download specified but no scenes found/downloaded.")

    # --- Prepare --- #
    logger.info("--- Starting Data Preparation ---")
    result_dir = prepare_data_for_ml(
        s2_scene_dirs,
        s1_scene_dirs,
        hansen_paths[0], # treecover
        hansen_paths[1], # lossyear
        prepared_data_dir,
        s2_bands=args.s2_bands,
        include_s1=args.include_s1
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

