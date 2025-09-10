# Carbon Credit Verification System - Data Sources & Pipeline Documentation

## Executive Summary

The Carbon Credit Verification System utilizes three primary satellite data sources to monitor and verify forest changes: Sentinel-1 SAR imagery, Sentinel-2 optical imagery, and Hansen Global Forest Change data. The combined dataset exceeds hundreds of gigabytes, requiring specialized download scripts and processing pipelines. All data sources are publicly available and used in compliance with their respective licenses.

## Table of Contents

1. [Data Sources Overview](#data-sources-overview)
2. [Data Volume and Scale](#data-volume-and-scale)
3. [Download Scripts](#download-scripts)
4. [Data Pipeline Architecture](#data-pipeline-architecture)
5. [Ethical and Legal Considerations](#ethical-and-legal-considerations)
6. [Example Data Formats](#example-data-formats)
7. [Processing Pipeline Steps](#processing-pipeline-steps)
8. [Storage Requirements](#storage-requirements)

## Data Sources Overview

### 1. Sentinel-1 (SAR Imagery)
- **Provider**: European Space Agency (ESA) / Copernicus Programme
- **Type**: Synthetic Aperture Radar (SAR) imagery
- **Resolution**: 10m Ground Range Detected (GRD) products
- **Frequency**: Every 6-12 days
- **Use Case**: Cloud-penetrating radar data for all-weather forest monitoring

### 2. Sentinel-2 (Optical Imagery)
- **Provider**: European Space Agency (ESA) / Copernicus Programme
- **Type**: Multispectral optical imagery
- **Resolution**: 10m (visible bands), 20m (vegetation red edge), 60m (atmospheric bands)
- **Frequency**: Every 5 days at equator
- **Use Case**: High-resolution optical imagery for vegetation analysis

### 3. Hansen Global Forest Change (GFC)
- **Provider**: University of Maryland / Google Earth Engine
- **Type**: Pre-processed forest change products
- **Resolution**: 30m
- **Coverage**: 2000-2023 annual updates
- **Layers**:
  - `treecover2000`: Percent tree cover in year 2000
  - `lossyear`: Year of forest loss (0-23)
  - `gain`: Forest gain 2000-2012 (binary)
  - `loss`: Forest loss 2000-2023 (binary)
  - `datamask`: Valid data mask

## Data Volume and Scale

### Approximate Data Sizes

#### Per Scene/Tile:
- **Sentinel-1 GRD**: ~1-2 GB per scene (compressed)
- **Sentinel-2 L2A**: ~500-800 MB per tile (compressed)
- **Hansen GFC**: ~50-200 MB per layer per tile

#### For Typical Project Area (1000 km²):
- **Annual Sentinel-1**: ~100-200 GB
- **Annual Sentinel-2**: ~50-100 GB (cloud-free scenes)
- **Hansen GFC (all layers)**: ~1-5 GB

#### Total Project Requirements:
- **Raw Data**: 200-500 GB per year
- **Processed Data**: 100-300 GB
- **ML Training Data**: 50-100 GB
- **Model Checkpoints**: 5-10 GB

## Download Scripts

### 1. Sentinel-1 Download Script

**Location**: `ml/download_sentinel1_stac.py`

**Usage**:
```bash
# Set credentials
export COPERNICUS_USERNAME="your_username"
export COPERNICUS_PASSWORD="your_password"

# Set date range (optional)
export S1_DATE_RANGE="2022-12-01/2022-12-31"

# Run download
python ml/download_sentinel1_stac.py
```

**Features**:
- Uses ESA's Copernicus Data Space STAC API
- Downloads GRD products for specified AOI
- Automatic authentication token management
- Resume capability for interrupted downloads
- Filters by date range and product type

### 2. Sentinel-2 Download Script

**Location**: `ml/download_sentinel2_stac.py`

**Usage**:
```bash
# Set credentials (same as Sentinel-1)
export COPERNICUS_USERNAME="your_username"
export COPERNICUS_PASSWORD="your_password"

# Run download
python ml/download_sentinel2_stac.py
```

**Features**:
- Downloads Level-2A (atmospherically corrected) products
- Cloud coverage filtering (<20%)
- Automatic scene selection for AOI
- Batch download with progress tracking

### 3. Hansen Global Forest Change Download Script

**Location**: `ml/download_hansen_gfc.py`

**Usage**:
```bash
# Download all layers for AOI
python ml/download_hansen_gfc.py \
    --aoi ml/pilot_aoi_novo_progresso.geojson \
    --output_dir ml/data/hansen_downloads \
    --layers treecover2000 lossyear gain loss datamask
```

**Features**:
- Downloads from University of Maryland servers
- Automatic tiling based on Hansen grid system
- Clips to AOI boundary
- Supports selective layer download

### 4. Batch Download Scripts

**Multiple time periods** (Sentinel-1):
```bash
python ml/download_sentinel1_all_periods.py
```

**Scene inventory check**:
```bash
python ml/scripts/check_data_inventory.py
```

## Data Pipeline Architecture

### Pipeline Flow Diagram

```
1. DATA ACQUISITION
   ├── Sentinel-1 STAC API → GRD Products (.zip)
   ├── Sentinel-2 STAC API → L2A Products (.zip)
   └── Hansen GFC HTTP → GeoTIFF layers

2. DATA EXTRACTION
   ├── Unzip Sentinel products → .SAFE folders
   ├── Extract bands → Individual GeoTIFFs
   └── Reproject/align all data → Common grid

3. PREPROCESSING
   ├── Cloud masking (Sentinel-2 SCL band)
   ├── Radiometric calibration
   ├── Calculate indices (NDVI, EVI)
   └── Create temporal stacks

4. PATCH GENERATION
   ├── Tile into 256x256 patches
   ├── Create training pairs (T1/T2)
   ├── Generate change labels
   └── Balance dataset

5. MODEL TRAINING
   ├── Forest Cover U-Net
   ├── Change Detection Siamese U-Net
   └── Time Series ConvLSTM

6. INFERENCE PIPELINE
   ├── Load new imagery
   ├── Apply preprocessing
   ├── Run ensemble model
   └── Generate change maps
```

### Detailed Pipeline Steps

#### Step 1: Area of Interest Definition
```bash
# Define AOI in GeoJSON format
# Example: ml/pilot_aoi_novo_progresso.geojson
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-55.5, -8.0],
        [-55.0, -8.0],
        [-55.0, -7.5],
        [-55.5, -7.5],
        [-55.5, -8.0]
      ]]
    }
  }]
}
```

#### Step 2: Data Download
```bash
# Download all data sources
./scripts/download_all_data.sh

# Or individually:
python ml/download_sentinel1_stac.py
python ml/download_sentinel2_stac.py
python ml/download_hansen_gfc.py --aoi AOI.geojson
```

#### Step 3: Data Preparation
```bash
# Extract and prepare Sentinel-2 data
python ml/utils/batch_prepare_sentinel2_from_zips.py

# Extract and prepare Sentinel-1 data
python ml/utils/batch_prepare_sentinel1_from_zips.py

# Prepare change detection pairs
python ml/scripts/prepare_change_detection_data.py
```

#### Step 4: Dataset Creation
```bash
# Create balanced training dataset
python ml/scripts/balance_forest_cover_data.py

# Extract patches for training
python ml/scripts/extract_sentinel2_patches.py

# Verify data quality
python ml/scripts/check_data_balance.py
```

#### Step 5: Model Training
```bash
# Train individual models
python ml/training/train_forest_cover_unet.py
python ml/training/train_change_detection.py
python ml/training/train_convlstm.py

# Or use unified training script
python ml/train_all_models.py
```

#### Step 6: Production Inference
```bash
# Run inference on new data
python ml/inference/predict_forest_change.py \
    --t1_dir data/sentinel_t1/new_scene \
    --t2_dir data/sentinel_t2/new_scene \
    --output results/
```

## Ethical and Legal Considerations

### Data Licensing

#### Sentinel Data (ESA/Copernicus)
- **License**: Full, free and open access
- **Terms**: Copernicus Data Policy
- **Citation Required**: Yes
- **Commercial Use**: Allowed
- **Redistribution**: Allowed with attribution

**Required Citation**:
```
"Contains modified Copernicus Sentinel data [Year]"
```

#### Hansen Global Forest Change
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Terms**: Free use with attribution
- **Commercial Use**: Allowed
- **Research Use**: Encouraged

**Required Citation**:
```
Hansen, M. C., et al. (2013). "High-Resolution Global Maps of 
21st-Century Forest Cover Change." Science 342(6160): 850-853.
```

### Ethical Considerations

1. **Environmental Impact**
   - Data used exclusively for environmental conservation
   - Supports UN Sustainable Development Goals
   - Helps combat illegal deforestation

2. **Privacy Protection**
   - Satellite resolution (10-30m) insufficient for individual identification
   - No personal data collected or processed
   - Focus on forest areas, not populated regions

3. **Transparency**
   - All algorithms open-source
   - Verification results auditable
   - Explainable AI provides decision transparency

4. **Data Sovereignty**
   - Respects national boundaries
   - Works with local environmental agencies
   - Supports indigenous land rights monitoring

### Compliance Requirements

1. **GDPR Compliance**
   - No personal data processing
   - Environmental data only
   - Right to explanation for AI decisions

2. **Terms of Service**
   - Copernicus Data Space account required
   - Rate limiting respected (max 4 concurrent downloads)
   - No redistribution of raw Copernicus data

3. **Academic Use**
   - Proper citations required
   - Research outputs should acknowledge data sources
   - Consider co-authorship for significant contributions

## Example Data Formats

### Sentinel-1 GRD Product Structure
```
S1A_IW_GRDH_1SDV_20221201T084831_20221201T084856_046091_0583FB_7C8B.SAFE/
├── annotation/
│   ├── calibration/
│   └── s1a-iw-grd-vv-20221201t084831-20221201t084856-046091-0583fb-001.xml
├── measurement/
│   └── s1a-iw-grd-vv-20221201t084831-20221201t084856-046091-0583fb-001.tiff
└── manifest.safe
```

### Sentinel-2 L2A Product Structure
```
S2A_MSIL2A_20221201T134201_N0400_R024_T22KGA_20221201T181920.SAFE/
├── DATASTRIP/
├── GRANULE/
│   └── L2A_T22KGA_A038968_20221201T134638/
│       └── IMG_DATA/
│           ├── R10m/  (B02, B03, B04, B08 - 10m resolution)
│           ├── R20m/  (B05, B06, B07, B8A, B11, B12, SCL - 20m)
│           └── R60m/  (B01, B09, B10 - 60m resolution)
└── MTD_MSIL2A.xml
```

### Hansen GFC GeoTIFF Format
```
# Each layer is a single-band GeoTIFF
treecover2000.tif  # Byte (0-100) - percent tree cover
lossyear.tif       # Byte (0-23) - year of loss since 2000
gain.tif           # Byte (0-1) - binary gain mask
loss.tif           # Byte (0-1) - binary loss mask
datamask.tif       # Byte (0-2) - 0=no data, 1=land, 2=water
```

### Processed Training Data Format
```
ml/data/
├── sentinel_t1/          # Time 1 imagery
│   └── TILE_DATE1_DATE2/
│       ├── B02.tif      # Blue band
│       ├── B03.tif      # Green band
│       ├── B04.tif      # Red band
│       └── B08.tif      # NIR band
├── sentinel_t2/          # Time 2 imagery
│   └── TILE_DATE1_DATE2/
│       └── (same structure as t1)
└── change_labels/        # Binary change masks
    └── TILE_DATE1_DATE2_change.tif
```

## Storage Requirements

### Development Environment
```
carbon_credit_project/
├── ml/
│   ├── data/                    # ~500 GB
│   │   ├── sentinel1_downloads/ # ~200 GB
│   │   ├── sentinel2_downloads/ # ~200 GB
│   │   ├── hansen_downloads/    # ~5 GB
│   │   ├── processed/          # ~50 GB
│   │   └── training_patches/   # ~45 GB
│   ├── models/                 # ~10 GB
│   └── checkpoints/            # ~20 GB
└── database/                   # ~1 GB
```

### Production Recommendations

1. **Storage Type**
   - SSD for active processing
   - HDD acceptable for archive
   - Cloud storage for backup (S3/GCS)

2. **Capacity Planning**
   - Minimum: 1 TB for development
   - Recommended: 2-3 TB for full pipeline
   - Production: 5+ TB with redundancy

3. **Performance Optimization**
   - Use compression for archived data
   - Implement data lifecycle policies
   - Regular cleanup of intermediate files

## Running the Complete Pipeline

### Quick Start
```bash
# 1. Set up credentials
export COPERNICUS_USERNAME="your_username"
export COPERNICUS_PASSWORD="your_password"

# 2. Download sample data
./scripts/download_sample_data.sh

# 3. Prepare training data
./scripts/prepare_ml_data.sh

# 4. Train models (or use pre-trained)
python ml/train_all_models.py

# 5. Run inference
python ml/inference/predict_forest_change.py \
    --config ml/configs/inference_config.yaml
```

### Production Pipeline
```bash
# Automated monthly processing
0 0 1 * * /path/to/carbon_credit_project/scripts/monthly_pipeline.sh
```

## Troubleshooting

### Common Issues

1. **Download Failures**
   - Check Copernicus credentials
   - Verify internet connectivity
   - Ensure sufficient disk space
   - Check API rate limits

2. **Memory Errors**
   - Process smaller tiles
   - Increase swap space
   - Use data generators for training

3. **Alignment Issues**
   - Verify CRS consistency
   - Check for corrupt downloads
   - Use provided alignment scripts

## References

1. ESA Copernicus Open Access Hub: https://scihub.copernicus.eu/
2. Copernicus Data Space Ecosystem: https://dataspace.copernicus.eu/
3. Hansen Global Forest Change: https://glad.earthengine.app/view/global-forest-change
4. Sentinel-1 User Guide: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar
5. Sentinel-2 User Guide: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi

---

*For technical support, refer to the project README.md or contact the development team.*
