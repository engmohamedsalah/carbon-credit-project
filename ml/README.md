# Machine Learning Component

This directory contains the machine learning components of the Carbon Credit Verification project, including data, models, and processing scripts.

## Directory Structure

```
ml/
├── data/                  # Data storage
│   ├── sentinel2_downloads/    # Raw Sentinel-2 satellite imagery
│   │   └── [S2A/B_MSIL2A_*.zip]  # Level-2A products
│   ├── hansen_downloads/       # Hansen Global Forest Change data
│   │   ├── hansen_clipped_treecover2000.tif  # Tree cover data
│   │   ├── hansen_clipped_lossyear.tif       # Forest loss data
│   │   └── hansen_clipped_datamask.tif       # Data quality mask
│   └── prepared/              # Processed data
│       ├── change_labels/     # Change detection outputs
│       ├── s2_stacks/        # Processed Sentinel-2 data
│       └── quicklooks/       # Visualization outputs
├── inference/            # Inference scripts
├── models/              # Trained models
├── training/            # Training scripts
└── utils/               # Utility functions
```

## Data Sources and Model Pipeline

### Data Sources

1. **Sentinel-1 Data (SAR)**
   - **Purpose**: Provides Synthetic Aperture Radar (SAR) imagery that can penetrate clouds and capture data regardless of weather conditions or time of day
   - **Key Advantages**:
     - Enables continuous monitoring even in tropical regions with persistent cloud cover
     - Works through clouds and at night
     - Sensitive to structural changes
   - **Usage in Project**:
     - Complements optical data for robust forest change detection
     - Essential during cloudy seasons when optical sensors cannot capture clear images
     - Provides structural information for biomass estimation

2. **Sentinel-2 Data (Optical)**
   - **Purpose**: Provides high-resolution optical imagery with multiple spectral bands (including visible, near-infrared, and short-wave infrared)
- **Key Advantages**:
     - Excellent for vegetation analysis through indices like NDVI
     - High spectral resolution for detailed land cover classification
     - Regular coverage (5-day revisit time)
   - **Usage in Project**:
     - Primary data source for detecting forest cover and health
     - Monitoring changes over time
     - Detailed land cover classification

3. **Hansen Global Forest Change Data**
   - **Purpose**: Provides pre-analyzed global forest cover and forest loss/gain information
- **Key Advantages**:
     - Already processed dataset with tree cover percentage
     - Annual forest loss data
     - Historical baseline from 2000
   - **Usage in Project**:
     - Training labels/ground truth for machine learning models
     - Validation of change detection results
     - Historical baseline establishment

### Data Preprocessing

1. **Spatial Alignment**:
   - Co-registration of Sentinel-1 and Sentinel-2 data
   - Alignment with Hansen data
   - Consistent spatial resolution across datasets

2. **Temporal Alignment**:
   - Time series synchronization
   - Seasonal pattern normalization
   - Gap filling for cloudy periods

3. **Quality Control**:
   - Cloud masking
   - Atmospheric correction
   - Data quality assessment

### Model Pipeline

1. **Forest Cover Classification (U-Net)**
   - **Type**: Classification (Semantic Segmentation)
   - **Output**: Pixel-wise classification of forest/non-forest areas
   - **Purpose**: Establishes baseline forest cover
   - **Validation**: Cross-validation with Hansen data

2. **Change Detection (Siamese U-Net)**
   - **Type**: Classification (Semantic Segmentation)
   - **Output**: Pixel-wise classification of changed/unchanged areas
   - **Purpose**: Identifies areas of deforestation or reforestation
   - **Integration**: Combines Sentinel-1 and Sentinel-2 results

3. **Time-Series Analysis (ConvLSTM)**
   - **Type**: Classification with temporal dimension
   - **Output**: Pixel-wise classification accounting for seasonal changes
   - **Purpose**: Distinguishes between seasonal changes and actual deforestation
   - **Features**: 
     - Temporal pattern recognition
     - Seasonal variation handling
    - Long-term trend analysis

4. **Carbon Estimation**
   - **Type**: Regression
   - **Output**: Estimated carbon sequestration/emission values
   - **Purpose**: Quantifies carbon impact based on forest changes
   - **Inputs**: 
     - Forest cover changes
     - Biomass estimates
     - Historical data

### Pipeline Integration

1. **Model Output Combination**:
   - Weighted fusion of different model outputs
   - Confidence scoring for each prediction
   - Uncertainty quantification

2. **Verification Process**:
   - Multi-source validation
   - Quality control checks
   - Human-in-the-loop verification

3. **Blockchain Integration**:
   - Model outputs stored on blockchain
   - Verification certificates
   - Immutable audit trail

### Validation and Quality Assurance

1. **Cross-Validation**:
   - Between different data sources
   - Across different time periods
   - Using multiple models

2. **Ground Truth Validation**:
   - Field data verification
   - High-resolution imagery checks
   - Expert review

3. **Continuous Monitoring**:
   - Regular model performance assessment
   - Data quality monitoring
   - System health checks

## Data Sources

1. **Sentinel-2 Data**
   - Source: Copernicus Open Access Hub
   - Level: 2A (Atmospheric corrected)
   - Time period: 2020-2024
   - Bands: All spectral bands
   - Format: SAFE format (zipped)

2. **Hansen Data**
   - Source: Global Forest Change dataset
   - Products:
     - Tree cover (2000)
     - Forest loss
     - Data quality mask
   - Format: GeoTIFF

## Processed Data

1. **Change Labels**
   - Generated from Sentinel-2 time series
   - Shows forest cover changes
   - Format: GeoTIFF

2. **Sentinel-2 Stacks**
   - Processed Sentinel-2 imagery
   - Includes cloud masks
   - Format: GeoTIFF

3. **Quicklooks**
   - Visualizations for analysis
   - Includes RGB composites and NDVI
   - Format: PNG

## Usage

- Raw data should not be modified
- Processed data can be regenerated from raw data
- Keep backups of important processed outputs
- Use the quicklooks for visual verification

## Data Management

- Large files are tracked using Git LFS
- Raw data should be downloaded using the provided scripts
- Processed data can be regenerated using the ML pipeline

## ML Pipeline

1. **Data Preparation**
   - Download raw satellite imagery
   - Preprocess and stack bands
   - Generate cloud masks

2. **Model Training**
   - Train change detection models
   - Validate model performance
   - Save trained models

3. **Inference**
   - Process new satellite imagery
   - Generate change detection results
   - Create visualizations

## Requirements

- Python 3.10+
- PyTorch
- GDAL
- scikit-learn
- Other dependencies listed in requirements.txt

## Data Summary

- **Data Sources Summary:**
  - See `ml/data/data_sources_summary.csv` for a list of all months/years with Sentinel-1 and Sentinel-2 data, and a reference to the Hansen Global Forest Change dataset.
  - This CSV serves as a unified reference for all three data sources used in the project.

## Hansen Global Forest Change Data

The Hansen Global Forest Change (GFC) dataset provides global, high-resolution (30m) maps of forest cover and change. The following layers are used in this project and are clipped to the AOI:

- **treecover2000**: Percent tree canopy cover for the year 2000 (0–100). Used as a baseline for forest extent.
- **lossyear**: Year of forest loss per pixel (0 = no loss, 1 = loss in 2001, 2 = loss in 2002, ..., 23 = loss in 2023).
- **gain**: Forest gain (2000–2012), binary (1 = gain, 0 = no gain).
- **datamask**: Data quality mask (1 = mapped land, 2 = water, 0 = no data).

All layers are stored as GeoTIFFs in `ml/data/hansen_downloads/` and are spatially aligned to the project AOI. These layers are used for:
- Generating training labels for change detection
- Providing ground truth for model validation
- Establishing historical forest baselines

For more details, see: https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.10.html
