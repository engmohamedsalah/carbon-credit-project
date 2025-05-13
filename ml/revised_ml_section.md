

### 1.1 ML Model Training - Technical Details (Revised)

This section details the enhanced machine learning pipeline incorporating multiple data sources and advanced models for robust carbon credit verification.

#### Data Sources and Preparation (Revised)

- **Sentinel-2 (Optical)**:
    - **Source**: Copernicus Open Access Hub (Level-2A products).
    - **Selection**: Filter for cloud cover < 20%, similar seasons.
    - **Bands**: Typically B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), but configurable.
    - **Preparation (`data_preparation.py`)**: Band extraction, stacking into multi-band GeoTIFFs.

- **Sentinel-1 (SAR - New)**:
    - **Source**: Copernicus Open Access Hub (GRD IW products).
    - **Purpose**: Overcomes cloud cover limitations of optical data.
    - **Selection**: Specify orbit direction (Ascending/Descending) and polarization (VV, VH, or both).
    - **Preparation (`data_preparation.py` & SNAP)**: Requires significant preprocessing using ESA SNAP (or similar SAR processing software) via the `preprocess_sentinel1_scene` function (which orchestrates SNAP GPT calls). Steps include: Apply Orbit File, Radiometric Calibration (to Sigma0), Speckle Filtering (e.g., Lee Sigma), and Terrain Correction (using a DEM like SRTM).

- **Hansen Global Forest Change (GFC)**:
    - **Source**: Google Cloud Storage.
    - **Layers**: `treecover2000` and `lossyear`.
    - **Preparation (`data_preparation.py`)**: Downloading relevant tiles, merging, clipping to AOI.

- **Alignment and Labeling (`data_preparation.py`)**:
    - All data sources (S1, Hansen) are aligned to the Sentinel-2 grid using `align_rasters` (GDAL Warp).
    - Change labels (`create_change_label`) are generated using Hansen data, aligned to the corresponding Sentinel-2 scene.

- **Data Structuring for Models (`data_preparation.py`)**:
    - The script now prepares data in multiple formats:
        - **Single Images**: For baseline forest cover models (like the original U-Net).
        - **Image Pairs**: For the Siamese Change Detection model (e.g., baseline vs. monitoring image).
        - **Image Sequences**: For the Time-Series (ConvLSTM) model (e.g., multiple images over time).

#### ML Models (Revised)

The pipeline now supports multiple complementary models:

1.  **Siamese UNet for Change Detection (New)**:
    - **Architecture (`ml/models/siamese_unet.py`)**: Takes pairs of aligned image patches (e.g., from different dates) as input. Uses a shared UNet encoder backbone to extract features from each image and then compares these features (e.g., by difference or concatenation) before a final decoder predicts areas of change.
    - **Purpose**: Directly models the *change* between two time points, potentially improving accuracy for deforestation/reforestation detection.
    - **Training (`ml/training/train_change_detection.py`)**: Uses the `PairedSatelliteDataset` and a contrastive or binary cross-entropy loss on the change prediction.
    - **Output**: Change map highlighting differences between the input pair.

2.  **ConvLSTM for Time-Series Analysis (New)**:
    - **Architecture (`ml/models/convlstm_model.py`)**: Processes sequences of image patches over time. Each ConvLSTM layer maintains an internal state, allowing the model to learn temporal dependencies and patterns (e.g., seasonal effects, gradual changes).
    - **Purpose**: Improves robustness to seasonal variations and captures temporal trends for more accurate land cover classification or change prediction over time.
    - **Training (`ml/training/train_time_series.py`)**: Uses the `SatelliteTimeSeriesDataset` and typically predicts the state (e.g., forest/non-forest or change/no-change) for the final time step in the sequence.
    - **Output**: Prediction map for the last time step, informed by the entire sequence history.

3.  **UNet for Forest Cover Segmentation (Original/Optional)**:
    - **Architecture (`ml/models/unet_model.py` - *Note: This file might need creation/update if not present*)**: Standard UNet for semantic segmentation on single image patches.
    - **Purpose**: Provides a baseline forest/non-forest classification for a single point in time.
    - **Training (`ml/training/train_forest_change.py` - *Note: This script might need renaming/refactoring*)**: Uses a standard `Dataset` loading single images.
    - **Output**: Forest cover probability map.

#### Training Process (Revised)

- **Scripts**: Separate training scripts exist for each model type (`train_change_detection.py`, `train_time_series.py`, potentially `train_forest_cover.py`).
- **Data**: Each script uses its corresponding `Dataset` class defined within it or imported from `ml/utils/data_preparation.py` (or a dedicated dataset file).
- **Parameters**: Hyperparameters (learning rate, batch size, epochs, loss function) are configured within each training script.
- **Hardware**: GPU acceleration (CUDA) is highly recommended for all models.
- **Output**: Each training script saves its trained model weights (e.g., `cd_model.pth`, `ts_model.pth`, `fc_model.pth`) to `/home/ubuntu/carbon_credit_project/ml/models/`.

#### Inference and Fusion (Revised)

- **Loading**: The backend service (`satellite_service.py`) needs to be updated to load the desired trained models (CD, TS, FC).
- **Prediction**: When a verification request is made, the backend preprocesses the input data (single image, pair, or sequence) and feeds it to the relevant model(s).
- **Fusion**: Results from different models (e.g., direct change map from Siamese UNet, time-series prediction from ConvLSTM) can be combined or used in conjunction (e.g., using the change map to focus analysis) to produce a final, more robust verification result. The specific fusion logic needs to be defined based on project requirements.

#### Explainable AI (XAI)

- **Methods**: Techniques like Integrated Gradients, SHAP, or Occlusion Sensitivity can be adapted and applied to the chosen model(s) (CD, TS, or FC) to understand feature importance and model predictions.
- **Visualization (`ml/utils/xai_visualization.py`)**: Scripts to generate visual explanations (e.g., heatmaps overlaid on input images).

#### Carbon Estimation (Revised)

- **Input**: Uses the final (potentially fused) analysis of forest change area derived from the ML model outputs.
- **Model (`ml/inference/estimate_carbon_sequestration.py`)**: Employs methods like look-up tables based on biome/region, or potentially a separate model (e.g., Random Forest trained on plot data, if available) using features derived from satellite data (NDVI, texture, backscatter from S1) to estimate biomass change and carbon impact.
- **Validation**: LiDAR data, where available, can be used as a high-quality reference to validate biomass/carbon estimates for specific areas.

#### ML Pipeline Diagram

Refer to the updated ML pipeline diagram (`/home/ubuntu/carbon_credit_project/ml_pipeline.png` or `.dot`) for a visual overview of the enhanced data flow and model interactions.


