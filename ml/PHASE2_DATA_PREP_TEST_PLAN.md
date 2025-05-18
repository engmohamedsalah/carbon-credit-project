# Phase 2: Data Preparation Test Plan

**Status Summary (as of latest run):**
- ✅ All core data preparation steps (Sentinel-2, Hansen GFC, stacking, label creation) are complete and tested for the pilot AOI and date range.
- ⚠️ Visual inspection, cloud masking, and some optional QA/documentation steps are not yet done.

**Recommended Pilot AOI:**
- **Location:** Near Novo Progresso, Pará, Brazil (Amazon deforestation hotspot)
- **Size:** 10km x 10km (modifiable)
- **GeoJSON Example:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-55.0000, -7.0000],
            [-54.9000, -7.0000],
            [-54.9000, -7.1000],
            [-55.0000, -7.1000],
            [-55.0000, -7.0000]
          ]
        ]
      }
    }
  ]
}
```
- **File to use:** `pilot_aoi_novo_progresso.geojson`
- **How to adjust:** You can visualize or modify this AOI at [geojson.io](http://geojson.io/).

**Recommended Date Ranges:**
- **For quick testing:** `2022-07-01` to `2022-07-10` (expect 1–3 images)
- **For initial model training:** `2021-01-01` to `2022-12-31` (expect 30–100+ images)

**Dataset Expectations:**
- **AOI (Area of Interest):**
    - For initial testing, use the above pilot AOI (10x10 km). For model training, you may expand the area or time range.
- **Sentinel-2 Images:**
    - Number of images depends on AOI size and date range.
    - For testing: expect 1–3 images (scenes).
    - For initial model training: expect 30–100+ images (scenes) for a 1–2 year period, subject to cloud cover and revisit frequency.
- **Hansen Global Forest Change (GFC):**
    - Only the tiles intersecting your AOI are downloaded automatically by the script.
    - Used for generating training labels, not as time-series images.

This plan outlines the steps to test the `ml/utils/data_preparation.py` script, which is responsible for downloading and processing Sentinel-2 and Hansen GFC data for the carbon credit project.

**Goal:** Execute the `ml/utils/data_preparation.py` script for a small test case to ensure Sentinel-2 imagery and Hansen GFC data are downloaded and processed correctly, generating analysis-ready data for subsequent ML model training.

---

## I. Preparation Tasks

*   [x] **1. Select a Small Area of Interest (AOI):**
    *   **Action:** Use the recommended pilot AOI (see above) or adjust as needed. Save as `pilot_aoi_novo_progresso.geojson`.
    *   **Tool(s):** GIS software (e.g., QGIS), geojson.io.
    *   **Deliverable:** A GeoJSON file (e.g., `pilot_aoi_novo_progresso.geojson`) defining the AOI polygon.
    *   **Notes/Path to AOI file:** `pilot_aoi_novo_progresso.geojson`

*   [x] **2. Identify Test Time Window & Expected Images:**
    *   **Action:** Use an online satellite imagery browser to inspect the chosen AOI. For quick testing, use the date range `2022-07-01` to `2022-07-10`. For initial model training, use `2021-01-01` to `2022-12-31`.
    *   **Tool(s):** [Copernicus Browser](https://dataspace.copernicus.eu/browser/), [EO Browser](https://www.sentinel-hub.com/explore/eobrowser/).
    *   **Deliverable:** Specific start and end dates for the test.
    *   **Selected Start Date:** `2022-07-01` (test) or `2021-01-01` (training)
    *   **Selected End Date:** `2022-07-10` (test) or `2022-12-31` (training)
    *   **Expected S2 Image Date(s):** _2022-07-10 (confirmed by pipeline run)_

*   [x] **3. Set Up Environment Variables:**
    *   **Action:** Set `SENTINEL_USER` and `SENTINEL_PASSWORD` environment variables with your Copernicus Open Access Hub (SciHub) credentials in your terminal session.
        ```bash
        export SENTINEL_USER="your_scihub_username"
        export SENTINEL_PASSWORD="your_scihub_password"
        ```
    *   **Status:** Done

*   [x] **4. Prepare Output Directory:**
    *   **Action:** Create an empty directory where the script will download and save all data.
    *   **Deliverable:** Path to the output directory.
    *   **Output Directory Path:** `data/` (used in all runs)

*   [x] **5. Verify Dependencies:**
    *   **Action:**
        *   Ensure all Python packages from `requirements.txt` are installed in your virtual environment (`pip install -r requirements.txt`).
        *   Confirm GDAL command-line tools (`gdal_translate`, `gdalbuildvrt`, `gdalwarp`) are installed and accessible in your system's `PATH`.
    *   **Status:** Python Env OK / GDAL Tools OK

---

## II. Execution

*   [x] **1. Navigate to Project Directory:**
    *   **Action:** In your terminal, `cd` into the `carbon_credit_project` root directory.
*   [x] **2. Run the Script:**
    *   **Action:** Execute `ml/utils/data_preparation.py` using the parameters prepared above.
    *   **Example Command:**
        ```bash
        python ml/utils/data_preparation.py \
            --aoi ml/pilot_aoi_novo_progresso.geojson \
            --start_date 2022-07-01 \
            --end_date 2022-07-10 \
            --output_dir data
        ```
    *   **Status:** Completed
    *   **Key Log Output/Error Message (if any):**
        ```
        [See logs: pipeline completed successfully, all outputs generated.]
        ```

---

## III. Verification

*   [x] **1. Inspect Output Directory Structure:**
    *   **Action:** After script execution, check the specified output directory.
    *   **Expected Sub-directories:**
        *   `raw/sentinel2/`
        *   `raw/hansen/raw_tiles/`
        *   `raw/hansen/`
        *   `prepared/s2_stacks/`
        *   `prepared/change_labels/`
    *   **Verification:** Yes
    *   **Notes:** All expected directories present.

*   [x] **2. Verify File Contents:**
    *   **Action:** Check for the presence of expected file types within the directories.
    *   **Verification:**
        *   `.SAFE` folders in `raw/sentinel2/`? Yes
        *   `.tif` files in `raw/hansen/raw_tiles/`? Yes
        *   `hansen_clipped_*.tif` files in `raw/hansen/`? Yes
        *   Scene-specific `_s2_stack.tif` files in `prepared/s2_stacks/`? Yes
        *   Date-specific `_change_label.tif` files in `prepared/change_labels/`? Yes
    *   **Notes:** All expected files present for the test AOI/date.

*   [ ] **3. Visual Check (GIS Software):**
    *   **Action:** Use GIS software (e.g., QGIS) to open and inspect:
        *   At least one Sentinel-2 stack from `prepared/s2_stacks/`.
        *   Its corresponding change label from `prepared/change_labels/`.
        *   Clipped Hansen layers from `raw/hansen/`.
    *   **Verification:**
        *   S2 stack covers AOI and looks correct? _Not yet checked_
        *   Label is aligned, binary, and appears reasonable? _Not yet checked_
        *   Hansen data covers AOI? _Not yet checked_
    *   **Notes:** _Visual inspection pending._

*   [ ] **4. Review Log Output:**
    *   **Action:** Check the terminal output from the script execution for any warning or error messages not captured during the run.
    *   **Status:** _Pending final review_
    *   **Notes:** _No critical errors seen, but full review recommended._

---

## IV. Next Steps (Post Successful Test)

*   [ ] **Debrief:** Document any issues found and resolutions.
*   [ ] **Scale Gradually (Optional):** If initial tests are successful, consider tests with slightly larger AOIs or longer time periods if necessary for further validation.
*   [ ] **Proceed to ML Model Training:** Use the successfully prepared data as input for training the machine learning models outlined in `ml/training/`.

---
Date Created: $(date +%Y-%m-%d) 