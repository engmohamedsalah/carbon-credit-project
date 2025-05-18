# Sentinel-2 Image Download

This document outlines the steps to download Sentinel-2 satellite imagery using the provided Python script.

## Prerequisites

- Python 3.x installed.
- Required Python packages (e.g., `requests`, `pystac-client`). Ensure these are installed in your environment (e.g., via pip).
- A Copernicus Data Space Ecosystem account.

## Download Steps

1.  **Set Environment Variables**:
    Before running the script, you need to set your Copernicus Data Space Ecosystem credentials as environment variables. Open your terminal and execute the following commands, replacing `your_username` and `your_password` with your actual credentials:

    ```bash
    export COPERNICUS_USERNAME="your_username"
    export COPERNICUS_PASSWORD="your_password"
    ```

    **Note**: Ensure your password is correctly quoted if it contains special characters.

2.  **Navigate to the Project Directory**:
    Open your terminal and change to the root directory of this project if you are not already there.

    ```bash
    cd /path/to/your/carbon_credit_project
    ```

3.  **Run the Download Script**:
    Execute the Python script `ml/download_sentinel2_stac.py` from the project's root directory:

    ```bash
    python ml/download_sentinel2_stac.py
    ```

4.  **Check Downloaded Files**:
    The script will download the Sentinel-2 product files (usually as `.zip` archives containing `.SAFE` directories) into the `sentinel2_downloads` directory within your project.

## Script Configuration

- The Area of Interest (AOI), date range, and other search parameters are defined at the beginning of the `ml/download_sentinel2_stac.py` script. You can modify these parameters as needed.
- The script currently attempts to download a maximum of 5 products per run (defined by the `limit=5` parameter in the script). This can also be adjusted. 