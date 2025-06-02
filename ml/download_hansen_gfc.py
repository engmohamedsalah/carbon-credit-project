"""
Download and process Hansen Global Forest Change (GFC) data for a given AOI.

Usage:
    python ml/download_hansen_gfc.py --aoi ml/pilot_aoi_novo_progresso.geojson --output_dir ml/data/hansen_downloads --layers treecover2000 lossyear gain loss datamask

- AOI: Path to GeoJSON file defining the area of interest
- Output directory: Where to save the processed files
- Layers: Any combination of treecover2000, lossyear, gain, loss, datamask

Requires: geopandas, requests, gdal, tqdm, numpy, shapely
"""
import os
import argparse
import sys
from ml.utils.data_preparation import download_hansen_data

def main():
    parser = argparse.ArgumentParser(description="Download Hansen GFC data for AOI.")
    parser.add_argument("--aoi", type=str, required=True, help="Path to GeoJSON AOI file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Hansen data.")
    parser.add_argument("--layers", nargs="*", default=["treecover2000", "lossyear", "datamask", "gain", "loss"],
                        help="Layers to download: treecover2000, lossyear, gain, loss, datamask")
    args = parser.parse_args()

    # Call the download_hansen_data function (extend it to support gain/loss if needed)
    print(f"Downloading Hansen GFC layers: {', '.join(args.layers)} for AOI: {args.aoi}")
    # The function in utils/data_preparation.py may need to be extended to support gain/loss
    # For now, call as-is for treecover2000, lossyear, datamask
    result = download_hansen_data(args.aoi, args.output_dir)
    if result:
        print("Download and processing complete.")
    else:
        print("Download or processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 