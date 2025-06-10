import os
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

# User configuration
USERNAME = 'eng.mohamed.tawab@gmail.com'
PASSWORD = '*EX_#y.S5#%PSE+'
API_URL = 'https://scihub.copernicus.eu/dhus'
TILE = 'T21MYN'  # <-- Replace with your tile of interest
DATE_RANGE = ('20220101', '20221231')  # <-- Replace with your desired date range (YYYYMMDD)
CLOUD_COVER = (0, 20)  # Maximum cloud cover percentage
DOWNLOAD_DIR = 'ml/data/sentinel2_downloads'

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def main():
    api = SentinelAPI(USERNAME, PASSWORD, API_URL)
    # Search for products
    products = api.query(
        area=None,  # No geojson, just tile
        date=DATE_RANGE,
        platformname='Sentinel-2',
        processinglevel='Level-2A',
        cloudcoverpercentage=CLOUD_COVER,
        filename=f'*_{TILE}_*'
    )
    print(f'[INFO] Found {len(products)} products for tile {TILE} in {DATE_RANGE}')
    # Download all products
    api.download_all(products, directory_path=DOWNLOAD_DIR)
    print('[DONE] Download complete.')

if __name__ == '__main__':
    main() 