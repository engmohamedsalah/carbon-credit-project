import json
import os
from pystac_client import Client
import requests

# --- Parameters ---
GEOJSON_PATH = "ml/pilot_aoi_novo_progresso.geojson"
STAC_API_URL = "https://catalogue.dataspace.copernicus.eu/stac"
COLLECTION = "SENTINEL-1"
DATE_RANGE = os.getenv("S1_DATE_RANGE", "2022-12-01/2022-12-31") # Use env var if set
DOWNLOAD_DIR = "ml/data/sentinel1_downloads"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CLIENT_ID = "cdse-public"

# --- Read AOI and extract bounding box ---
with open(GEOJSON_PATH) as f:
    geojson = json.load(f)
coords = geojson["features"][0]["geometry"]["coordinates"][0]
lons = [c[0] for c in coords]
lats = [c[1] for c in coords]
bbox = [min(lons), min(lats), max(lons), max(lats)]
print("AOI BBOX:", bbox)

def get_access_token(username, password):
    data = {
        "grant_type": "password",
        "client_id": CLIENT_ID,
        "username": username,
        "password": password,
    }
    resp = requests.post(TOKEN_URL, data=data)
    if resp.status_code == 200:
        return resp.json()["access_token"]
    else:
        print(f"Failed to get access token: HTTP {resp.status_code} {resp.text}")
        return None

# --- Get Copernicus Credentials from Environment Variables ---
copernicus_username = os.getenv("COPERNICUS_USERNAME")
copernicus_password = os.getenv("COPERNICUS_PASSWORD")

if not copernicus_username or not copernicus_password:
    print("Error: COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables must be set.")
    print("Please set them before running the script, e.g.:")
    print("  export COPERNICUS_USERNAME=\"your_username\"")
    print("  export COPERNICUS_PASSWORD=\"your_password\"")
    exit(1)

# --- Connect to STAC API ---
catalog = Client.open(STAC_API_URL)

# --- Search for items ---
search = catalog.search(
    collections=[COLLECTION],
    bbox=bbox,
    datetime=DATE_RANGE,
    limit=10 # Ask for up to 10 items to check for the month
)
items = list(search.get_items())
print(f"Found {len(items)} item(s) matching criteria for {DATE_RANGE}.")

# --- Download assets ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
success_count_total = 0
fail_count_total = 0
monthly_downloads_succeeded = 0
failures_current_run = []

for item in items:
    if monthly_downloads_succeeded >= 2:
        print(f"  Target of 2 successful (or already existing and verified) downloads for this month ({DATE_RANGE}) reached. Moving to next item or month.")
        break

    # Sentinel-1 GRD products are most common for change detection
    if "GRD" not in item.id:
        print(f"Skipping non-GRD product: {item.id}")
        continue

    print(f"Product: {item.id}")

    out_filename_check = f"{item.id}_PRODUCT.zip"
    out_path_check = os.path.join(DOWNLOAD_DIR, out_filename_check)
    if os.path.exists(out_path_check) and os.path.getsize(out_path_check) > 100000:
        print(f"  File {out_filename_check} already exists and seems complete. Counting towards monthly goal.")
        monthly_downloads_succeeded += 1
        success_count_total += 1
        continue

    # Sentinel-1 STAC assets may use 'product' as the asset key
    asset_key = None
    for key in item.assets:
        if key.lower() == "product" or key.lower() == "grd":
            asset_key = key
            break
    if asset_key:
        print("  Attempting to get a new access token...")
        current_access_token = get_access_token(copernicus_username, copernicus_password)
        if not current_access_token:
            print(f"  Skipping download for {item.id} due to token acquisition failure.")
            fail_count_total += 1
            failures_current_run.append(f"{item.id} (PRODUCT): Token acquisition failed")
            continue
        headers = {"Authorization": f"Bearer {current_access_token}"}
        asset = item.assets[asset_key]
        original_url = asset.href
        download_url = original_url.replace("https://catalogue.dataspace.copernicus.eu/", "https://download.dataspace.copernicus.eu/")
        out_path = os.path.join(DOWNLOAD_DIR, f"{item.id}_PRODUCT.zip")
        print(f"  Attempting download of PRODUCT from {download_url} (original: {original_url})")
        try:
            response = requests.get(download_url, stream=True, headers=headers, timeout=180)
            if response.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Saved to {out_path}")
                monthly_downloads_succeeded += 1
                success_count_total += 1
            else:
                print(f"  Failed to download {item.id} (PRODUCT): HTTP {response.status_code} - {response.reason}")
                fail_count_total += 1
                failures_current_run.append(f"{item.id} (PRODUCT): HTTP {response.status_code} - {response.reason}")
        except requests.exceptions.RequestException as e:
            print(f"  Failed to download {item.id} (PRODUCT) due to a request exception: {e}")
            fail_count_total += 1
            failures_current_run.append(f"{item.id} (PRODUCT): RequestException - {e}")
    else:
        print(f"  No PRODUCT or GRD asset found for {item.id}!")

print(f"\nDownload summary for {DATE_RANGE}: {monthly_downloads_succeeded} succeeded this run, {len(failures_current_run)} failed this run.")
if len(failures_current_run) > 0:
    print("Failed downloads:")
    for fail in failures_current_run:
        print(f"  - {fail}") 