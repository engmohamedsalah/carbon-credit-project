import json
import os
from pystac_client import Client
import requests

# --- Parameters ---
GEOJSON_PATH = "ml/pilot_aoi_novo_progresso.geojson"
STAC_API_URL = "https://catalogue.dataspace.copernicus.eu/stac"
COLLECTION = "SENTINEL-2"
DATE_RANGE = "2022-12-01/2022-12-31" # December 2022
DOWNLOAD_DIR = "./sentinel2_downloads"
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
        return None # Return None on failure

# --- Get Copernicus Credentials from Environment Variables ---
copernicus_username = os.getenv("COPERNICUS_USERNAME")
copernicus_password = os.getenv("COPERNICUS_PASSWORD")

if not copernicus_username or not copernicus_password:
    print("Error: COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables must be set.")
    print("Please set them before running the script, e.g.:")
    print("  export COPERNICUS_USERNAME=\"your_username\"")
    print("  export COPERNICUS_PASSWORD=\"your_password\"")
    exit(1)

# The initial token fetch and headers definition here are removed,
# as token will be fetched per item.

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
success_count_total = 0 # To track total success across multiple runs if we were to call this function multiple times
fail_count_total = 0    # To track total failures

monthly_downloads_succeeded = 0
failures_current_run = []

for item in items:
    if monthly_downloads_succeeded >= 2:
        print(f"  Target of 2 successful (or already existing and verified) downloads for this month ({DATE_RANGE}) reached. Moving to next item or month.")
        break

    if "L2A" not in item.id:
        print(f"Skipping non-Level-2A product: {item.id}")
        continue
    
    print(f"Product: {item.id}")

    # Check if this L2A product already exists and is valid before attempting token refresh/download
    out_filename_check = f"{item.id}_PRODUCT.zip"
    out_path_check = os.path.join(DOWNLOAD_DIR, out_filename_check)
    if os.path.exists(out_path_check) and os.path.getsize(out_path_check) > 100000: # Check if exists and >100KB
        print(f"  File {out_filename_check} already exists and seems complete. Counting towards monthly goal.")
        monthly_downloads_succeeded += 1
        success_count_total +=1 # Consider it a success for overall stats too
        continue # Move to the next item in the STAC results

    # If we are here, the file doesn't exist or is too small, and we haven't hit our 2-per-month goal yet.
    # Proceed to download only if PRODUCT asset is available.
    if "PRODUCT" in item.assets:
        print("  Attempting to get a new access token...")
        current_access_token = get_access_token(copernicus_username, copernicus_password)

        if not current_access_token:
            print(f"  Skipping download for {item.id} due to token acquisition failure.")
            fail_count_total +=1
            failures_current_run.append(f"{item.id} (PRODUCT): Token acquisition failed")
            continue

        headers = {"Authorization": f"Bearer {current_access_token}"}
        
        asset = item.assets["PRODUCT"]
        original_url = asset.href
        download_url = original_url.replace("https://catalogue.dataspace.copernicus.eu/", "https://download.dataspace.copernicus.eu/")
        
        # out_path is now defined inside this block, using the same name as out_path_check but it's fine
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
        print(f"  No PRODUCT asset found for {item.id}!")

print(f"\nDownload summary for {DATE_RANGE}: {monthly_downloads_succeeded} succeeded this run, {len(failures_current_run)} failed this run.")
# The success_count_total and fail_count_total would be more for a script that runs across multiple date ranges by itself.
# For now, the monthly_downloads_succeeded is the key metric for this run.

if len(failures_current_run) > 0:
    print("Failed downloads:")
    for fail in failures_current_run:
        print(f"  - {fail}") 