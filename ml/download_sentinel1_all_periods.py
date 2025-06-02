import os
import re
import calendar
import subprocess

S2_DIR = "ml/data/sentinel2_downloads"
S1_DIR = "ml/data/sentinel1_downloads"
S1_SCRIPT = "ml/download_sentinel1_stac.py"

pattern = re.compile(r"_([0-9]{8})T")  # Matches the date part
unique_months = set()

# Extract all unique year-months from Sentinel-2 filenames
for fname in os.listdir(S2_DIR):
    match = pattern.search(fname)
    if match:
        date_str = match.group(1)  # e.g., 20221207
        year = date_str[:4]
        month = date_str[4:6]
        unique_months.add((year, month))

# Find the latest year-month in Sentinel-1 downloads
s1_months = set()
for fname in os.listdir(S1_DIR):
    match = pattern.search(fname)
    if match:
        date_str = match.group(1)
        year = date_str[:4]
        month = date_str[4:6]
        s1_months.add((year, month))

# Only download for months not already present in Sentinel-1, and only for 2024
months_to_download = [
    (year, month) for (year, month) in sorted(unique_months - s1_months)
    if (year == '2024')
]

for year, month in months_to_download:
    start_date = f"{year}-{month}-01"
    end_day = calendar.monthrange(int(year), int(month))[1]
    end_date = f"{year}-{month}-{end_day}"
    date_range = f"{start_date}/{end_date}"
    print(f"\n=== Downloading Sentinel-1 for {date_range} ===")
    env = os.environ.copy()
    env["S1_DATE_RANGE"] = date_range
    subprocess.run(["python3", S1_SCRIPT], env=env)

print("\nFinished downloading Sentinel-1 for 2024. All available periods are now processed!") 