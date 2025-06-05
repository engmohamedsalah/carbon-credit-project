import os
import re
import subprocess
import time
import zipfile

# List of suspected corrupted files
corrupted_files = [
    "S2A_MSIL2A_20200102T140051_N0500_R067_T21MYN_20230628T154148.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20200112T135641_N0500_R067_T21MYN_20230604T033350.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20200211T140051_N0500_R067_T21MYN_20230624T150234.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20200221T135641_N0500_R067_T21MYN_20230610T073204.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20210205T140051_N0500_R067_T21MYN_20230604T232943.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20210516T140051_N0500_R067_T21MYN_20230222T140823.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20210705T140101_N0500_R067_T21MYN_20230201T230618.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20210725T140101_N0500_R067_T21MYN_20230220T133906.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20211013T140101_N0500_R067_T21MYN_20230127T102905.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20220908T140101_N0510_R067_T21MYN_20240723T060614.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20240609T135711_N0510_R067_T21MYN_20240609T182950.SAFE_PRODUCT.zip",
    "S2A_MSIL2A_20240619T135711_N0510_R067_T21MYN_20240619T190551.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20200127T135639_N0500_R067_T21MYN_20230622T191728.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20200216T135639_N0500_R067_T21MYN_20230610T051615.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20200307T140049_N0500_R067_T21MYN_20230422T223734.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20210210T140049_N0500_R067_T21MYN_20230608T033106.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20210220T140049_N0500_R067_T21MYN_20230520T125249.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20210501T140049_N0500_R067_T21MYN_20230204T001728.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20210521T140049_N0500_R067_T21MYN_20230207T011050.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20210630T140059_N0500_R067_T21MYN_20230203T074708.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20210720T140059_N0500_R067_T21MYN_20230128T210343.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20220903T135709_N0510_R067_T21MYN_20240704T120413.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20240505T135659_N0510_R067_T21MYN_20240505T162431.SAFE_PRODUCT.zip",
    "S2B_MSIL2A_20240604T135709_N0510_R067_T21MYN_20240604T160626.SAFE_PRODUCT.zip",
]

zip_dir = "ml/data/sentinel2_downloads"
download_script = "ml/download_sentinel2_stac.py"

def main():
    for fname in corrupted_files:
        product_id = fname.split(".SAFE_PRODUCT.zip")[0]
        date_match = re.search(r"_(\d{8})T", fname)
        if date_match:
            date_str = date_match.group(1)
            date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            date_range = f"{date_fmt}/{date_fmt}"
            print(f"\nAttempting to download {product_id} for date {date_fmt}...")
            env = os.environ.copy()
            env["DATE_RANGE"] = date_range
            result = subprocess.run(["python3", download_script], env=env)
            if result.returncode == 0:
                print(f"Download script completed for {product_id}.")
            else:
                print(f"Download script failed for {product_id}.")
            time.sleep(2)
        else:
            print(f"Product: {product_id}, Date: UNKNOWN (skipping download)")

if __name__ == "__main__":
    main()
    # Verification step
    print("\nVerifying integrity of newly downloaded files...")
    valid = []
    invalid = []
    for fname in corrupted_files:
        path = os.path.join(zip_dir, fname)
        if not os.path.exists(path):
            print(f"{fname}: MISSING")
            invalid.append(fname)
            continue
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                bad = zf.testzip()
                if bad is not None:
                    print(f"{fname}: CORRUPTED (bad file in archive: {bad})")
                    invalid.append(fname)
                else:
                    print(f"{fname}: OK")
                    valid.append(fname)
        except Exception as e:
            print(f"{fname}: CORRUPTED (exception: {e})")
            invalid.append(fname)
    print(f"\nSummary: {len(valid)} valid, {len(invalid)} missing or corrupted.")
    if invalid:
        print("Problem files:")
        for f in invalid:
            print(f"  - {f}") 