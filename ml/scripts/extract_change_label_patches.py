import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window

PATCH_SIZE = 128
CHANGE_LABEL_DIR = 'ml/data/change_labels'
PATCH_SAVE_DIR = 'ml/data/change_label_patches'
CSV_IN = 'ml/data/sentinel2_annual_pairs.csv'
CSV_OUT = 'ml/data/sentinel2_annual_pairs_with_labels.csv'
os.makedirs(PATCH_SAVE_DIR, exist_ok=True)

def find_label_file(year1, year2):
    # Try to match files like T21MYN_YYYYMMDD_YYYYMMDD_change.tif
    for fname in os.listdir(CHANGE_LABEL_DIR):
        if fname.endswith('_change.tif') and year1 in fname and year2 in fname:
            return os.path.join(CHANGE_LABEL_DIR, fname)
    return None

def main():
    df = pd.read_csv(CSV_IN)
    label_paths = []
    for idx, row in df.iterrows():
        year1 = str(row['year1'])
        year2 = str(row['year2'])
        x = int(row['x'])
        y = int(row['y'])
        label_file = find_label_file(year1, year2)
        if label_file is None:
            label_paths.append('')
            continue
        with rasterio.open(label_file) as src:
            patch = src.read(1, window=Window(x, y, PATCH_SIZE, PATCH_SIZE))
        patch_path = os.path.join(PATCH_SAVE_DIR, f'label_{idx}.npy')
        np.save(patch_path, patch)
        nonzero = np.count_nonzero(patch)
        print(f'Patch {idx} at ({x},{y}) from {label_file}: nonzero={nonzero}, total={patch.size}, ratio={nonzero/patch.size:.6f}')
        label_paths.append(patch_path)
    df['label'] = label_paths
    df.to_csv(CSV_OUT, index=False)
    print(f'Wrote {CSV_OUT} with label column.')

if __name__ == '__main__':
    main() 