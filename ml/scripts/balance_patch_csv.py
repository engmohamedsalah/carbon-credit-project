import pandas as pd
import numpy as np
import os
import random

CSV_IN = 'ml/data/sentinel2_annual_pairs_with_labels.csv'
CSV_OUT = 'ml/data/sentinel2_annual_pairs_balanced.csv'

# Read the CSV
print(f'Reading {CSV_IN}...')
df = pd.read_csv(CSV_IN)

# Identify positive and negative samples
positive_indices = []
negative_indices = []
for idx, row in df.iterrows():
    label_path = row['label']
    if not isinstance(label_path, str) or not os.path.exists(label_path):
        continue
    arr = np.load(label_path)
    if np.count_nonzero(arr) > 0:
        positive_indices.append(idx)
    else:
        negative_indices.append(idx)

print(f'Found {len(positive_indices)} positive and {len(negative_indices)} negative samples.')

# Randomly sample negatives to match positives
random.seed(42)
num_pos = len(positive_indices)
if num_pos == 0:
    raise ValueError('No positive samples found!')
if len(negative_indices) < num_pos:
    print('Warning: Not enough negatives to match positives, using all negatives.')
    selected_negatives = negative_indices
else:
    selected_negatives = random.sample(negative_indices, num_pos)

# Combine and shuffle
balanced_indices = positive_indices + selected_negatives
random.shuffle(balanced_indices)

# Write new CSV
balanced_df = df.loc[balanced_indices].reset_index(drop=True)
balanced_df.to_csv(CSV_OUT, index=False)
print(f'Wrote balanced CSV to {CSV_OUT} with {len(balanced_df)} samples.') 