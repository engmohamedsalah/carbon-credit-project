import os
import numpy as np

PATCH_DIR = 'ml/data/change_label_patches'

group_counts = {'all_zero': 0, 'some_change': 0, 'mostly_change': 0}
total = 0

for fname in os.listdir(PATCH_DIR):
    if fname.endswith('.npy'):
        path = os.path.join(PATCH_DIR, fname)
        arr = np.load(path)
        nonzero = np.count_nonzero(arr)
        total_pixels = arr.size
        ratio = nonzero / total_pixels
        if nonzero == 0:
            group_counts['all_zero'] += 1
        elif ratio > 0.95:
            group_counts['mostly_change'] += 1
        else:
            group_counts['some_change'] += 1
        total += 1
        print(f'{fname}: nonzero={nonzero}, total={total_pixels}, ratio={ratio:.6f}')

print('\nSummary:')
for group, count in group_counts.items():
    percent = 100 * count / total if total > 0 else 0
    print(f'{group}: {count} patches ({percent:.2f}%)')
print(f'Total patches: {total}') 