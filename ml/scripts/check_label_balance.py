import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob('ml/data/change_label_patches/*.npy')
ratios = []
sample = files[:1000]
for f in sample:
    arr = np.load(f)
    ratio = (arr > 0).sum() / arr.size
    ratios.append(ratio)
print(f'Checked {len(sample)} patches.')
print(f'Min ratio: {min(ratios):.6f}, Max ratio: {max(ratios):.6f}, Mean: {np.mean(ratios):.6f}')
plt.hist(ratios, bins=50, log=True)
plt.xlabel('Change pixel ratio per patch')
plt.ylabel('Count (log scale)')
plt.title('Histogram of change pixel ratios in label patches')
plt.show() 