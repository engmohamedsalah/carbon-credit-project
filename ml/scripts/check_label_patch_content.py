import numpy as np
import pandas as pd

df = pd.read_csv('ml/data/sentinel2_annual_pairs_with_labels.csv')
for i, p in enumerate(df['label'].head(20)):
    arr = np.load(p)
    print(f'{p}: nonzero={np.count_nonzero(arr)}, total={arr.size}') 