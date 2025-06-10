import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

CSV_PATH = 'ml/data/sentinel2_annual_pairs_balanced.csv'
PRED_DIR = 'ml/inference/results/change_detection_preds'
N_SAMPLES = 5
THRESHOLD = 0.9  # Use a higher threshold for binarization

# Helper to plot a single sample
def plot_sample(img1, img2, label, pred, idx, threshold):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(np.moveaxis(img1, 0, -1)[:, :, :3])
    axs[0].set_title('Image 1 (RGB)')
    axs[1].imshow(np.moveaxis(img2, 0, -1)[:, :, :3])
    axs[1].set_title('Image 2 (RGB)')
    axs[2].imshow(label.squeeze(), cmap='gray')
    axs[2].set_title('Ground Truth')
    axs[3].imshow(pred.squeeze(), cmap='gray')
    axs[3].set_title(f'Prediction (thresh={threshold})')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'visualization_{idx}_thresh{threshold}.png')
    plt.show()

def main():
    df = pd.read_csv(CSV_PATH)
    for idx, row in df.head(N_SAMPLES).iterrows():
        img1 = np.load(row['patch1'])
        img2 = np.load(row['patch2'])
        label = np.load(row['label'])
        pred_path = os.path.join(PRED_DIR, f'pred_{idx}.npy')
        if not os.path.exists(pred_path):
            print(f'Prediction not found for sample {idx}')
            continue
        pred = np.load(pred_path)
        # If pred has shape (1, H, W), squeeze channel
        if pred.shape[0] == 1:
            pred = pred[0]
        # Binarize prediction at threshold
        pred_bin = (1 / (1 + np.exp(-pred))) > THRESHOLD
        plot_sample(img1, img2, label, pred_bin, idx, THRESHOLD)

if __name__ == '__main__':
    main() 