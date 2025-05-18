import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Paths (update if needed)
s2_stack_path = "data/prepared/s2_stacks/S2A_MSIL2A_20220710T135721_N0510_R067_T21MYN_20240703T190212.SAFE_stack.tif"
label_path = "data/prepared/change_labels/20220710_change_label.tif"
quicklook_dir = "data/prepared/quicklooks"
os.makedirs(quicklook_dir, exist_ok=True)

# 1. RGB Composite (B04=red, B03=green, B02=blue)
with rasterio.open(s2_stack_path) as src:
    bands = src.read()
    # Assume order is [B02, B03, B04, B08]
    b2, b3, b4 = bands[0], bands[1], bands[2]
    rgb = np.stack([b4, b3, b2], axis=-1)
    # Normalize to 0-1 for display
    rgb_min, rgb_max = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb_disp = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
    plt.figure(figsize=(6,6))
    plt.imshow(rgb_disp)
    plt.axis('off')
    plt.title('Sentinel-2 RGB Composite')
    plt.savefig(os.path.join(quicklook_dir, 's2_rgb.png'), bbox_inches='tight', dpi=150)
    plt.close()

# 2. NDVI (B08 - B04) / (B08 + B04)
    b8 = bands[3]
    ndvi = (b8.astype(float) - b4.astype(float)) / (b8 + b4 + 1e-8)
    plt.figure(figsize=(6,6))
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI')
    plt.axis('off')
    plt.savefig(os.path.join(quicklook_dir, 's2_ndvi.png'), bbox_inches='tight', dpi=150)
    plt.close()

# 3. Overlay label on RGB
with rasterio.open(label_path) as lbl_src:
    label = lbl_src.read(1)
    # Resize label to match RGB if needed (should match)
    overlay = rgb_disp.copy()
    # Red overlay where label==1
    overlay[label == 1] = [1, 0, 0]
    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Change Label Overlay')
    plt.savefig(os.path.join(quicklook_dir, 'label_overlay.png'), bbox_inches='tight', dpi=150)
    plt.close()

print(f"Quicklooks saved to {quicklook_dir}") 