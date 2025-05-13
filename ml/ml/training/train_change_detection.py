# ml/training/train_change_detection.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Assuming siamese_unet model is defined in models.siamese_unet
from ..models.siamese_unet import SiameseUNet

# Configure logging
logging.basicConfig(level=logging.INFO, format=\"%(asctime)s - %(name)s - %(levelname)s - %(message)s\")
logger = logging.getLogger(__name__)

# Constants
SENTINEL_BANDS = [\"B02\", \"B03\", \"B04\", \"B08\"]  # Blue, Green, Red, NIR
PATCH_SIZE = 256
BATCH_SIZE = 8 # Reduced batch size for Siamese network due to potentially higher memory usage
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
DATA_DIR = os.path.join(os.path.dirname(__file__), \"..\", \"data\")
MODEL_DIR = os.path.join(os.path.dirname(__file__), \"..\", \"models\")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, \"change_detection_siamese_unet.pth\")

class ChangeDetectionDataset(Dataset):
    \"\"\"Dataset for change detection using pairs of Sentinel-2 imagery.\"\"\"

    def __init__(self, data_dir, transform=None, mode=\"train\"):
        \"\"\"
        Args:
            data_dir (str): Directory containing paired Sentinel-2 images and change labels.
                            Expected structure:
                            data_dir/
                                sentinel_t1/scene1/B02.tif, ...
                                sentinel_t2/scene1/B02.tif, ...
                                change_labels/scene1_change.tif
            transform (callable, optional): Optional transform to be applied on image pairs.
            mode (str): \"train\", \"val\", or \"test\".
        \"\"\"
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        self.scenes = self._get_scenes()

        if mode != \"test\":
            train_scenes, val_scenes = train_test_split(self.scenes, test_size=0.2, random_state=42)
            self.scenes = train_scenes if mode == \"train\" else val_scenes

        logger.info(f\"Loaded {len(self.scenes)} scene pairs for {mode}\")

    def _get_scenes(self):
        \"\"\"Get list of available scene pairs with corresponding change labels.\"\"\"
        scenes_t1 = glob.glob(os.path.join(self.data_dir, \"sentinel_t1\", \"*\"))
        change_labels = glob.glob(os.path.join(self.data_dir, \"change_labels\", \"*_change.tif\"))

        scene_pairs = []
        for scene_t1_path in scenes_t1:
            scene_id = os.path.basename(scene_t1_path)
            scene_t2_path = os.path.join(self.data_dir, \"sentinel_t2\", scene_id)
            label_path = os.path.join(self.data_dir, \"change_labels\", f\"{scene_id}_change.tif\")

            if os.path.exists(scene_t2_path) and os.path.exists(label_path):
                scene_pairs.append({
                    \"t1_path\": scene_t1_path,
                    \"t2_path\": scene_t2_path,
                    \"label_path\": label_path,
                    \"scene_id\": scene_id
                })
        return scene_pairs

    def __len__(self):
        # Assuming 10 patches per scene pair for simplicity
        return len(self.scenes) * 10

    def _load_patch(self, scene_path, patch_idx):
        \"\"\"Loads a specific patch from a Sentinel-2 scene directory.\"\"\"
        bands = []
        profile = None
        for band in SENTINEL_BANDS:
            band_path = os.path.join(scene_path, f\"{band}.tif\")
            try:
                with rasterio.open(band_path) as src:
                    if profile is None: profile = src.profile
                    height, width = src.height, src.width
                    # Simple patching strategy - needs improvement for real-world use
                    row_offset = (patch_idx // 3) * PATCH_SIZE
                    col_offset = (patch_idx % 3) * PATCH_SIZE
                    row_offset = min(row_offset, height - PATCH_SIZE)
                    col_offset = min(col_offset, width - PATCH_SIZE)
                    window = Window(col_offset, row_offset, PATCH_SIZE, PATCH_SIZE)
                    band_data = src.read(1, window=window)
                    # Handle potential nodata values if necessary (e.g., fill with 0)
                    # band_data = np.nan_to_num(band_data)
                    bands.append(band_data)
            except rasterio.RasterioIOError as e:
                logger.error(f\"Error reading {band_path}: {e}\")
                return None # Return None if a band file is missing or corrupt

        if len(bands) != len(SENTINEL_BANDS):
             return None # Ensure all bands were loaded

        image = np.stack(bands, axis=0)
        return torch.from_numpy(image).float()

    def __getitem__(self, idx):
        scene_idx = idx // 10
        patch_idx = idx % 10
        scene_pair = self.scenes[scene_idx]

        img_t1 = self._load_patch(scene_pair[\"t1_path\"], patch_idx)
        img_t2 = self._load_patch(scene_pair[\"t2_path\"], patch_idx)

        # Load change label patch
        label = None
        try:
            with rasterio.open(scene_pair[\"label_path\"]) as src:
                height, width = src.height, src.width
                row_offset = (patch_idx // 3) * PATCH_SIZE
                col_offset = (patch_idx % 3) * PATCH_SIZE
                row_offset = min(row_offset, height - PATCH_SIZE)
                col_offset = min(col_offset, width - PATCH_SIZE)
                window = Window(col_offset, row_offset, PATCH_SIZE, PATCH_SIZE)
                label_data = src.read(1, window=window)
                # Ensure label is binary (0: No Change, 1: Change)
                label_data = (label_data > 0).astype(np.int64)
                label = torch.from_numpy(label_data).long()
        except rasterio.RasterioIOError as e:
            logger.error(f\"Error reading label {scene_pair[\"label_path\"]}: {e}\")
            # Handle error - maybe return None or skip this sample

        # Handle cases where loading failed
        if img_t1 is None or img_t2 is None or label is None:
            # Return dummy data or skip - simple approach: return None
            # A better approach in DataLoader is to filter these out
            logger.warning(f\"Skipping sample {idx} due to loading errors.\")
            # To make DataLoader work, return valid tensors even if empty/dummy
            # This requires careful handling in the training loop or a custom collate_fn
            # For simplicity here, we might raise an error or return placeholder tensors
            # Let's return zero tensors of expected shape
            img_t1 = torch.zeros((len(SENTINEL_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=torch.float)
            img_t2 = torch.zeros((len(SENTINEL_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=torch.float)
            label = torch.zeros((PATCH_SIZE, PATCH_SIZE), dtype=torch.long)

        sample = {\"image_t1\": img_t1, \"image_t2\": img_t2, \"label\": label, \"scene_id\": scene_pair[\"scene_id\"]}

        # Apply transformations if any (applied to both images)
        # Note: Transforms like normalization should be applied carefully to paired images
        if self.transform:
            # Assuming transform handles a dictionary or apply individually
            # This needs careful implementation of the transform function
            # For simplicity, let's assume transform applies to each image tensor
            sample[\"image_t1\"] = self.transform(sample[\"image_t1\"])
            sample[\"image_t2\"] = self.transform(sample[\"image_t2\"])

        return sample

def train_change_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    \"\"\"Train the change detection model.\"\"\"
    best_val_loss = float(\"inf\")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_samples = 0

        for batch in tqdm(train_loader, desc=f\"Epoch {epoch+1}/{num_epochs} - Training\"):
            img1 = batch[\"image_t1\"].to(DEVICE)
            img2 = batch[\"image_t2\"].to(DEVICE)
            labels = batch[\"label\"].to(DEVICE)

            # Skip batch if it contains dummy data (simple check)
            if torch.sum(img1) == 0 and torch.sum(img2) == 0:
                logger.warning(\"Skipping dummy batch\")
                continue

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * img1.size(0)
            num_samples += img1.size(0)

        if num_samples > 0:
            train_loss /= num_samples
        else:
            train_loss = 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f\"Epoch {epoch+1}/{num_epochs} - Validation\"):
                img1 = batch[\"image_t1\"].to(DEVICE)
                img2 = batch[\"image_t2\"].to(DEVICE)
                labels = batch[\"label\"].to(DEVICE)

                if torch.sum(img1) == 0 and torch.sum(img2) == 0:
                    continue

                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * img1.size(0)
                num_val_samples += img1.size(0)

        if num_val_samples > 0:
            val_loss /= num_val_samples
        else:
            val_loss = float('inf') # Assign high loss if no valid validation samples

        logger.info(f\"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\")

        if val_loss < best_val_loss and num_val_samples > 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f\"Saved best model with validation loss: {val_loss:.4f}\")

def main():
    logger.info(f\"Using device: {DEVICE}\")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define transformations (Example: normalization - adapt mean/std if needed)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
    ])

    # Create datasets and data loaders
    # Note: Ensure your data_dir has sentinel_t1, sentinel_t2, and change_labels subdirs
    try:
        train_dataset = ChangeDetectionDataset(DATA_DIR, transform=transform, mode=\"train\")
        val_dataset = ChangeDetectionDataset(DATA_DIR, transform=transform, mode=\"val\")
    except FileNotFoundError as e:
        logger.error(f\"Data directory structure error: {e}. Ensure sentinel_t1, sentinel_t2, and change_labels subdirectories exist in {DATA_DIR}\")
        return

    # Consider a custom collate_fn to filter out None samples if __getitem__ returns None
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = SiameseUNet(in_channels=len(SENTINEL_BANDS), out_channels=2).to(DEVICE) # 2 classes: No Change, Change
    # Use weights for class imbalance if necessary
    # weight = torch.tensor([0.5, 2.0]).to(DEVICE) # Example: Higher weight for 'Change' class
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info(\"Starting training...\")
    train_change_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH)
    logger.info(f\"Training finished. Best model saved to {MODEL_SAVE_PATH}\")

if __name__ == \"__main__\":
    main()


