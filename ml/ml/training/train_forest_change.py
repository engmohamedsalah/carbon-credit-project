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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SENTINEL_BANDS = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
PATCH_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ForestChangeDataset(Dataset):
    """Dataset for forest cover change detection using Sentinel-2 imagery."""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Args:
            data_dir (str): Directory with Sentinel-2 images and Hansen data
            transform (callable, optional): Optional transform to be applied on a sample
            mode (str): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # Get list of all available scenes
        self.scenes = self._get_scenes()
        
        # Split data for train/val/test
        if mode != 'test':
            train_scenes, val_scenes = train_test_split(self.scenes, test_size=0.2, random_state=42)
            self.scenes = train_scenes if mode == 'train' else val_scenes
        
        logger.info(f"Loaded {len(self.scenes)} scenes for {mode}")
    
    def _get_scenes(self):
        """Get list of all available scenes with both Sentinel-2 and Hansen data."""
        # This is a placeholder - in a real implementation, you would scan your data directory
        # and match Sentinel-2 scenes with corresponding Hansen data
        
        # For demonstration, we'll assume a directory structure like:
        # data_dir/
        #   sentinel/
        #     scene1/
        #       B02.tif, B03.tif, B04.tif, B08.tif
        #     scene2/
        #       ...
        #   hansen/
        #     scene1_forest_change.tif
        #     scene2_forest_change.tif
        
        sentinel_scenes = glob.glob(os.path.join(self.data_dir, 'sentinel', '*'))
        hansen_files = glob.glob(os.path.join(self.data_dir, 'hansen', '*_forest_change.tif'))
        
        # Match scenes with labels
        scenes = []
        for scene_path in sentinel_scenes:
            scene_id = os.path.basename(scene_path)
            hansen_path = os.path.join(self.data_dir, 'hansen', f"{scene_id}_forest_change.tif")
            
            if os.path.exists(hansen_path):
                scenes.append({
                    'sentinel_path': scene_path,
                    'hansen_path': hansen_path,
                    'scene_id': scene_id
                })
        
        return scenes
    
    def __len__(self):
        return len(self.scenes) * 10  # Each scene will generate multiple patches
    
    def __getitem__(self, idx):
        # Determine which scene and which patch within the scene
        scene_idx = idx // 10
        patch_idx = idx % 10
        
        scene = self.scenes[scene_idx]
        
        # Load Sentinel-2 bands
        bands = []
        for band in SENTINEL_BANDS:
            band_path = os.path.join(scene['sentinel_path'], f"{band}.tif")
            with rasterio.open(band_path) as src:
                # For simplicity, we'll just take a fixed window for each patch
                # In a real implementation, you would use a more sophisticated approach
                height, width = src.height, src.width
                row_offset = (patch_idx // 3) * PATCH_SIZE
                col_offset = (patch_idx % 3) * PATCH_SIZE
                
                # Ensure we don't go out of bounds
                if row_offset + PATCH_SIZE > height or col_offset + PATCH_SIZE > width:
                    row_offset = max(0, height - PATCH_SIZE)
                    col_offset = max(0, width - PATCH_SIZE)
                
                window = Window(col_offset, row_offset, PATCH_SIZE, PATCH_SIZE)
                band_data = src.read(1, window=window)
                bands.append(band_data)
        
        # Stack bands to create a multi-channel image
        image = np.stack(bands, axis=0)
        
        # Load Hansen forest change data for the same patch
        with rasterio.open(scene['hansen_path']) as src:
            window = Window(col_offset, row_offset, PATCH_SIZE, PATCH_SIZE)
            label = src.read(1, window=window)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'label': label, 'scene_id': scene['scene_id']}

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation of satellite imagery."""
    
    def __init__(self, in_channels=4, out_channels=2):
        """
        Args:
            in_channels (int): Number of input channels (4 for Sentinel-2: B, G, R, NIR)
            out_channels (int): Number of output classes (2 for binary forest/non-forest)
        """
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._encoder_block(in_channels, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._bottleneck_block(512, 1024)
        
        # Decoder (upsampling)
        self.dec4 = self._decoder_block(1024 + 512, 512)
        self.dec3 = self._decoder_block(512 + 256, 256)
        self.dec2 = self._decoder_block(256 + 128, 128)
        self.dec1 = self._decoder_block(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _bottleneck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([bottleneck, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        
        # Final layer
        return self.final(dec1)

def calculate_ndvi(red, nir):
    """Calculate Normalized Difference Vegetation Index."""
    return (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero

def preprocess_sentinel2(image_path, output_path=None):
    """
    Preprocess Sentinel-2 imagery for forest change detection.
    
    Args:
        image_path (str): Path to Sentinel-2 image (directory with band files)
        output_path (str, optional): Path to save preprocessed image
    
    Returns:
        np.ndarray: Preprocessed image with 5 channels (B, G, R, NIR, NDVI)
    """
    # Load bands
    bands = {}
    for band in SENTINEL_BANDS:
        band_path = os.path.join(image_path, f"{band}.tif")
        with rasterio.open(band_path) as src:
            bands[band] = src.read(1)
            profile = src.profile  # Save profile for writing output
    
    # Calculate NDVI
    ndvi = calculate_ndvi(bands['B04'], bands['B08'])
    
    # Stack bands
    preprocessed = np.stack([
        bands['B02'],  # Blue
        bands['B03'],  # Green
        bands['B04'],  # Red
        bands['B08'],  # NIR
        ndvi           # NDVI
    ], axis=0)
    
    # Normalize bands to [0, 1]
    for i in range(4):  # Don't normalize NDVI which is already in [-1, 1]
        band_min, band_max = preprocessed[i].min(), preprocessed[i].max()
        preprocessed[i] = (preprocessed[i] - band_min) / (band_max - band_min + 1e-8)
    
    # Save preprocessed image if output_path is provided
    if output_path:
        # Update profile for the new stacked image
        profile.update(count=5, dtype=rasterio.float32)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(preprocessed.astype(rasterio.float32))
    
    return preprocessed

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    """Train the forest change detection model."""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

def main():
    """Main function to train the forest change detection model."""
    # Set up data directories
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
    ])
    
    # Create datasets and data loaders
    train_dataset = ForestChangeDataset(data_dir, transform=transform, mode='train')
    val_dataset = ForestChangeDataset(data_dir, transform=transform, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=4, out_channels=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    model_save_path = os.path.join(model_dir, 'forest_change_unet.pth')
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, model_save_path)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
