import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import rasterio
from rasterio.windows import Window
import glob
import logging
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PATCH_SIZE = 256
BATCH_SIZE = 4  # Reduced batch size
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ForestChangePreparedDataset(Dataset):
    """Dataset for forest cover change detection using prepared Sentinel-2 stacks and change labels."""
    def __init__(self, stacks_dir, labels_dir, transform=None, mode='train', val_split=0.2, random_seed=42):
        self.stacks_dir = stacks_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.mode = mode

        # Find all stack and label files
        stack_files = sorted(glob.glob(os.path.join(stacks_dir, '*.tif')))
        label_files = sorted(glob.glob(os.path.join(labels_dir, '*.tif')))

        def extract_key(f):
            base = os.path.basename(f)
            match = re.search(r'(\d{8})', base)
            if match:
                return match.group(1)
            return base.split('_')[0]

        stack_keys = {extract_key(f): f for f in stack_files}
        label_keys = {extract_key(f): f for f in label_files}
        common_keys = sorted(set(stack_keys.keys()) & set(label_keys.keys()))
        self.pairs = [(stack_keys[k], label_keys[k], k) for k in common_keys]

        # Split into train/val
        if mode != 'test':
            train_pairs, val_pairs = train_test_split(self.pairs, test_size=val_split, random_state=random_seed)
            self.pairs = train_pairs if mode == 'train' else val_pairs
        logger.info(f"Loaded {len(self.pairs)} stack/label pairs for {mode}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stack_path, label_path, key = self.pairs[idx]
        
        # Read stack in chunks
        with rasterio.open(stack_path) as src:
            height, width = src.height, src.width
            # Ensure patch fits within image
            if width < PATCH_SIZE or height < PATCH_SIZE:
                raise ValueError(f"Image {stack_path} is smaller than patch size {PATCH_SIZE}")
            x = np.random.randint(0, width - PATCH_SIZE + 1)
            y = np.random.randint(0, height - PATCH_SIZE + 1)
            window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
            image = src.read(window=window).astype(np.float32)
            
        # Read label in the same window
        with rasterio.open(label_path) as src:
            label = src.read(1, window=window).astype(np.int64)

        # Convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'scene_id': key}

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation of satellite imagery."""
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet, self).__init__()
        # Encoder (downsampling)
        self.enc1_conv = self._double_conv(in_channels, 64)
        self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_conv = self._double_conv(64, 128)
        self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3_conv = self._double_conv(128, 256)
        self.enc3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4_conv = self._double_conv(256, 512)
        self.enc4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # Decoder (upsampling at the start of each block)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, enc_feat, target_shape):
        _, _, h, w = enc_feat.shape
        target_h, target_w = target_shape
        crop_h = max((h - target_h) // 2, 0)
        crop_w = max((w - target_w) // 2, 0)
        return enc_feat[:, :, crop_h:crop_h+target_h, crop_w:crop_w+target_w]

    def forward(self, x):
        # Encoder
        enc1 = self.enc1_conv(x)
        enc2 = self.enc2_conv(self.enc1_pool(enc1))
        enc3 = self.enc3_conv(self.enc2_pool(enc2))
        enc4 = self.enc4_conv(self.enc3_pool(enc3))
        # Bottleneck
        bottleneck = self.bottleneck(self.enc4_pool(enc4))
        # Decoder
        up4 = self.up4(bottleneck)
        enc4_cropped = self.center_crop(enc4, up4.shape[2:])
        dec4 = self.dec4(torch.cat([up4, enc4_cropped], dim=1))
        up3 = self.up3(dec4)
        enc3_cropped = self.center_crop(enc3, up3.shape[2:])
        dec3 = self.dec3(torch.cat([up3, enc3_cropped], dim=1))
        up2 = self.up2(dec3)
        enc2_cropped = self.center_crop(enc2, up2.shape[2:])
        dec2 = self.dec2(torch.cat([up2, enc2_cropped], dim=1))
        up1 = self.up1(dec2)
        enc1_cropped = self.center_crop(enc1, up1.shape[2:])
        dec1 = self.dec1(torch.cat([up1, enc1_cropped], dim=1))
        out = self.final(dec1)
        return out

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
    """Train the forest change detection model with memory-efficient practices."""
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
            
            # Clear memory
            del images, labels, outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
                
                # Clear memory
                del images, labels, outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        val_loss = val_loss / len(val_loader.dataset)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    """Main function to train the forest change detection model with prepared data."""
    # Set up data directories
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'prepared')
    stacks_dir = os.path.join(base_dir, 's2_stacks')
    labels_dir = os.path.join(base_dir, 'change_labels')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
    ])

    # Create datasets and data loaders with reduced batch size and num_workers
    train_dataset = ForestChangePreparedDataset(stacks_dir, labels_dir, transform=transform, mode='train')
    val_dataset = ForestChangePreparedDataset(stacks_dir, labels_dir, transform=transform, mode='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,  # Reduced number of workers
        pin_memory=True  # Enable pin memory for faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,  # Reduced number of workers
        pin_memory=True  # Enable pin memory for faster data transfer to GPU
    )

    # Initialize model, loss function, and optimizer
    sample_stack = next(iter(train_dataset))['image']
    in_channels = sample_stack.shape[0]
    model = UNet(in_channels=in_channels, out_channels=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    model_save_path = os.path.join(model_dir, 'forest_change_unet.pth')
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, model_save_path)
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
