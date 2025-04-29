# Sample Data and Pre-trained Model Guidance

This document provides guidance on obtaining sample data and pre-trained models for the Carbon Credit Verification SaaS application, focusing on the machine learning components for forest change detection and carbon sequestration estimation.

## 1. Sample Data Sources

### 1.1 Satellite Imagery

#### Sentinel-2 Imagery
- **Description**: Sentinel-2 is a European Space Agency (ESA) satellite mission providing high-resolution optical imagery with 13 spectral bands.
- **Resolution**: 10m, 20m, and 60m (depending on band)
- **Revisit Time**: 5 days (with two satellites)
- **Access Methods**:
  - **Copernicus Open Access Hub**: https://scihub.copernicus.eu/
    - Requires free registration
    - Provides full-resolution imagery
    - API available for automated downloads
  - **Google Earth Engine**: https://earthengine.google.com/
    - Requires Google account
    - Provides cloud-based access and processing
    - Python API available (ee package)
  - **AWS Registry of Open Data**: https://registry.opendata.aws/sentinel-2/
    - No registration required
    - S3 bucket access
    - Requires AWS SDK for efficient access
  - **Microsoft Planetary Computer**: https://planetarycomputer.microsoft.com/
    - Requires Microsoft account
    - STAC API for searching and accessing data
    - Python SDK available

#### Sample Areas for Testing
For dissertation purposes, select 3-5 diverse test areas:
- **Tropical Forest**: e.g., Amazon Basin (Brazil), Congo Basin, Southeast Asia
- **Temperate Forest**: e.g., Pacific Northwest (USA), Central Europe
- **Boreal Forest**: e.g., Northern Canada, Scandinavia, Russia
- **Areas with Known Change**: Select regions with documented deforestation or reforestation

#### Time Periods
- **Baseline Period**: 1-2 years before expected change
- **Monitoring Period**: Recent imagery (within last year)
- **Seasonal Consideration**: Try to match seasons between baseline and monitoring to reduce false positives

### 1.2 Reference Data for Training and Validation

#### Hansen Global Forest Change Dataset
- **Description**: Annual global forest change data from 2000 to present
- **Resolution**: 30m
- **Access**: https://glad.earthengine.app/view/global-forest-change
- **Download**: Available through Google Earth Engine or direct download from University of Maryland
- **Usage in Project**: Primary source for training and validation data
- **Preprocessing Needed**:
  - Resampling to match Sentinel-2 resolution
  - Converting to binary masks (forest/non-forest, change/no-change)
  - Splitting into training/validation sets

#### Global Forest Watch
- **Description**: Platform providing forest monitoring data and tools
- **Access**: https://www.globalforestwatch.org/
- **Usage in Project**: Additional validation data, contextual information

#### Field Validation Data (If Available)
- **Description**: Ground-truth data from field surveys
- **Sources**: Research papers, forestry departments, conservation organizations
- **Usage in Project**: Validation of model predictions

### 1.3 Carbon Sequestration Reference Data

#### IPCC Guidelines
- **Description**: Guidelines for national greenhouse gas inventories
- **Access**: https://www.ipcc-nggip.iges.or.jp/public/2006gl/
- **Usage in Project**: Default values for carbon stock factors

#### Global Forest Biomass
- **Description**: Global maps of forest biomass
- **Sources**: ESA Climate Change Initiative, NASA GEDI
- **Access**: Various research repositories
- **Usage in Project**: Reference for carbon estimation models

## 2. Data Preparation Workflow

### 2.1 Sample Data Preparation Script

The following Python script outlines the process for preparing a small sample dataset:

```python
# sample_data_preparation.py
import os
import rasterio
import numpy as np
import geopandas as gpd
from sentinelsat import SentinelAPI
from datetime import date
import matplotlib.pyplot as plt

# Configuration
USER = 'your_copernicus_username'  # Replace with your credentials
PASSWORD = 'your_copernicus_password'
OUTPUT_DIR = 'sample_data'
AOI_FILE = 'sample_area.geojson'  # GeoJSON defining your area of interest

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'sentinel'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'hansen'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'processed'), exist_ok=True)

# 1. Download Sentinel-2 imagery
def download_sentinel_imagery():
    # Connect to Copernicus Open Access Hub
    api = SentinelAPI(USER, PASSWORD, 'https://scihub.copernicus.eu/dhus')
    
    # Load area of interest
    aoi = gpd.read_file(AOI_FILE)
    
    # Define search criteria
    footprint = None
    if not aoi.empty:
        footprint = aoi.geometry[0]
    
    # Search for Sentinel-2 imagery (baseline period)
    products_baseline = api.query(
        footprint,
        date=('20190101', '20191231'),  # Baseline year (example: 2019)
        platformname='Sentinel-2',
        cloudcoverpercentage=(0, 20)  # Low cloud cover
    )
    
    # Search for Sentinel-2 imagery (monitoring period)
    products_monitoring = api.query(
        footprint,
        date=('20220101', '20221231'),  # Monitoring year (example: 2022)
        platformname='Sentinel-2',
        cloudcoverpercentage=(0, 20)  # Low cloud cover
    )
    
    # Download one image from each period (for sample purposes)
    if products_baseline:
        product_id = list(products_baseline.keys())[0]
        api.download(product_id, directory_path=os.path.join(OUTPUT_DIR, 'sentinel', 'baseline'))
    
    if products_monitoring:
        product_id = list(products_monitoring.keys())[0]
        api.download(product_id, directory_path=os.path.join(OUTPUT_DIR, 'sentinel', 'monitoring'))
    
    print(f"Downloaded Sentinel-2 imagery to {os.path.join(OUTPUT_DIR, 'sentinel')}")

# 2. Download Hansen Global Forest Change data
def download_hansen_data():
    # For dissertation purposes, you might need to manually download this data
    # from Google Earth Engine or University of Maryland website
    # This is a placeholder for the process
    print("Please download Hansen Global Forest Change data manually:")
    print("1. Visit https://glad.earthengine.app/view/global-forest-change")
    print("2. Select your area of interest")
    print("3. Download the data")
    print(f"4. Save to {os.path.join(OUTPUT_DIR, 'hansen')}")
    
    # Alternative: Use Google Earth Engine Python API if you have access
    # This requires additional setup and authentication

# 3. Process the data to create training samples
def process_data_for_training():
    # This is a simplified example - actual implementation would be more complex
    
    # Load Sentinel-2 bands (example for a single image)
    sentinel_dir = os.path.join(OUTPUT_DIR, 'sentinel')
    # Find the downloaded .SAFE directories
    baseline_dirs = [d for d in os.listdir(os.path.join(sentinel_dir, 'baseline')) if d.endswith('.SAFE')]
    monitoring_dirs = [d for d in os.listdir(os.path.join(sentinel_dir, 'monitoring')) if d.endswith('.SAFE')]
    
    if not baseline_dirs or not monitoring_dirs:
        print("Sentinel-2 data not found. Please run download_sentinel_imagery() first.")
        return
    
    # Example: Process a single 10m resolution band (B8 - NIR)
    # In a real implementation, you would process multiple bands and create composites
    baseline_img_path = os.path.join(sentinel_dir, 'baseline', baseline_dirs[0], 'GRANULE', 
                                    os.listdir(os.path.join(sentinel_dir, 'baseline', baseline_dirs[0], 'GRANULE'))[0],
                                    'IMG_DATA', 'R10m', 'T31UFS_20190715T105031_B08_10m.jp2')
    
    monitoring_img_path = os.path.join(sentinel_dir, 'monitoring', monitoring_dirs[0], 'GRANULE',
                                      os.listdir(os.path.join(sentinel_dir, 'monitoring', monitoring_dirs[0], 'GRANULE'))[0],
                                      'IMG_DATA', 'R10m', 'T31UFS_20220710T105031_B08_10m.jp2')
    
    # Load Hansen data (placeholder - you would need to adapt this to your actual data)
    hansen_treecover_path = os.path.join(OUTPUT_DIR, 'hansen', 'treecover2000.tif')
    hansen_lossyear_path = os.path.join(OUTPUT_DIR, 'hansen', 'lossyear.tif')
    
    # Create training patches
    # This is highly simplified - in reality, you would:
    # 1. Ensure all rasters are in the same CRS and resolution
    # 2. Create a multiband image from Sentinel-2 bands
    # 3. Create binary masks from Hansen data
    # 4. Split into training patches
    # 5. Apply augmentation
    
    print("Data processing complete. Training samples created.")
    
    # Visualize a sample (placeholder)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Baseline Image (NIR Band)")
    # plt.imshow(baseline_img)
    plt.subplot(122)
    plt.title("Forest Loss Mask")
    # plt.imshow(forest_loss_mask)
    plt.savefig(os.path.join(OUTPUT_DIR, 'processed', 'sample_visualization.png'))

if __name__ == "__main__":
    download_sentinel_imagery()
    download_hansen_data()
    process_data_for_training()
```

### 2.2 Data Volume Considerations

For a dissertation project, consider these data volume guidelines:

- **Training Dataset**:
  - 500-1000 image patches (256×256 pixels) is often sufficient for proof-of-concept
  - Balance between forest change and no-change samples
  - Include diverse forest types and change patterns

- **Validation Dataset**:
  - 100-200 image patches, separate from training data
  - Representative of different conditions

- **Test Areas**:
  - 3-5 complete project areas (e.g., 10km × 10km)
  - Different forest types and geographic regions

- **Storage Requirements**:
  - Raw Sentinel-2 images: ~1GB per scene
  - Processed training data: ~2-5GB
  - Hansen reference data: ~500MB per region
  - Total: ~20-30GB for a reasonable sample dataset

## 3. Pre-trained Models

### 3.1 Using Existing Pre-trained Models

For dissertation purposes, leveraging existing pre-trained models can save significant time:

#### Forest Change Detection Models

1. **U-Net with ImageNet Pre-trained Encoder**
   - **Description**: U-Net architecture with ResNet/EfficientNet backbone pre-trained on ImageNet
   - **Adaptation Needed**: Fine-tune on forest change detection task
   - **Code Example**:
   ```python
   import segmentation_models_pytorch as smp
   
   # Create U-Net model with pre-trained encoder
   model = smp.Unet(
       encoder_name="resnet34",        # Use pre-trained ResNet34
       encoder_weights="imagenet",     # Use weights from ImageNet
       in_channels=4,                  # 4 input channels (RGB + NIR)
       classes=1,                      # Binary segmentation (change/no-change)
   )
   ```

2. **Existing Land Cover Models**
   - **Sources**:
     - [Sentinel Hub Custom Scripts](https://custom-scripts.sentinel-hub.com/)
     - [eo-learn](https://github.com/sentinel-hub/eo-learn) examples
     - [TorchGeo](https://github.com/microsoft/torchgeo)
   - **Adaptation Needed**: Transfer learning to adapt to forest change detection

#### Carbon Estimation Models

1. **Allometric Equations**
   - **Description**: Not ML models, but established equations relating forest parameters to carbon content
   - **Sources**: IPCC Guidelines, scientific literature
   - **Implementation**: Python functions applying these equations to forest change maps

2. **Random Forest Regression Models**
   - **Description**: Models relating spectral indices and texture features to biomass/carbon
   - **Training**: Requires reference biomass data
   - **Sources**: Research papers, may need to train your own

### 3.2 Training Your Own Models

If you choose to train your own models from scratch:

#### Recommended Model Architecture for Forest Change Detection

```python
# model_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 512)
        )

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        x = self.outc(x)
        return self.sigmoid(x)
```

#### Training Script Outline

```python
# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model_architecture import UNet
from dataset import ForestChangeDataset  # You would need to implement this

# Configuration
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'models/forest_change_unet.pth'

# Create model directory
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
])

# Create datasets and dataloaders
train_dataset = ForestChangeDataset(
    data_dir='sample_data/processed/train',
    transform=transform
)

val_dataset = ForestChangeDataset(
    data_dir='sample_data/processed/val',
    transform=transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize model, loss function, and optimizer
model = UNet(n_channels=4, n_classes=1).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}')
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch: {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'Model saved to {MODEL_SAVE_PATH}')

print('Training complete!')
```

### 3.3 Model Sharing and Distribution

For your dissertation project, consider these options for sharing models:

1. **GitHub Repository**:
   - Store model architecture code
   - Include training scripts
   - Provide detailed documentation
   - **Note**: Avoid committing large model weight files directly to GitHub

2. **Model Weights Distribution**:
   - **Small Models** (<100MB): Can be shared via GitHub Releases
   - **Medium Models** (100MB-1GB): Consider Hugging Face Model Hub, Google Drive, or Dropbox
   - **Large Models** (>1GB): Use specialized platforms like Zenodo or academic repositories

3. **Model Card Documentation**:
   - Create a model card documenting:
     - Model architecture
     - Training data description
     - Performance metrics
     - Limitations and biases
     - Intended use cases
     - Example usage code

## 4. Integration with the SaaS Application

### 4.1 Model Serving

For the dissertation project, a simple file-based model loading approach is sufficient:

```python
# backend/app/services/ml_service.py
import os
import torch
import numpy as np
from PIL import Image
import rasterio
from model_architecture import UNet  # Import your model architecture

class ForestChangePredictor:
    def __init__(self, model_path='models/forest_change_unet.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        if self.model is None:
            self.model = UNet(n_channels=4, n_classes=1)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
    
    def process_satellite_image(self, image_path, output_dir):
        """Process a satellite image to detect forest change."""
        self.load_model()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the satellite image (simplified example)
        with rasterio.open(image_path) as src:
            # Read 4 bands (RGB + NIR)
            image = src.read([1, 2, 3, 4])
            meta = src.meta.copy()
        
        # Normalize the image
        image = image.astype(np.float32)
        image = (image - np.array([0.485, 0.456, 0.406, 0.5])[:, np.newaxis, np.newaxis]) / np.array([0.229, 0.224, 0.225, 0.2])[:, np.newaxis, np.newaxis]
        
        # Process in patches if the image is large
        patch_size = 256
        stride = 128
        height, width = image.shape[1], image.shape[2]
        
        # Create an empty prediction mask
        prediction = np.zeros((1, height, width), dtype=np.float32)
        counts = np.zeros((1, height, width), dtype=np.float32)
        
        # Process image in patches with overlap
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image[:, y:y+patch_size, x:x+patch_size]
                
                # Convert to PyTorch tensor
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    output = self.model(patch_tensor)
                
                # Add to prediction mask
                output_np = output.squeeze().cpu().numpy()
                prediction[:, y:y+patch_size, x:x+patch_size] += output_np
                counts[:, y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping predictions
        prediction = prediction / np.maximum(counts, 1)
        
        # Threshold the prediction (e.g., change > 0.5)
        change_mask = (prediction > 0.5).astype(np.uint8)
        
        # Save the prediction mask
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0
        })
        
        change_mask_path = os.path.join(output_dir, 'forest_change_mask.tif')
        with rasterio.open(change_mask_path, 'w', **meta) as dst:
            dst.write(change_mask)
        
        # Calculate statistics
        total_pixels = change_mask.size
        change_pixels = np.sum(change_mask)
        change_percentage = (change_pixels / total_pixels) * 100
        
        # Estimate affected area (simplified)
        pixel_area_m2 = 100  # Assuming 10m resolution (10m x 10m = 100m²)
        affected_area_m2 = change_pixels * pixel_area_m2
        affected_area_ha = affected_area_m2 / 10000  # Convert to hectares
        
        # Create a simple visualization
        rgb = image[[2, 1, 0], :, :]  # Extract RGB bands
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to 0-1
        
        # Create a color overlay
        vis_image = np.zeros((3, height, width), dtype=np.float32)
        vis_image[0, :, :] = rgb[0, :, :]  # Red channel
        vis_image[1, :, :] = rgb[1, :, :]  # Green channel
        vis_image[2, :, :] = rgb[2, :, :]  # Blue channel
        
        # Overlay change mask in red
        vis_image[0, :, :] = np.maximum(vis_image[0, :, :], change_mask[0, :, :] * 0.7)
        vis_image[1, :, :] = vis_image[1, :, :] * (1 - change_mask[0, :, :] * 0.7)
        vis_image[2, :, :] = vis_image[2, :, :] * (1 - change_mask[0, :, :] * 0.7)
        
        # Save visualization
        vis_image = (vis_image * 255).astype(np.uint8)
        vis_image = np.transpose(vis_image, (1, 2, 0))
        vis_path = os.path.join(output_dir, 'forest_change_visualization.png')
        Image.fromarray(vis_image).save(vis_path)
        
        # Return results
        results = {
            'change_mask_path': change_mask_path,
            'visualization_path': vis_path,
            'change_percentage': float(change_percentage),
            'affected_area_ha': float(affected_area_ha),
            'statistics': {
                'total_pixels': int(total_pixels),
                'change_pixels': int(change_pixels)
            }
        }
        
        return results
```

### 4.2 Sample Data for Testing

For testing the complete SaaS application, prepare a small set of sample projects:

1. **Demo Project 1: Amazon Deforestation**
   - Location: Brazilian Amazon (e.g., Rondônia state)
   - Time Period: 2018-2022
   - Expected Change: Significant deforestation
   - Files: Pre-processed Sentinel-2 images, GeoJSON boundary

2. **Demo Project 2: European Reforestation**
   - Location: Central Europe (e.g., Germany)
   - Time Period: 2017-2022
   - Expected Change: Moderate reforestation
   - Files: Pre-processed Sentinel-2 images, GeoJSON boundary

3. **Demo Project 3: Boreal Forest Fire Recovery**
   - Location: Canada or Siberia
   - Time Period: 2016-2022
   - Expected Change: Fire damage followed by recovery
   - Files: Pre-processed Sentinel-2 images, GeoJSON boundary

Package these sample projects in a way that allows easy import into the application for demonstration purposes.

## 5. Conclusion

This guide provides a framework for obtaining and preparing sample data and models for the Carbon Credit Verification SaaS application. For a dissertation project, focus on:

1. **Quality over Quantity**: A few well-chosen sample areas with clear forest change patterns
2. **Documentation**: Thoroughly document data sources, preprocessing steps, and model architecture
3. **Reproducibility**: Ensure your data preparation and model training processes are reproducible
4. **Integration**: Demonstrate how the ML components integrate with the broader SaaS application

Remember that the goal is to demonstrate the concept and methodology, not to build a production-ready system with global coverage. The sample data and models should be sufficient to showcase the application's capabilities and support your research findings.
