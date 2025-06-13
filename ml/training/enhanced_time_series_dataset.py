# ml/training/enhanced_time_series_dataset.py

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import glob
from datetime import datetime, timedelta
import re
import json
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalTimeSeriesDataset(Dataset):
    """
    Enhanced dataset for ConvLSTM training combining Sentinel-1 and Sentinel-2 data.
    Creates temporal sequences with multi-class change labels.
    """
    
    def __init__(self, 
                 s2_stacks_dir,
                 s1_stacks_dir=None,
                 change_labels_dir=None,
                 seq_length=6,
                 patch_size=128,
                 temporal_gap_days=30,
                 min_cloud_free_ratio=0.7,
                 use_augmentation=True):
        """
        Args:
            s2_stacks_dir: Directory with processed Sentinel-2 stacks
            s1_stacks_dir: Directory with processed Sentinel-1 stacks (optional)
            change_labels_dir: Directory with change labels
            seq_length: Number of time steps in sequence
            patch_size: Size of patches to extract
            temporal_gap_days: Minimum days between consecutive images
            min_cloud_free_ratio: Minimum cloud-free ratio for patches
            use_augmentation: Whether to apply data augmentation
        """
        self.s2_stacks_dir = s2_stacks_dir
        self.s1_stacks_dir = s1_stacks_dir
        self.change_labels_dir = change_labels_dir
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.temporal_gap_days = temporal_gap_days
        self.min_cloud_free_ratio = min_cloud_free_ratio
        self.use_augmentation = use_augmentation
        
        # Load and organize temporal data
        self._load_temporal_inventory()
        self._create_temporal_sequences()
        self._generate_patch_samples()
        
        logger.info(f"Dataset initialized with {len(self.sequences)} temporal sequences")
        logger.info(f"Total patch samples: {self.total_patches}")
    
    def _load_temporal_inventory(self):
        """Load and organize all available imagery by date"""
        self.s2_inventory = {}
        self.s1_inventory = {}
        self.change_inventory = {}
        
        # Sentinel-2 inventory
        s2_files = glob.glob(os.path.join(self.s2_stacks_dir, "*_stack.tif"))
        for f in s2_files:
            date = self._extract_date_from_filename(f)
            if date:
                self.s2_inventory[date] = f
        
        # Sentinel-1 inventory (if available)
        if self.s1_stacks_dir and os.path.exists(self.s1_stacks_dir):
            s1_files = glob.glob(os.path.join(self.s1_stacks_dir, "*_stack.tif"))
            for f in s1_files:
                date = self._extract_date_from_filename(f)
                if date:
                    self.s1_inventory[date] = f
        
        # Change labels inventory
        if self.change_labels_dir and os.path.exists(self.change_labels_dir):
            change_files = glob.glob(os.path.join(self.change_labels_dir, "*_change_label.tif"))
            for f in change_files:
                # Extract date from change label filename (use the date from the original S2 filename)
                date = self._extract_date_from_filename(f)
                if date:
                    self.change_inventory[date] = f
        
        logger.info(f"Found {len(self.s2_inventory)} S2 images, {len(self.s1_inventory)} S1 images, {len(self.change_inventory)} change labels")
    
    def _extract_date_from_filename(self, filename):
        """Extract date from various filename formats"""
        patterns = [
            r'(\d{8})T\d{6}',  # YYYYMMDDTHHMMSS
            r'(\d{8})',        # YYYYMMDD
            r'(\d{4})-(\d{2})-(\d{2})'  # YYYY-MM-DD
        ]
        
        basename = os.path.basename(filename)
        for pattern in patterns:
            match = re.search(pattern, basename)
            if match:
                if len(match.groups()) == 1:
                    date_str = match.group(1)
                    if len(date_str) == 8:
                        return datetime.strptime(date_str, '%Y%m%d')
                else:
                    year, month, day = match.groups()
                    return datetime(int(year), int(month), int(day))
        return None
    
    def _extract_date_range_from_change_label(self, filename):
        """Extract start and end dates from change label filename"""
        # Format: T21MYN_20200117_20230829_change.tif
        pattern = r'(\d{8})_(\d{8})_change'
        match = re.search(pattern, os.path.basename(filename))
        if match:
            start_str, end_str = match.groups()
            start_date = datetime.strptime(start_str, '%Y%m%d')
            end_date = datetime.strptime(end_str, '%Y%m%d')
            return (start_date, end_date)
        return None
    
    def _create_temporal_sequences(self):
        """Create valid temporal sequences for training"""
        self.sequences = []
        
        # Get all available dates and sort them
        all_dates = sorted(set(self.s2_inventory.keys()) | set(self.s1_inventory.keys()))
        
        # Create sequences with flexible temporal gaps
        for i in range(len(all_dates) - self.seq_length + 1):
            sequence_dates = all_dates[i:i+self.seq_length]
            
            # Check if sequence has acceptable temporal gaps
            valid_sequence = True
            for j in range(1, len(sequence_dates)):
                gap = (sequence_dates[j] - sequence_dates[j-1]).days
                if gap > self.temporal_gap_days:
                    valid_sequence = False
                    break
            
            if not valid_sequence:
                continue
            
            # Only use sequences with required length
            if len(sequence_dates) == self.seq_length:
                # Check data availability - only require S2 data
                s2_available = all(date in self.s2_inventory for date in sequence_dates)
                
                if s2_available:
                    # Find appropriate change label for sequence end date
                    end_date = sequence_dates[-1]
                    change_label_file = self._find_change_label_for_date(end_date)
                    
                    # Only include sequences that have change labels
                    if change_label_file:
                        sequence_info = {
                            'dates': sequence_dates,
                            's2_files': [self.s2_inventory[date] for date in sequence_dates],
                            's1_files': [self.s1_inventory.get(date) for date in sequence_dates] if self.s1_inventory else None,
                            'change_label': change_label_file
                        }
                        self.sequences.append(sequence_info)
        
        logger.info(f"Created {len(self.sequences)} valid temporal sequences")
    
    def _find_change_label_for_date(self, target_date):
        """Find the most appropriate change label for a target date"""
        if not self.change_inventory:
            return None
        
        # Direct match first
        if target_date in self.change_inventory:
            return self.change_inventory[target_date]
        
        # Find closest date if no exact match
        closest_date = None
        min_diff = float('inf')
        
        for date in self.change_inventory.keys():
            diff = abs((date - target_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_date = date
        
        if closest_date and min_diff <= 30:  # Within 30 days
            return self.change_inventory[closest_date]
        
        return None
    
    def _generate_patch_samples(self):
        """Generate patch sampling locations"""
        if not self.sequences:
            self.total_patches = 0
            return
        
        # Use first sequence to determine image dimensions
        first_s2_file = self.sequences[0]['s2_files'][0]
        with rasterio.open(first_s2_file) as src:
            self.height = src.height
            self.width = src.width
        
        # Calculate patch grid
        self.patches_per_row = self.height // self.patch_size
        self.patches_per_col = self.width // self.patch_size
        self.patches_per_image = self.patches_per_row * self.patches_per_col
        self.total_patches = len(self.sequences) * self.patches_per_image
        
        logger.info(f"Image dimensions: {self.height}x{self.width}")
        logger.info(f"Patch grid: {self.patches_per_row}x{self.patches_per_col}")
        logger.info(f"Patches per sequence: {self.patches_per_image}")
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        # Determine sequence and patch indices
        seq_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Get patch coordinates
        patch_row = (patch_idx // self.patches_per_col) * self.patch_size
        patch_col = (patch_idx % self.patches_per_col) * self.patch_size
        window = Window(patch_col, patch_row, self.patch_size, self.patch_size)
        
        sequence_info = self.sequences[seq_idx]
        
        # Load sequence data
        sequence_data = self._load_sequence_patch(sequence_info, window)
        
        # Load label
        label_data = self._load_label_patch(sequence_info, window)
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            sequence_data, label_data = self._apply_augmentation(sequence_data, label_data)
        
        return sequence_data, label_data
    
    def _load_sequence_patch(self, sequence_info, window):
        """Load patch data for entire sequence"""
        sequence_patches = []
        
        for i in range(self.seq_length):
            s2_file = sequence_info['s2_files'][i]
            s1_file = sequence_info['s1_files'][i] if sequence_info['s1_files'] else None
            
            # Load Sentinel-2 data (B02, B03, B04, B08)
            with rasterio.open(s2_file) as src:
                s2_patch = src.read([1, 2, 3, 4], window=window)  # R, G, B, NIR
                s2_patch = s2_patch.astype(np.float32) / 10000.0  # Scale to 0-1
            
            # Load Sentinel-1 data if available (VV, VH)
            if s1_file and os.path.exists(s1_file):
                with rasterio.open(s1_file) as src:
                    s1_patch = src.read([1, 2], window=window)  # VV, VH
                    s1_patch = s1_patch.astype(np.float32)
                    # Convert from dB to linear scale and normalize
                    s1_patch = np.power(10, s1_patch / 10.0)
                    s1_patch = np.clip(s1_patch / 100.0, 0, 1)  # Normalize
                
                # Combine S1 and S2 data
                combined_patch = np.concatenate([s2_patch, s1_patch], axis=0)
            else:
                combined_patch = s2_patch
            
            sequence_patches.append(torch.from_numpy(combined_patch))
        
        # Stack into sequence tensor: (seq_len, channels, height, width)
        sequence_tensor = torch.stack(sequence_patches)
        
        return sequence_tensor
    
    def _load_label_patch(self, sequence_info, window):
        """Load change label patch"""
        if sequence_info['change_label'] and os.path.exists(sequence_info['change_label']):
            with rasterio.open(sequence_info['change_label']) as src:
                label_patch = src.read(1, window=window)
                label_patch = label_patch.astype(np.int64)
                
                # Convert to multi-class labels
                label_patch = self._convert_to_multiclass_labels(label_patch)
                
                return torch.from_numpy(label_patch)
        else:
            # Return dummy label if no change label available
            return torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)
    
    def _convert_to_multiclass_labels(self, binary_labels):
        """Convert binary change labels to multi-class"""
        # For now, simple conversion:
        # 0: No change, 1: Change
        # TODO: Implement more sophisticated labeling based on temporal patterns
        return binary_labels
    
    def _apply_augmentation(self, sequence_data, label_data):
        """Apply data augmentation"""
        if not self.use_augmentation:
            return sequence_data, label_data
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            sequence_data = torch.flip(sequence_data, dims=[-1])
            label_data = torch.flip(label_data, dims=[-1])
        
        # Random vertical flip
        if torch.rand(1) > 0.5:
            sequence_data = torch.flip(sequence_data, dims=[-2])
            label_data = torch.flip(label_data, dims=[-2])
        
        # Random rotation (90, 180, 270 degrees)
        rotation_angle = torch.randint(0, 4, (1,)).item()
        if rotation_angle > 0:
            for _ in range(rotation_angle):
                sequence_data = torch.rot90(sequence_data, dims=[-2, -1])
                label_data = torch.rot90(label_data, dims=[-2, -1])
        
        return sequence_data, label_data

def create_time_series_dataloaders(s2_stacks_dir,
                                  s1_stacks_dir=None,
                                  change_labels_dir=None,
                                  seq_length=6,
                                  patch_size=128,
                                  batch_size=8,
                                  val_split=0.2,
                                  num_workers=4):
    """Create train and validation dataloaders"""
    
    # Create full dataset
    full_dataset = MultiModalTimeSeriesDataset(
        s2_stacks_dir=s2_stacks_dir,
        s1_stacks_dir=s1_stacks_dir,
        change_labels_dir=change_labels_dir,
        seq_length=seq_length,
        patch_size=patch_size,
        use_augmentation=True
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create validation dataset without augmentation
    val_dataset.dataset.use_augmentation = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test dataset creation
    s2_dir = "ml/data/prepared/s2_stacks"
    s1_dir = "ml/data/prepared/s1_stacks"
    change_dir = "ml/data/change_labels"
    
    dataset = MultiModalTimeSeriesDataset(
        s2_stacks_dir=s2_dir,
        s1_stacks_dir=s1_dir,
        change_labels_dir=change_dir,
        seq_length=5,
        patch_size=64
    )
    
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample_data, sample_label = dataset[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label shape: {sample_label.shape}") 