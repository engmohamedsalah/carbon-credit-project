#!/usr/bin/env python3
"""
Simple data preprocessing utilities for ensemble model
"""

import torch
import numpy as np
import rasterio
from typing import Tuple, Optional

def load_and_preprocess_image(image_path: str, 
                            target_size: Tuple[int, int] = (64, 64),
                            normalize: bool = True) -> torch.Tensor:
    """
    Load and preprocess a satellite image for model inference
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width)
        normalize: Whether to normalize the image
    
    Returns:
        Preprocessed image tensor [C, H, W]
    """
    try:
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()  # Shape: [C, H, W]
            
            # Convert to float32
            image = image.astype(np.float32)
            
            # Handle different number of channels - default to 12 for forest model compatibility
            target_channels = 12
            if image.shape[0] > target_channels:
                # Take first target_channels bands if more
                image = image[:target_channels]
            elif image.shape[0] < target_channels:
                # Pad with zeros if less than target_channels
                padding = np.zeros((target_channels - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=0)
            
            # Resize if needed (simple nearest neighbor)
            if image.shape[1:] != target_size:
                # For simplicity, just crop/pad to target size
                h, w = image.shape[1], image.shape[2]
                target_h, target_w = target_size
                
                # Center crop or pad
                if h > target_h:
                    start_h = (h - target_h) // 2
                    image = image[:, start_h:start_h + target_h, :]
                elif h < target_h:
                    pad_h = target_h - h
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant')
                
                if w > target_w:
                    start_w = (w - target_w) // 2
                    image = image[:, :, start_w:start_w + target_w]
                elif w < target_w:
                    pad_w = target_w - w
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    image = np.pad(image, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
            
            # Normalize if requested
            if normalize:
                # Simple normalization to [0, 1] range
                image = np.clip(image, 0, 10000)  # Clip extreme values
                image = image / 10000.0
            
            # Convert to torch tensor
            tensor = torch.from_numpy(image)
            
            return tensor
            
    except Exception as e:
        # If loading fails, create a dummy tensor
        print(f"Warning: Could not load {image_path}, creating dummy tensor: {e}")
        return torch.randn(4, target_size[0], target_size[1])

def load_and_preprocess_image_for_model(image_path: str, 
                                       model_type: str = 'forest',
                                       target_size: Tuple[int, int] = (64, 64),
                                       normalize: bool = True) -> torch.Tensor:
    """
    Load and preprocess image for specific model type
    
    Args:
        image_path: Path to the image file
        model_type: 'forest' (12 channels) or 'change' (4 channels)
        target_size: Target size (height, width)
        normalize: Whether to normalize the image
    
    Returns:
        Preprocessed image tensor [C, H, W]
    """
    target_channels = 12 if model_type == 'forest' else 4
    
    try:
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()  # Shape: [C, H, W]
            
            # Convert to float32
            image = image.astype(np.float32)
            
            # Handle different number of channels
            if image.shape[0] > target_channels:
                # Take first target_channels bands if more
                image = image[:target_channels]
            elif image.shape[0] < target_channels:
                # Pad with zeros if less than target_channels
                padding = np.zeros((target_channels - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=0)
            
            # Resize if needed (simple nearest neighbor)
            if image.shape[1:] != target_size:
                # For simplicity, just crop/pad to target size
                h, w = image.shape[1], image.shape[2]
                target_h, target_w = target_size
                
                # Center crop or pad
                if h > target_h:
                    start_h = (h - target_h) // 2
                    image = image[:, start_h:start_h + target_h, :]
                elif h < target_h:
                    pad_h = target_h - h
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    image = np.pad(image, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant')
                
                if w > target_w:
                    start_w = (w - target_w) // 2
                    image = image[:, :, start_w:start_w + target_w]
                elif w < target_w:
                    pad_w = target_w - w
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    image = np.pad(image, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
            
            # Normalize if requested
            if normalize:
                # Simple normalization to [0, 1] range
                image = np.clip(image, 0, 10000)  # Clip extreme values
                image = image / 10000.0
            
            # Convert to torch tensor
            tensor = torch.from_numpy(image)
            
            return tensor
            
    except Exception as e:
        # If loading fails, create a dummy tensor
        print(f"Warning: Could not load {image_path}, creating dummy tensor: {e}")
        return torch.randn(target_channels, target_size[0], target_size[1])

def create_dummy_image_tensor(channels: int = 4, 
                            size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """Create a dummy image tensor for testing"""
    return torch.randn(channels, size[0], size[1]) 