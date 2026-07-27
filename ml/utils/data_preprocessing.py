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
        # Fail loudly — never fabricate an analysis from random data.
        raise RuntimeError(f"Failed to load/preprocess image {image_path}: {e}") from e

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
        # Fail loudly — never fabricate an analysis from random data.
        raise RuntimeError(f"Failed to load/preprocess image {image_path}: {e}") from e

def load_full_image(image_path: str,
                    target_channels: int = 12,
                    normalize: bool = True) -> torch.Tensor:
    """Load a satellite image at NATIVE resolution (no crop/resize).

    Unlike load_and_preprocess_image (which center-crops to a fixed 64x64 and so
    only ever sees ~41 ha), this preserves the real spatial extent so area/carbon
    can be computed correctly for a project of any size (see tile_image).

    NOTE: pass normalize=False for the forest-cover U-Net. That checkpoint's
    BatchNorm was trained on RAW reflectance DN, so the [0,1] normalization here
    flatlines it to a constant ~0.5 output (see _tiled_forest_prediction). normalize
    is kept for callers whose models expect [0,1].
    """
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # [C, H, W]
    if image.shape[0] > target_channels:
        image = image[:target_channels]
    elif image.shape[0] < target_channels:
        pad = np.zeros((target_channels - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
        image = np.concatenate([image, pad], axis=0)
    if normalize:
        image = np.clip(image, 0, 10000) / 10000.0
    return torch.from_numpy(image)


def tile_image(tensor: torch.Tensor, tile: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split [C, H, W] into non-overlapping tile x tile patches at native resolution.

    Edge patches are zero-padded to full size; a boolean mask marks the real
    (non-padded) pixels so padding is excluded from area counts.

    Returns:
        patches:   [N, C, tile, tile]
        valid_mask:[N, tile, tile]  (True = real pixel, False = padding)
    """
    C, H, W = tensor.shape
    pad_h = (tile - H % tile) % tile
    pad_w = (tile - W % tile) % tile
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    valid = torch.zeros((Hp, Wp), dtype=torch.bool)
    valid[:H, :W] = True
    patches, masks = [], []
    for i in range(0, Hp, tile):
        for j in range(0, Wp, tile):
            patches.append(padded[:, i:i + tile, j:j + tile])
            masks.append(valid[i:i + tile, j:j + tile])
    return torch.stack(patches), torch.stack(masks)


def create_dummy_image_tensor(channels: int = 4,
                            size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """Create a dummy image tensor for testing"""
    return torch.randn(channels, size[0], size[1])