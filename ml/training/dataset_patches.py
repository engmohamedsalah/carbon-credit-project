# ml/training/dataset_patches.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2

def custom_jitter(image_tensor, brightness_factor=0.3, contrast_factor=0.3):
    """
    Applies brightness and contrast jitter to a 12-channel image tensor.
    """
    # Adjust brightness
    brightness_delta = (torch.rand(1) * 2 - 1) * brightness_factor
    image_tensor = image_tensor + brightness_delta

    # Adjust contrast
    mean = torch.mean(image_tensor, dim=(1, 2), keepdim=True)
    contrast_mult = 1.0 + (torch.rand(1) * 2 - 1) * contrast_factor
    image_tensor = (image_tensor - mean) * contrast_mult + mean
    
    return torch.clamp(image_tensor, 0, 1) # Clamp to valid range

def custom_gaussian_blur(image_tensor, kernel_size=3, sigma_min=0.1, sigma_max=2.0):
    """
    Applies Gaussian blur to a 12-channel image tensor.
    """
    sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()
    # Create a Gaussian kernel
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*np.pi*variance)) * torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    # Repeat for all 12 channels
    gaussian_kernel = gaussian_kernel.repeat(12, 1, 1, 1)
    
    # Apply blur
    padding = (kernel_size - 1) // 2
    image_tensor = F.conv2d(image_tensor.unsqueeze(0), weight=gaussian_kernel, padding=padding, groups=12).squeeze(0)
    return image_tensor

def add_spectral_noise(image_tensor, noise_factor=0.05):
    """
    Adds channel-specific noise to simulate atmospheric variations.
    """
    # Different noise levels for different spectral bands
    # Higher noise for atmospheric windows, lower for stable bands
    noise_levels = torch.tensor([0.03, 0.04, 0.05, 0.03, 0.04, 0.06, 0.07, 0.03, 0.04, 0.08, 0.05, 0.06])
    noise_levels = noise_levels * noise_factor
    
    for i in range(12):
        noise = torch.randn_like(image_tensor[i]) * noise_levels[i]
        image_tensor[i] = image_tensor[i] + noise
    
    return torch.clamp(image_tensor, 0, 1)

def channel_dropout(image_tensor, dropout_prob=0.1, max_channels=3):
    """
    Randomly zero out some spectral channels to simulate sensor issues.
    """
    if random.random() < dropout_prob:
        n_channels_to_drop = random.randint(1, max_channels)
        channels_to_drop = random.sample(range(12), n_channels_to_drop)
        for ch in channels_to_drop:
            image_tensor[ch] = 0.0
    return image_tensor

def spectral_band_shuffle(image_tensor, shuffle_prob=0.1):
    """
    Randomly shuffle similar spectral bands to add robustness.
    """
    if random.random() < shuffle_prob:
        # Define groups of similar bands that can be shuffled
        # Visible: [1, 2, 3] (Blue, Green, Red)
        # NIR: [4, 7] (B05, B8A) 
        # SWIR: [10, 11] (B11, B12)
        
        shuffle_groups = [
            [1, 2, 3],  # Visible bands
            [4, 7],     # NIR variants
            [10, 11]    # SWIR bands
        ]
        
        for group in shuffle_groups:
            if len(group) > 1 and random.random() < 0.5:
                shuffled_indices = group.copy()
                random.shuffle(shuffled_indices)
                original_data = image_tensor[group].clone()
                for i, original_idx in enumerate(group):
                    new_idx = shuffled_indices[i]
                    image_tensor[original_idx] = original_data[group.index(new_idx)]
    
    return image_tensor

def elastic_deformation(image_tensor, label_tensor, alpha=1.0, sigma=50.0, random_state=None):
    """
    Apply elastic deformation to both image and label tensors.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image_tensor.shape[1:]  # (H, W)
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Apply to image (all 12 channels)
    deformed_image = torch.zeros_like(image_tensor)
    for c in range(12):
        channel_data = image_tensor[c].numpy()
        deformed_channel = map_coordinates(channel_data, indices, order=1, mode='reflect')
        deformed_image[c] = torch.from_numpy(deformed_channel.reshape(shape))
    
    # Apply to label
    label_data = label_tensor.squeeze().numpy()
    deformed_label = map_coordinates(label_data, indices, order=0, mode='reflect')  # order=0 for labels
    deformed_label = torch.from_numpy(deformed_label.reshape(shape)).unsqueeze(0)
    
    return deformed_image, deformed_label

def random_crop_and_resize(image_tensor, label_tensor, scale_range=(0.8, 1.0)):
    """
    Random crop and resize back to original size.
    """
    c, h, w = image_tensor.shape
    scale = random.uniform(*scale_range)
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Random crop coordinates
    top = random.randint(0, h - new_h) if new_h < h else 0
    left = random.randint(0, w - new_w) if new_w < w else 0
    
    # Crop
    image_crop = image_tensor[:, top:top+new_h, left:left+new_w]
    label_crop = label_tensor[:, top:top+new_h, left:left+new_w]
    
    # Resize back to original size
    image_resized = F.interpolate(image_crop.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
    label_resized = F.interpolate(label_crop.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
    
    return image_resized, label_resized

def mixup_augmentation(image1, label1, image2, label2, alpha=0.2):
    """
    Apply mixup augmentation between two samples.
    """
    if random.random() < 0.5:  # 50% chance to apply mixup
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_image, mixed_label
    return image1, label1


class PatchDataset(Dataset):
    """Dataset for loading single image patches and their corresponding masks."""
    def __init__(self, data, transform=None, augment=False, augment_strength='medium'):
        """
        Args:
            data (str or pd.DataFrame): Path to CSV or a pandas DataFrame.
                                        CSV must contain 'image_path' and 'label_path' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool, optional): Whether to apply data augmentation.
            augment_strength (str): 'light', 'medium', or 'heavy' augmentation intensity.
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("data must be a filepath string or a pandas DataFrame")

        self.transform = transform
        self.augment = augment
        self.augment_strength = augment_strength
        
        # Set augmentation probabilities based on strength
        if augment_strength == 'light':
            self.aug_probs = {
                'jitter': 0.3, 'blur': 0.2, 'noise': 0.2, 'dropout': 0.05,
                'shuffle': 0.05, 'elastic': 0.1, 'crop': 0.2, 'flip': 0.5, 'rot': 0.3
            }
        elif augment_strength == 'medium':
            self.aug_probs = {
                'jitter': 0.5, 'blur': 0.3, 'noise': 0.3, 'dropout': 0.1,
                'shuffle': 0.1, 'elastic': 0.2, 'crop': 0.3, 'flip': 0.5, 'rot': 0.5
            }
        else:  # heavy
            self.aug_probs = {
                'jitter': 0.7, 'blur': 0.5, 'noise': 0.5, 'dropout': 0.15,
                'shuffle': 0.15, 'elastic': 0.3, 'crop': 0.4, 'flip': 0.5, 'rot': 0.7
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label_path = row['label_path']

        image = np.load(image_path)  # (C, H, W)
        label = np.load(label_path)  # (H, W)

        # Ensure label is single-channel if it has multiple
        if label.ndim == 3 and label.shape[0] > 1:
            label = label[0, :, :] # Take the first channel

        # Convert to torch tensors and normalize label
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float().unsqueeze(0) / 100.0 # Normalize to [0, 1]

        # Apply augmentation
        if self.augment:
            # Spectral augmentations (applied to image only)
            if random.random() < self.aug_probs['jitter']:
                image = custom_jitter(image, brightness_factor=0.2, contrast_factor=0.2)
            
            if random.random() < self.aug_probs['noise']:
                image = add_spectral_noise(image, noise_factor=0.03)
            
            if random.random() < self.aug_probs['dropout']:
                image = channel_dropout(image, dropout_prob=1.0, max_channels=2)
            
            if random.random() < self.aug_probs['shuffle']:
                image = spectral_band_shuffle(image, shuffle_prob=1.0)
            
            if random.random() < self.aug_probs['blur']:
                image = custom_gaussian_blur(image, sigma_min=0.1, sigma_max=1.5)

            # Geometric augmentations (applied to both image and label)
            if random.random() < self.aug_probs['elastic']:
                try:
                    image, label = elastic_deformation(image, label, alpha=0.5, sigma=20.0)
                except:
                    pass  # Skip if elastic deformation fails
            
            if random.random() < self.aug_probs['crop']:
                image, label = random_crop_and_resize(image, label, scale_range=(0.85, 1.0))
            
            # Standard geometric transforms
            if random.random() < self.aug_probs['flip']:
                image = T.functional.hflip(image)
                label = T.functional.hflip(label)
            if random.random() < self.aug_probs['flip']:
                image = T.functional.vflip(image)
                label = T.functional.vflip(label)
            
            # Random 90-degree rotation
            if random.random() < self.aug_probs['rot']:
                k = random.randint(1, 3)
                image = torch.rot90(image, k, dims=[1, 2])
                label = torch.rot90(label, k, dims=[1, 2])

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label 