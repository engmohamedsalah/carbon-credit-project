#!/usr/bin/env python3
"""
Enhanced training script for U-Net with comprehensive data augmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import time
import argparse

from ml.models.unet import UNet
from ml.training.dataset_patches import PatchDataset
from ml.utils.losses import FocalLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model_enhanced(
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    model_save_path='ml/models/forest_cover_unet_enhanced.pth',
    epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    val_split=0.2,
    use_augmentation=True,
    augment_strength='medium',
    focal_alpha=0.75,
    focal_gamma=2.0,
    patience=10,
    save_best_only=True
):
    """
    Enhanced training function with advanced data augmentation.
    
    Args:
        csv_file: Path to the dataset CSV file
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        val_split: Validation split ratio
        use_augmentation: Whether to use data augmentation
        augment_strength: 'light', 'medium', or 'heavy' augmentation
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        patience: Early stopping patience
        save_best_only: Whether to save only the best model
    """
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded dataset with {len(df)} patches")
    
    # Create datasets with and without augmentation
    full_dataset = PatchDataset(df, augment=False)  # No augmentation for splitting
    train_dataset_no_aug = PatchDataset(df, augment=False)
    train_dataset_aug = PatchDataset(df, augment=use_augmentation, augment_strength=augment_strength)
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    torch.manual_seed(42)  # For reproducible splits
    train_indices, val_indices = random_split(range(dataset_size), [train_size, val_size])
    
    # Create subset datasets
    train_subset_df = df.iloc[train_indices.indices].reset_index(drop=True)
    val_subset_df = df.iloc[val_indices.indices].reset_index(drop=True)
    
    # Create final datasets
    train_dataset = PatchDataset(train_subset_df, augment=use_augmentation, augment_strength=augment_strength)
    val_dataset = PatchDataset(val_subset_df, augment=False)  # No augmentation for validation
    
    logger.info(f"Training set: {len(train_dataset)} patches")
    logger.info(f"Validation set: {len(val_dataset)} patches")
    if use_augmentation:
        logger.info(f"Using {augment_strength} data augmentation")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model
    model = UNet(n_channels=12, n_classes=1)
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info("Starting enhanced training...")
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{train_loss/train_batches:.4f}'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{val_loss/val_batches:.4f}'
                })

        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Val Loss: {avg_val_loss:.6f}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            if save_best_only:
                # Save best model
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"  ✓ New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs (patience: {patience})")
            break
            
        logger.info("")

    # Save final model if not saving best only
    if not save_best_only:
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Final model saved to {model_save_path}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    return {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': epoch + 1,
        'total_time': total_time
    }

def main():
    parser = argparse.ArgumentParser(description='Enhanced U-Net Training with Data Augmentation')
    parser.add_argument('--csv_file', type=str, default='ml/data/forest_cover_patches_balanced.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--model_save_path', type=str, default='ml/models/forest_cover_unet_enhanced.pth',
                       help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augment_strength', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Data augmentation strength')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Train model
    results = train_model_enhanced(
        csv_file=args.csv_file,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        use_augmentation=not args.no_augmentation,
        augment_strength=args.augment_strength,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        patience=args.patience
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Results: {results}")

if __name__ == '__main__':
    main() 