#!/usr/bin/env python3
"""
Fast ConvLSTM Training Script
Optimized for large datasets with efficient sampling and processing
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
import gc
from datetime import datetime
from pathlib import Path
import time
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.convlstm_model import ConvLSTM
from ml.training.enhanced_time_series_dataset import MultiModalTimeSeriesDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FAST configuration optimized for large datasets
CONFIG = {
    # Data paths
    'data_dir': 'ml/data/prepared',
    's2_stacks_dir': 'ml/data/prepared/s2_stacks',
    's1_stacks_dir': 'ml/data/prepared/s1_stacks',
    'change_labels_dir': 'ml/data/prepared/change_labels',
    'model_save_dir': 'ml/models',
    
    # Optimal sequence configuration
    'seq_length': 3,
    'temporal_gap_days': 45,
    'patch_size': 64,  # REDUCED from 128 for 4x faster processing
    
    # Model architecture - smaller for faster training
    'input_channels': 4,
    'hidden_channels': [32, 64],  # REDUCED from [64, 64, 128]
    'kernel_sizes': [(3, 3), (3, 3)],
    'num_layers': 2,  # REDUCED from 3
    'output_channels': 1,
    
    # Training parameters - optimized for speed
    'batch_size': 8,  # INCREASED for efficiency
    'num_epochs': 15,  # REDUCED for faster completion
    'learning_rate': 0.002,  # INCREASED for faster convergence
    'weight_decay': 1e-4,
    'val_split': 0.2,
    'num_workers': 4,  # INCREASED for faster data loading
    
    # Data settings
    'use_augmentation': False,  # DISABLED for speed
    'min_cloud_free_ratio': 0.7,
    
    # CRITICAL: Limit samples for manageable training time
    'max_samples_per_epoch': 2000,  # Only 2000 samples per epoch
    'max_val_samples': 500,  # Only 500 validation samples
    
    # Early stopping
    'patience': 5,
    'min_delta': 0.001,
    'save_every_n_epochs': 3,
}

class FastDataset:
    """Wrapper to limit dataset size for faster training"""
    def __init__(self, dataset, max_samples=None):
        self.dataset = dataset
        self.max_samples = max_samples or len(dataset)
        self.indices = list(range(min(len(dataset), self.max_samples)))
        random.shuffle(self.indices)  # Random sampling
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def calculate_metrics(outputs, targets, threshold=0.5):
    """Fast metrics calculation"""
    try:
        with torch.no_grad():
            predictions = torch.sigmoid(outputs) > threshold
            targets_bool = targets > threshold
            
            tp = (predictions & targets_bool).sum().float()
            fp = (predictions & ~targets_bool).sum().float()
            fn = (~predictions & targets_bool).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            return {
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item()
            }
    except Exception as e:
        logger.warning(f"Metrics calculation failed: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Fast training epoch"""
    model.train()
    total_loss = 0.0
    all_metrics = {'precision': [], 'recall': [], 'f1': []}
    
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.squeeze(1).float().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(sequences)
        loss = criterion(outputs.squeeze(1), labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics every 10 batches for speed
        if batch_idx % 10 == 0:
            metrics = calculate_metrics(outputs.squeeze(1), labels)
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # Progress logging
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Memory cleanup
        del sequences, labels, outputs, loss
        if batch_idx % 20 == 0:
            gc.collect()
    
    # Average metrics
    avg_metrics = {key: np.mean(values) if values else 0.0 for key, values in all_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def validate_epoch(model, dataloader, criterion, device):
    """Fast validation epoch"""
    model.eval()
    total_loss = 0.0
    all_metrics = {'precision': [], 'recall': [], 'f1': []}
    
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(device)
            labels = labels.squeeze(1).float().to(device)
            
            outputs, _ = model(sequences)
            loss = criterion(outputs.squeeze(1), labels)
            
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs.squeeze(1), labels)
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Memory cleanup
            del sequences, labels, outputs, loss
    
    # Average metrics
    avg_metrics = {key: np.mean(values) if values else 0.0 for key, values in all_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def main():
    """Fast training main function"""
    logger.info("🚀 Starting FAST ConvLSTM Training")
    logger.info("=" * 60)
    
    # Print configuration
    logger.info("📋 FAST Training Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"   {key}: {value}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")
    
    # Create directories
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    
    try:
        # Create dataset
        logger.info("📊 Creating datasets...")
        
        full_dataset = MultiModalTimeSeriesDataset(
            s2_stacks_dir=CONFIG['s2_stacks_dir'],
            s1_stacks_dir=CONFIG['s1_stacks_dir'],
            change_labels_dir=CONFIG['change_labels_dir'],
            seq_length=CONFIG['seq_length'],
            patch_size=CONFIG['patch_size'],
            temporal_gap_days=CONFIG['temporal_gap_days'],
            min_cloud_free_ratio=CONFIG['min_cloud_free_ratio'],
            use_augmentation=CONFIG['use_augmentation']
        )
        
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty!")
        
        logger.info(f"📊 Full dataset size: {len(full_dataset)} samples")
        
        # Create fast limited datasets
        train_dataset = FastDataset(full_dataset, CONFIG['max_samples_per_epoch'])
        val_dataset = FastDataset(full_dataset, CONFIG['max_val_samples'])
        
        logger.info(f"📊 Limited dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"✅ Dataloaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Initialize smaller model
        logger.info("🏗️  Initializing FAST ConvLSTM model...")
        model = ConvLSTM(
            input_dim=CONFIG['input_channels'],
            hidden_dim=CONFIG['hidden_channels'],
            kernel_size=CONFIG['kernel_sizes'],
            num_layers=CONFIG['num_layers'],
            output_dim=CONFIG['output_channels'],
            batch_first=True,
            bias=True
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,} (much smaller for speed!)")
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'epochs_completed': 0
        }
        
        best_f1 = 0.0
        patience_counter = 0
        
        # Training loop
        logger.info("🎯 Starting FAST training loop...")
        start_time = time.time()
        
        for epoch in range(CONFIG['num_epochs']):
            epoch_start = time.time()
            logger.info(f"\n📅 Epoch {epoch+1}/{CONFIG['num_epochs']}")
            logger.info("-" * 40)
            
            # Train
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}, Epoch Time: {epoch_time:.1f}s")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            history['epochs_completed'] = epoch + 1
            
            # Check for improvement
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(CONFIG['model_save_dir'], 'convlstm_fast_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_f1': val_metrics['f1'],
                    'config': CONFIG
                }, best_model_path)
                logger.info(f"💾 New best model saved! F1: {val_metrics['f1']:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= CONFIG['patience']:
                logger.info(f"⏹️  Early stopping after {epoch+1} epochs")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % CONFIG['save_every_n_epochs'] == 0:
                checkpoint_path = os.path.join(CONFIG['model_save_dir'], f'convlstm_fast_checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'config': CONFIG
                }, checkpoint_path)
                logger.info(f"📁 Checkpoint saved: {checkpoint_path}")
            
            # Memory cleanup
            gc.collect()
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"\n🎉 FAST Training completed in {total_time/60:.1f} minutes!")
        
        # Save final results
        final_model_path = os.path.join(CONFIG['model_save_dir'], 'convlstm_fast_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'history': history,
            'total_training_time': total_time,
            'best_f1': best_f1
        }, final_model_path)
        
        # Save training history
        history_path = os.path.join(CONFIG['model_save_dir'], 'convlstm_fast_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
                 logger.info(f"📊 Best validation F1: {best_f1:.4f}")
         best_model_path = os.path.join(CONFIG['model_save_dir'], 'convlstm_fast_best.pth')
         logger.info(f"💾 Best model: {best_model_path}")
         logger.info(f"💾 Final model: {final_model_path}")
         logger.info(f"📈 Training history: {history_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ FAST ConvLSTM training completed successfully!")
    else:
        logger.error("❌ FAST ConvLSTM training failed!")
        sys.exit(1) 