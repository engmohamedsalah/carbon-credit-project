#!/usr/bin/env python3
"""
Comprehensive ConvLSTM Model Evaluation
Detailed analysis of model performance, data distribution, and recommendations
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.convlstm_model import ConvLSTM
from ml.training.enhanced_time_series_dataset import MultiModalTimeSeriesDataset

def load_model(model_path):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    model = ConvLSTM(
        input_dim=config['input_channels'],
        hidden_dim=config['hidden_channels'],
        kernel_size=config['kernel_sizes'],
        num_layers=config['num_layers'],
        output_dim=config['output_channels'],
        batch_first=True,
        bias=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, checkpoint

def analyze_training_history(history_path):
    """Analyze training history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print("📈 TRAINING HISTORY ANALYSIS:")
    print("-" * 40)
    print(f"Epochs completed: {history['epochs_completed']}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Loss reduction: {history['train_loss'][0]:.6f} → {history['train_loss'][-1]:.6f}")
    print(f"Loss improvement: {((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.1f}%")
    
    return history

def comprehensive_prediction_analysis(model, dataloader, num_batches=20):
    """Comprehensive analysis of model predictions"""
    model.eval()
    
    all_probabilities = []
    all_targets = []
    batch_losses = []
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("🔍 ANALYZING MODEL PREDICTIONS...")
    print("-" * 40)
    
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            outputs, _ = model(sequences)
            probabilities = torch.sigmoid(outputs.squeeze(1))
            targets = labels.squeeze(1)
            
            # Calculate batch loss
            loss = criterion(outputs.squeeze(1), targets.float())
            batch_losses.append(loss.item())
            
            all_probabilities.extend(probabilities.flatten().numpy())
            all_targets.extend(targets.flatten().numpy())
            
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                      f"Prob range=[{probabilities.min():.4f}, {probabilities.max():.4f}]")
    
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    return all_probabilities, all_targets, batch_losses

def calculate_metrics_at_thresholds(probabilities, targets, thresholds):
    """Calculate metrics at multiple thresholds"""
    results = {}
    
    for threshold in thresholds:
        predictions = probabilities > threshold
        targets_bool = targets > 0.5
        
        tp = np.sum((predictions == 1) & (targets_bool == 1))
        fp = np.sum((predictions == 1) & (targets_bool == 0))
        fn = np.sum((predictions == 0) & (targets_bool == 1))
        tn = np.sum((predictions == 0) & (targets_bool == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(targets)
        
        results[threshold] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'positive_predictions': np.sum(predictions),
            'negative_predictions': np.sum(~predictions)
        }
    
    return results

def analyze_data_distribution(targets, probabilities):
    """Analyze the data distribution"""
    print("\n📊 DATA DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    total_samples = len(targets)
    positive_samples = np.sum(targets > 0.5)
    negative_samples = total_samples - positive_samples
    
    print(f"Total samples analyzed: {total_samples:,}")
    print(f"Positive samples: {positive_samples:,} ({positive_samples/total_samples*100:.4f}%)")
    print(f"Negative samples: {negative_samples:,} ({negative_samples/total_samples*100:.4f}%)")
    
    # Probability distribution
    print(f"\nProbability Statistics:")
    print(f"Mean: {np.mean(probabilities):.6f}")
    print(f"Median: {np.median(probabilities):.6f}")
    print(f"Std: {np.std(probabilities):.6f}")
    print(f"Min: {np.min(probabilities):.6f}")
    print(f"Max: {np.max(probabilities):.6f}")
    
    # Percentiles
    percentiles = [90, 95, 99, 99.9, 99.99]
    print(f"\nProbability Percentiles:")
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(probabilities, p):.6f}")
    
    return {
        'total_samples': total_samples,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'class_balance': positive_samples / total_samples,
        'prob_stats': {
            'mean': np.mean(probabilities),
            'median': np.median(probabilities),
            'std': np.std(probabilities),
            'min': np.min(probabilities),
            'max': np.max(probabilities)
        }
    }

def model_architecture_analysis(model, config):
    """Analyze model architecture"""
    print("\n🏗️  MODEL ARCHITECTURE ANALYSIS:")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Type: ConvLSTM")
    print(f"Input Channels: {config['input_channels']}")
    print(f"Hidden Channels: {config['hidden_channels']}")
    print(f"Number of Layers: {config['num_layers']}")
    print(f"Kernel Sizes: {config['kernel_sizes']}")
    print(f"Patch Size: {config['patch_size']}")
    print(f"Sequence Length: {config['seq_length']}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

def training_configuration_analysis(config):
    """Analyze training configuration"""
    print("\n⚙️  TRAINING CONFIGURATION ANALYSIS:")
    print("-" * 40)
    
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Weight Decay: {config['weight_decay']}")
    print(f"Number of Epochs: {config['num_epochs']}")
    print(f"Patience: {config['patience']}")
    print(f"Max Samples per Epoch: {config['max_samples_per_epoch']}")
    print(f"Max Val Samples: {config['max_val_samples']}")
    print(f"Augmentation: {config['use_augmentation']}")

def generate_recommendations(data_stats, threshold_results, config):
    """Generate recommendations based on analysis"""
    print("\n💡 RECOMMENDATIONS & ANALYSIS:")
    print("-" * 40)
    
    # Class imbalance analysis
    if data_stats['class_balance'] < 0.001:
        print("🔴 CRITICAL: Severe class imbalance (<0.1% positive)")
        print("   → This explains why F1=0 with standard threshold")
        print("   → Model learned to predict all negative (optimal for accuracy)")
    elif data_stats['class_balance'] < 0.01:
        print("🟡 WARNING: High class imbalance (<1% positive)")
    else:
        print("🟢 GOOD: Reasonable class balance")
    
    # Model learning analysis
    best_threshold = None
    best_f1 = 0
    for threshold, metrics in threshold_results.items():
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
    
    if best_f1 > 0:
        print(f"✅ Model CAN detect positive class!")
        print(f"   → Best F1: {best_f1:.4f} at threshold {best_threshold}")
        print(f"   → Model is learning, just needs proper threshold")
    else:
        print("❌ Model cannot detect positive class")
        print("   → Need different training approach")
    
    # Probability distribution analysis
    if data_stats['prob_stats']['max'] < 0.1:
        print("🔴 Model probabilities very low (max < 0.1)")
        print("   → Model not confident about any predictions")
    elif data_stats['prob_stats']['max'] < 0.5:
        print("🟡 Model probabilities moderate (max < 0.5)")
        print("   → Model somewhat confident but needs lower threshold")
    
    # Training recommendations
    print(f"\n🎯 SPECIFIC RECOMMENDATIONS:")
    if data_stats['class_balance'] < 0.001:
        print("1. Use weighted loss: BCEWithLogitsLoss(pos_weight=1000)")
        print("2. Implement balanced sampling (equal pos/neg per batch)")
        print("3. Use focal loss for extreme imbalance")
        print("4. Consider synthetic positive sample generation")
    
    if best_f1 > 0:
        print(f"5. Use threshold {best_threshold} instead of 0.5 for inference")
        print("6. Current model is usable with proper threshold!")
    
    print("7. Consider ensemble with existing U-Net models")
    print("8. Validate on held-out test set")

def main():
    """Main evaluation function"""
    print("🔍 COMPREHENSIVE CONVLSTM MODEL EVALUATION")
    print("=" * 60)
    print(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model_path = 'ml/models/convlstm_fast_final.pth'
    history_path = 'ml/models/convlstm_fast_history.json'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📂 Loading model: {model_path}")
    model, config, checkpoint = load_model(model_path)
    
    # Model architecture analysis
    model_architecture_analysis(model, config)
    
    # Training configuration analysis
    training_configuration_analysis(config)
    
    # Training history analysis
    if os.path.exists(history_path):
        history = analyze_training_history(history_path)
    
    # Create dataset for evaluation
    print(f"\n📊 Creating evaluation dataset...")
    dataset = MultiModalTimeSeriesDataset(
        s2_stacks_dir=config['s2_stacks_dir'],
        s1_stacks_dir=config['s1_stacks_dir'],
        change_labels_dir=config['change_labels_dir'],
        seq_length=config['seq_length'],
        patch_size=config['patch_size'],
        temporal_gap_days=config['temporal_gap_days'],
        min_cloud_free_ratio=config['min_cloud_free_ratio'],
        use_augmentation=False
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Comprehensive prediction analysis
    probabilities, targets, batch_losses = comprehensive_prediction_analysis(model, dataloader)
    
    # Data distribution analysis
    data_stats = analyze_data_distribution(targets, probabilities)
    
    # Multi-threshold analysis
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"\n🎯 MULTI-THRESHOLD PERFORMANCE ANALYSIS:")
    print("-" * 40)
    threshold_results = calculate_metrics_at_thresholds(probabilities, targets, thresholds)
    
    print(f"{'Threshold':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 70)
    for threshold in thresholds:
        metrics = threshold_results[threshold]
        print(f"{threshold:<10.3f} {metrics['f1']:<8.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<8.4f} {metrics['tp']:<8.0f} {metrics['fp']:<8.0f} {metrics['fn']:<8.0f}")
    
    # Generate recommendations
    generate_recommendations(data_stats, threshold_results, config)
    
    # Summary
    print(f"\n📋 EVALUATION SUMMARY:")
    print("=" * 40)
    print(f"✅ Model successfully trained and evaluated")
    print(f"✅ Architecture: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Training: {history['epochs_completed']} epochs, loss reduced by {((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.1f}%")
    print(f"✅ Data: {len(targets):,} samples analyzed")
    
    # Find best performance
    best_threshold = max(threshold_results.keys(), key=lambda t: threshold_results[t]['f1'])
    best_metrics = threshold_results[best_threshold]
    
    if best_metrics['f1'] > 0:
        print(f"✅ Best Performance: F1={best_metrics['f1']:.4f} at threshold={best_threshold}")
        print(f"🎯 MODEL IS FUNCTIONAL - Use threshold {best_threshold} for inference")
    else:
        print(f"⚠️  No positive predictions at any threshold")
        print(f"🔄 MODEL NEEDS RETRAINING with balanced sampling")
    
    print(f"\n✅ Comprehensive evaluation completed!")

if __name__ == "__main__":
    main() 