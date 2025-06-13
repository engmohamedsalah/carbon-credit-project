#!/usr/bin/env python3
"""
Evaluate Fast ConvLSTM Model
Check model predictions and understand the F1=0 issue
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    
    return model, config

def analyze_predictions(model, dataloader, num_batches=5):
    """Analyze model predictions to understand the F1=0 issue"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            outputs, _ = model(sequences)
            probabilities = torch.sigmoid(outputs.squeeze(1))
            predictions = probabilities > 0.5
            targets = labels.squeeze(1) > 0.5
            
            all_predictions.extend(predictions.flatten().numpy())
            all_targets.extend(targets.flatten().numpy())
            all_probabilities.extend(probabilities.flatten().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    return all_predictions, all_targets, all_probabilities

def calculate_detailed_metrics(predictions, targets, probabilities):
    """Calculate detailed metrics and statistics"""
    # Basic counts
    total_samples = len(targets)
    positive_targets = np.sum(targets)
    negative_targets = total_samples - positive_targets
    positive_predictions = np.sum(predictions)
    negative_predictions = total_samples - positive_predictions
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_samples
    
    # Probability statistics
    prob_mean = np.mean(probabilities)
    prob_std = np.std(probabilities)
    prob_min = np.min(probabilities)
    prob_max = np.max(probabilities)
    
    return {
        'total_samples': total_samples,
        'positive_targets': positive_targets,
        'negative_targets': negative_targets,
        'positive_predictions': positive_predictions,
        'negative_predictions': negative_predictions,
        'class_balance': positive_targets / total_samples,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'prob_mean': prob_mean,
        'prob_std': prob_std,
        'prob_min': prob_min,
        'prob_max': prob_max
    }

def main():
    print("🔍 Evaluating Fast ConvLSTM Model")
    print("=" * 50)
    
    # Load model
    model_path = 'ml/models/convlstm_fast_final.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📂 Loading model: {model_path}")
    model, config = load_model(model_path)
    
    print("📋 Model Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create dataset
    print("\n📊 Creating evaluation dataset...")
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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Analyze predictions
    print("\n🔍 Analyzing model predictions...")
    predictions, targets, probabilities = analyze_predictions(model, dataloader, num_batches=10)
    
    # Calculate metrics
    metrics = calculate_detailed_metrics(predictions, targets, probabilities)
    
    print("\n📊 EVALUATION RESULTS:")
    print("-" * 30)
    print(f"Total samples analyzed: {metrics['total_samples']:,}")
    print(f"Positive targets: {metrics['positive_targets']:,} ({metrics['class_balance']:.4f})")
    print(f"Negative targets: {metrics['negative_targets']:,}")
    print(f"Positive predictions: {metrics['positive_predictions']:,}")
    print(f"Negative predictions: {metrics['negative_predictions']:,}")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"True Positives (TP): {metrics['tp']:,}")
    print(f"False Positives (FP): {metrics['fp']:,}")
    print(f"False Negatives (FN): {metrics['fn']:,}")
    print(f"True Negatives (TN): {metrics['tn']:,}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print(f"\n🎲 PROBABILITY STATISTICS:")
    print(f"Mean probability: {metrics['prob_mean']:.4f}")
    print(f"Std probability: {metrics['prob_std']:.4f}")
    print(f"Min probability: {metrics['prob_min']:.4f}")
    print(f"Max probability: {metrics['prob_max']:.4f}")
    
    # Diagnosis
    print(f"\n🔬 DIAGNOSIS:")
    if metrics['class_balance'] < 0.01:
        print("⚠️  SEVERE CLASS IMBALANCE: <1% positive samples")
        print("   → Model learns to predict all negative")
        print("   → Need balanced sampling or weighted loss")
    elif metrics['positive_predictions'] == 0:
        print("⚠️  MODEL PREDICTS ALL NEGATIVE")
        print("   → Threshold too high or model not learning positive class")
        print("   → Try lower threshold or different loss function")
    elif metrics['prob_max'] < 0.5:
        print("⚠️  ALL PROBABILITIES < 0.5")
        print("   → Model not confident about positive class")
        print("   → Need more training or different architecture")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if metrics['f1'] == 0:
        print("1. Use weighted loss function (BCEWithLogitsLoss with pos_weight)")
        print("2. Try different threshold (e.g., 0.1 instead of 0.5)")
        print("3. Use focal loss for class imbalance")
        print("4. Implement balanced sampling")
    
    print(f"\n✅ Evaluation completed!")

if __name__ == "__main__":
    main() 