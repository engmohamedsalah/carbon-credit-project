import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from ml.models.unet import UNet
from ml.training.dataset_patches import PatchDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(preds_proba, labels, threshold):
    """Calculate metrics for a given threshold"""
    preds = preds_proba > threshold
    
    # Flatten for metric calculation
    preds_flat = preds.view(-1)
    labels_flat = (labels > 0.5).view(-1)  # Threshold labels to binary and flatten

    # Move to CPU for metric calculation
    preds_flat = preds_flat.cpu()
    labels_flat = labels_flat.cpu()

    # Calculate metrics
    intersection = (preds_flat & labels_flat).sum().float()
    
    tp = intersection
    fp = (preds_flat & ~labels_flat).sum().float()
    fn = (~preds_flat & labels_flat).sum().float()
    tn = (~preds_flat & ~labels_flat).sum().float()

    iou = (tp / (tp + fp + fn + 1e-6)).item()
    dice = ((2. * tp) / (2. * tp + fp + fn + 1e-6)).item()
    accuracy = ((tp + tn) / (tp + tn + fp + fn + 1e-6)).item()
    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    
    # Calculate specificity (true negative rate)
    specificity = (tn / (tn + fp + 1e-6)).item()
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }

def evaluate_comprehensive_thresholds(
    model_path='ml/models/forest_cover_unet_focal_alpha_0.75.pth',
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    results_dir='ml/evaluation_results',
    batch_size=16,
    val_split=0.2
):
    """
    Evaluates a trained U-Net model with comprehensive threshold values across the full spectrum.
    """
    
    # Define comprehensive threshold ranges
    threshold_ranges = {
        'very_low': np.arange(0.01, 0.1, 0.01),         # 0.01 to 0.09
        'low': np.arange(0.1, 0.3, 0.02),               # 0.1 to 0.28
        'medium_low': np.arange(0.3, 0.5, 0.01),        # 0.3 to 0.49
        'optimal_zone': np.arange(0.50, 0.60, 0.005),   # 0.50 to 0.595 (fine-grained around optimal)
        'medium_high': np.arange(0.6, 0.8, 0.01),       # 0.6 to 0.79
        'high': np.arange(0.8, 0.9, 0.02),              # 0.8 to 0.88
        'very_high': np.arange(0.9, 1.0, 0.01),         # 0.9 to 0.99
        'extreme': [0.995, 0.999]                        # Edge cases
    }
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model
    if not Path(model_path).exists():
        logger.error(f"Model file not found at {model_path}. Exiting.")
        return
        
    model = UNet(n_channels=12, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    df = pd.read_csv(csv_file)
    dataset = PatchDataset(df)

    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    torch.manual_seed(42)  # Use the same seed as in training for consistent split
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    logger.info(f"Validation set size: {len(val_dataset)}")

    # Store all predictions and labels for threshold testing
    all_predictions = []
    all_labels = []

    # Get predictions for all data
    logger.info("Collecting predictions...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds_proba = torch.sigmoid(outputs)
            
            all_predictions.append(preds_proba.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Test different threshold ranges
    all_results = {}
    total_thresholds = sum(len(thresholds) for thresholds in threshold_ranges.values())
    logger.info(f"Testing {total_thresholds} thresholds across {len(threshold_ranges)} ranges...")
    
    for range_name, thresholds in threshold_ranges.items():
        logger.info(f"Testing {range_name} range with {len(thresholds)} thresholds...")
        results = []
        
        for threshold in tqdm(thresholds, desc=f"Testing {range_name} thresholds"):
            metrics = calculate_metrics(all_predictions, all_labels, threshold)
            metrics['threshold'] = threshold
            metrics['range'] = range_name
            results.append(metrics)
            
        all_results[range_name] = pd.DataFrame(results)
    
    # Combine all results
    combined_results = pd.concat(all_results.values(), ignore_index=True)
    
    # Remove duplicates and sort
    combined_results = combined_results.drop_duplicates(subset=['threshold']).sort_values('threshold').reset_index(drop=True)
    
    # Find best thresholds for different metrics
    best_f1_idx = combined_results['f1'].idxmax()
    best_precision_idx = combined_results['precision'].idxmax()
    best_recall_idx = combined_results['recall'].idxmax()
    best_iou_idx = combined_results['iou'].idxmax()
    best_accuracy_idx = combined_results['accuracy'].idxmax()
    
    # Find best balanced metric (harmonic mean of precision and recall)
    combined_results['balanced_score'] = 2 / (1/combined_results['precision'] + 1/combined_results['recall'])
    combined_results['balanced_score'] = combined_results['balanced_score'].fillna(0)
    best_balanced_idx = combined_results['balanced_score'].idxmax()
    
    # Calculate Youden's J statistic (sensitivity + specificity - 1)
    combined_results['youden_j'] = combined_results['recall'] + combined_results['specificity'] - 1
    best_youden_idx = combined_results['youden_j'].idxmax()
    
    logger.info("=== COMPREHENSIVE THRESHOLD ANALYSIS RESULTS ===")
    logger.info(f"Total thresholds tested: {len(combined_results)}")
    logger.info("")
    
    logger.info(f"🏆 Best F1 Score: {combined_results.loc[best_f1_idx, 'f1']:.6f} at threshold {combined_results.loc[best_f1_idx, 'threshold']:.4f}")
    logger.info(f"   Precision: {combined_results.loc[best_f1_idx, 'precision']:.6f}, Recall: {combined_results.loc[best_f1_idx, 'recall']:.6f}, IoU: {combined_results.loc[best_f1_idx, 'iou']:.6f}")
    logger.info("")
    
    logger.info(f"🎯 Best IoU: {combined_results.loc[best_iou_idx, 'iou']:.6f} at threshold {combined_results.loc[best_iou_idx, 'threshold']:.4f}")
    logger.info(f"   F1: {combined_results.loc[best_iou_idx, 'f1']:.6f}, Precision: {combined_results.loc[best_iou_idx, 'precision']:.6f}, Recall: {combined_results.loc[best_iou_idx, 'recall']:.6f}")
    logger.info("")
    
    logger.info(f"⚖️  Best Balanced Score: {combined_results.loc[best_balanced_idx, 'balanced_score']:.6f} at threshold {combined_results.loc[best_balanced_idx, 'threshold']:.4f}")
    logger.info(f"   F1: {combined_results.loc[best_balanced_idx, 'f1']:.6f}, Precision: {combined_results.loc[best_balanced_idx, 'precision']:.6f}, Recall: {combined_results.loc[best_balanced_idx, 'recall']:.6f}")
    logger.info("")
    
    logger.info(f"📊 Best Youden's J: {combined_results.loc[best_youden_idx, 'youden_j']:.6f} at threshold {combined_results.loc[best_youden_idx, 'threshold']:.4f}")
    logger.info(f"   F1: {combined_results.loc[best_youden_idx, 'f1']:.6f}, Precision: {combined_results.loc[best_youden_idx, 'precision']:.6f}, Recall: {combined_results.loc[best_youden_idx, 'recall']:.6f}")
    logger.info("")
    
    logger.info(f"🎖️  Best Accuracy: {combined_results.loc[best_accuracy_idx, 'accuracy']:.6f} at threshold {combined_results.loc[best_accuracy_idx, 'threshold']:.4f}")
    logger.info(f"   F1: {combined_results.loc[best_accuracy_idx, 'f1']:.6f}, Precision: {combined_results.loc[best_accuracy_idx, 'precision']:.6f}, Recall: {combined_results.loc[best_accuracy_idx, 'recall']:.6f}")
    logger.info("")
    
    # Show top 10 thresholds by F1 score
    top_10_f1 = combined_results.nlargest(10, 'f1')
    logger.info("🔝 Top 10 Thresholds by F1 Score:")
    for i, (idx, row) in enumerate(top_10_f1.iterrows(), 1):
        logger.info(f"  {i:2d}. {row['threshold']:.4f}: F1={row['f1']:.6f}, P={row['precision']:.4f}, R={row['recall']:.4f}, IoU={row['iou']:.4f}")
    logger.info("")
    
    # Analyze extreme thresholds
    very_low_results = combined_results[combined_results['threshold'] <= 0.1]
    very_high_results = combined_results[combined_results['threshold'] >= 0.9]
    
    if len(very_low_results) > 0:
        best_very_low = very_low_results.loc[very_low_results['f1'].idxmax()]
        logger.info(f"📉 Best Very Low Threshold: {best_very_low['threshold']:.4f}")
        logger.info(f"   F1={best_very_low['f1']:.6f}, Precision={best_very_low['precision']:.6f}, Recall={best_very_low['recall']:.6f}")
        logger.info("")
    
    if len(very_high_results) > 0:
        best_very_high = very_high_results.loc[very_high_results['f1'].idxmax()]
        logger.info(f"📈 Best Very High Threshold: {best_very_high['threshold']:.4f}")
        logger.info(f"   F1={best_very_high['f1']:.6f}, Precision={best_very_high['precision']:.6f}, Recall={best_very_high['recall']:.6f}")
        logger.info("")
    
    # Key threshold comparisons
    key_thresholds = [0.1, 0.3, 0.5, 0.53, 0.7, 0.9]
    logger.info("📋 Key Threshold Comparisons:")
    for thresh in key_thresholds:
        closest_result = combined_results.iloc[(combined_results['threshold'] - thresh).abs().argsort()[:1]]
        if len(closest_result) > 0:
            row = closest_result.iloc[0]
            logger.info(f"   {thresh:.1f}: F1={row['f1']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}, IoU={row['iou']:.4f} (actual: {row['threshold']:.4f})")
    
    # Create results directory and save results
    Path(results_dir).mkdir(exist_ok=True)
    combined_results.to_csv(f"{results_dir}/comprehensive_threshold_analysis.csv", index=False)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    
    # Plot 1: F1 Score vs Threshold (full range)
    axes[0, 0].plot(combined_results['threshold'], combined_results['f1'], 'g-', linewidth=2, alpha=0.8)
    axes[0, 0].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'f1'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score vs Threshold (Full Range)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Precision vs Threshold (full range)
    axes[0, 1].plot(combined_results['threshold'], combined_results['precision'], 'b-', linewidth=2, alpha=0.8)
    axes[0, 1].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'precision'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Threshold (Full Range)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Recall vs Threshold (full range)
    axes[1, 0].plot(combined_results['threshold'], combined_results['recall'], 'r-', linewidth=2, alpha=0.8)
    axes[1, 0].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'recall'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs Threshold (Full Range)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: IoU vs Threshold (full range)
    axes[1, 1].plot(combined_results['threshold'], combined_results['iou'], 'm-', linewidth=2, alpha=0.8)
    axes[1, 1].scatter(combined_results.loc[best_iou_idx, 'threshold'], combined_results.loc[best_iou_idx, 'iou'], 
                       color='red', s=100, zorder=5, label=f"Best IoU ({combined_results.loc[best_iou_idx, 'threshold']:.4f})")
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('IoU vs Threshold (Full Range)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 5: Combined metrics overview
    axes[2, 0].plot(combined_results['threshold'], combined_results['precision'], 'b-', label='Precision', linewidth=2, alpha=0.7)
    axes[2, 0].plot(combined_results['threshold'], combined_results['recall'], 'r-', label='Recall', linewidth=2, alpha=0.7)
    axes[2, 0].plot(combined_results['threshold'], combined_results['f1'], 'g-', label='F1 Score', linewidth=2, alpha=0.7)
    axes[2, 0].plot(combined_results['threshold'], combined_results['specificity'], 'orange', label='Specificity', linewidth=2, alpha=0.7)
    axes[2, 0].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'f1'], 
                       color='black', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[2, 0].set_xlabel('Threshold')
    axes[2, 0].set_ylabel('Score')
    axes[2, 0].set_title('All Metrics vs Threshold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Precision-Recall Curve
    axes[2, 1].plot(combined_results['recall'], combined_results['precision'], 'o-', linewidth=2, markersize=3, alpha=0.7)
    axes[2, 1].scatter(combined_results.loc[best_f1_idx, 'recall'], combined_results.loc[best_f1_idx, 'precision'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[2, 1].set_xlabel('Recall')
    axes[2, 1].set_ylabel('Precision')
    axes[2, 1].set_title('Precision-Recall Curve')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    # Plot 7: Zoomed view around optimal region (0.4-0.7)
    optimal_region = combined_results[(combined_results['threshold'] >= 0.4) & (combined_results['threshold'] <= 0.7)]
    axes[3, 0].plot(optimal_region['threshold'], optimal_region['precision'], 'b-', label='Precision', linewidth=2)
    axes[3, 0].plot(optimal_region['threshold'], optimal_region['recall'], 'r-', label='Recall', linewidth=2)
    axes[3, 0].plot(optimal_region['threshold'], optimal_region['f1'], 'g-', label='F1 Score', linewidth=2)
    axes[3, 0].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'f1'], 
                       color='black', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[3, 0].set_xlabel('Threshold')
    axes[3, 0].set_ylabel('Score')
    axes[3, 0].set_title('Optimal Region Detail (0.4-0.7)')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    # Plot 8: Youden's J Statistic
    axes[3, 1].plot(combined_results['threshold'], combined_results['youden_j'], 'purple', linewidth=2, alpha=0.8)
    axes[3, 1].scatter(combined_results.loc[best_youden_idx, 'threshold'], combined_results.loc[best_youden_idx, 'youden_j'], 
                       color='red', s=100, zorder=5, label=f"Best Youden's J ({combined_results.loc[best_youden_idx, 'threshold']:.4f})")
    axes[3, 1].set_xlabel('Threshold')
    axes[3, 1].set_ylabel("Youden's J Statistic")
    axes[3, 1].set_title("Youden's J Statistic vs Threshold")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comprehensive_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved to {results_dir}/comprehensive_threshold_analysis.csv")
    logger.info(f"Comprehensive plots saved to {results_dir}/comprehensive_threshold_analysis.png")
    
    return combined_results

if __name__ == '__main__':
    evaluate_comprehensive_thresholds() 