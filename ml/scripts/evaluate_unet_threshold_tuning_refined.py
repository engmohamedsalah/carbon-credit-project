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
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model_with_refined_thresholds(
    model_path='ml/models/forest_cover_unet_focal_alpha_0.75.pth',
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    results_dir='ml/evaluation_results',
    batch_size=16,
    val_split=0.2,
    threshold_ranges=None
):
    """
    Evaluates a trained U-Net model with refined threshold values.
    """
    if threshold_ranges is None:
        # Test multiple ranges for comprehensive analysis
        threshold_ranges = {
            'coarse': np.arange(0.1, 1.0, 0.1),  # Coarse overview
            'medium': np.arange(0.3, 0.8, 0.02),  # Medium around promising area
            'fine': np.arange(0.50, 0.65, 0.01),  # Fine-grained around optimal
            'extra_fine': np.arange(0.53, 0.58, 0.005)  # Extra fine around 0.55
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
    
    # Remove duplicates (same threshold tested in multiple ranges)
    combined_results = combined_results.drop_duplicates(subset=['threshold']).sort_values('threshold').reset_index(drop=True)
    
    # Find best thresholds for different metrics
    best_f1_idx = combined_results['f1'].idxmax()
    best_precision_idx = combined_results['precision'].idxmax()
    best_recall_idx = combined_results['recall'].idxmax()
    best_iou_idx = combined_results['iou'].idxmax()
    
    # Find best balanced metric (harmonic mean of precision and recall)
    combined_results['balanced_score'] = 2 / (1/combined_results['precision'] + 1/combined_results['recall'])
    combined_results['balanced_score'] = combined_results['balanced_score'].fillna(0)
    best_balanced_idx = combined_results['balanced_score'].idxmax()
    
    logger.info("=== REFINED THRESHOLD TUNING RESULTS ===")
    logger.info(f"Best F1 Score: {combined_results.loc[best_f1_idx, 'f1']:.6f} at threshold {combined_results.loc[best_f1_idx, 'threshold']:.4f}")
    logger.info(f"  - Precision: {combined_results.loc[best_f1_idx, 'precision']:.6f}")
    logger.info(f"  - Recall: {combined_results.loc[best_f1_idx, 'recall']:.6f}")
    logger.info(f"  - IoU: {combined_results.loc[best_f1_idx, 'iou']:.6f}")
    logger.info("")
    
    logger.info(f"Best Balanced Score: {combined_results.loc[best_balanced_idx, 'balanced_score']:.6f} at threshold {combined_results.loc[best_balanced_idx, 'threshold']:.4f}")
    logger.info(f"  - Precision: {combined_results.loc[best_balanced_idx, 'precision']:.6f}")
    logger.info(f"  - Recall: {combined_results.loc[best_balanced_idx, 'recall']:.6f}")
    logger.info(f"  - F1: {combined_results.loc[best_balanced_idx, 'f1']:.6f}")
    logger.info(f"  - IoU: {combined_results.loc[best_balanced_idx, 'iou']:.6f}")
    logger.info("")
    
    logger.info(f"Best IoU: {combined_results.loc[best_iou_idx, 'iou']:.6f} at threshold {combined_results.loc[best_iou_idx, 'threshold']:.4f}")
    logger.info(f"  - F1: {combined_results.loc[best_iou_idx, 'f1']:.6f}")
    logger.info(f"  - Precision: {combined_results.loc[best_iou_idx, 'precision']:.6f}")
    logger.info(f"  - Recall: {combined_results.loc[best_iou_idx, 'recall']:.6f}")
    logger.info("")
    
    # Show top 5 thresholds by F1 score
    top_5_f1 = combined_results.nlargest(5, 'f1')
    logger.info("Top 5 Thresholds by F1 Score:")
    for idx, row in top_5_f1.iterrows():
        logger.info(f"  {row['threshold']:.4f}: F1={row['f1']:.6f}, Precision={row['precision']:.6f}, Recall={row['recall']:.6f}, IoU={row['iou']:.6f}")
    logger.info("")
    
    # Original threshold comparisons
    original_thresholds = [0.5, 0.55]
    logger.info("Comparison with key thresholds:")
    for thresh in original_thresholds:
        thresh_result = combined_results[abs(combined_results['threshold'] - thresh) < 0.001]
        if len(thresh_result) > 0:
            row = thresh_result.iloc[0]
            logger.info(f"  Threshold {thresh}: F1={row['f1']:.6f}, Precision={row['precision']:.6f}, Recall={row['recall']:.6f}, IoU={row['iou']:.6f}")
    
    # Create results directory and save results
    Path(results_dir).mkdir(exist_ok=True)
    combined_results.to_csv(f"{results_dir}/refined_threshold_tuning_results.csv", index=False)
    
    # Create detailed plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot 1: Precision vs Threshold
    axes[0, 0].plot(combined_results['threshold'], combined_results['precision'], 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'precision'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Recall vs Threshold
    axes[0, 1].plot(combined_results['threshold'], combined_results['recall'], 'r-', linewidth=2, alpha=0.7)
    axes[0, 1].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'recall'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: F1 Score vs Threshold
    axes[1, 0].plot(combined_results['threshold'], combined_results['f1'], 'g-', linewidth=2, alpha=0.7)
    axes[1, 0].scatter(combined_results.loc[best_f1_idx, 'threshold'], combined_results.loc[best_f1_idx, 'f1'], 
                       color='red', s=100, zorder=5, label=f"Best F1 ({combined_results.loc[best_f1_idx, 'threshold']:.4f})")
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: IoU vs Threshold
    axes[1, 1].plot(combined_results['threshold'], combined_results['iou'], 'm-', linewidth=2, alpha=0.7)
    axes[1, 1].scatter(combined_results.loc[best_iou_idx, 'threshold'], combined_results.loc[best_iou_idx, 'iou'], 
                       color='red', s=100, zorder=5, label=f"Best IoU ({combined_results.loc[best_iou_idx, 'threshold']:.4f})")
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('IoU vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 5: Combined metrics
    axes[2, 0].plot(combined_results['threshold'], combined_results['precision'], 'b-', label='Precision', linewidth=2, alpha=0.7)
    axes[2, 0].plot(combined_results['threshold'], combined_results['recall'], 'r-', label='Recall', linewidth=2, alpha=0.7)
    axes[2, 0].plot(combined_results['threshold'], combined_results['f1'], 'g-', label='F1 Score', linewidth=2, alpha=0.7)
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
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/refined_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved to {results_dir}/refined_threshold_tuning_results.csv")
    logger.info(f"Plots saved to {results_dir}/refined_threshold_analysis.png")
    
    return combined_results

if __name__ == '__main__':
    evaluate_model_with_refined_thresholds() 