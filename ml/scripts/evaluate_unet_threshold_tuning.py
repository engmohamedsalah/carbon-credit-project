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

def evaluate_model_with_thresholds(
    model_path='ml/models/forest_cover_unet_focal_alpha_0.75.pth',
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    results_dir='ml/evaluation_results',
    batch_size=16,
    val_split=0.2,
    thresholds=None
):
    """
    Evaluates a trained U-Net model with multiple threshold values.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)  # Test thresholds from 0.1 to 0.95
    
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

    # Test different thresholds
    results = []
    logger.info(f"Testing {len(thresholds)} different thresholds...")
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        metrics = calculate_metrics(all_predictions, all_labels, threshold)
        metrics['threshold'] = threshold
        results.append(metrics)
        
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Find best thresholds for different metrics
    best_f1_idx = results_df['f1'].idxmax()
    best_precision_idx = results_df['precision'].idxmax()
    best_recall_idx = results_df['recall'].idxmax()
    best_iou_idx = results_df['iou'].idxmax()
    
    logger.info("=== THRESHOLD TUNING RESULTS ===")
    logger.info(f"Best F1 Score: {results_df.loc[best_f1_idx, 'f1']:.4f} at threshold {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    logger.info(f"  - Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
    logger.info(f"  - Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")
    logger.info(f"  - IoU: {results_df.loc[best_f1_idx, 'iou']:.4f}")
    logger.info("")
    
    logger.info(f"Best Precision: {results_df.loc[best_precision_idx, 'precision']:.4f} at threshold {results_df.loc[best_precision_idx, 'threshold']:.2f}")
    logger.info(f"  - F1: {results_df.loc[best_precision_idx, 'f1']:.4f}")
    logger.info(f"  - Recall: {results_df.loc[best_precision_idx, 'recall']:.4f}")
    logger.info("")
    
    logger.info(f"Best IoU: {results_df.loc[best_iou_idx, 'iou']:.4f} at threshold {results_df.loc[best_iou_idx, 'threshold']:.2f}")
    logger.info(f"  - F1: {results_df.loc[best_iou_idx, 'f1']:.4f}")
    logger.info(f"  - Precision: {results_df.loc[best_iou_idx, 'precision']:.4f}")
    logger.info(f"  - Recall: {results_df.loc[best_iou_idx, 'recall']:.4f}")
    logger.info("")
    
    # Original threshold (0.5) for comparison
    original_idx = results_df[results_df['threshold'] == 0.5].index
    if len(original_idx) > 0:
        original_idx = original_idx[0]
        logger.info(f"Original threshold (0.5) results:")
        logger.info(f"  - F1: {results_df.loc[original_idx, 'f1']:.4f}")
        logger.info(f"  - Precision: {results_df.loc[original_idx, 'precision']:.4f}")
        logger.info(f"  - Recall: {results_df.loc[original_idx, 'recall']:.4f}")
        logger.info(f"  - IoU: {results_df.loc[original_idx, 'iou']:.4f}")
    
    # Create results directory and save results
    Path(results_dir).mkdir(exist_ok=True)
    results_df.to_csv(f"{results_dir}/threshold_tuning_results.csv", index=False)
    
    # Plot threshold vs metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(results_df['threshold'], results_df['precision'], 'b-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Threshold')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(results_df['threshold'], results_df['recall'], 'r-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs Threshold')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(results_df['threshold'], results_df['f1'], 'g-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(results_df['threshold'], results_df['iou'], 'm-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('IoU')
    plt.title('IoU vs Threshold')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], 'r-', label='Recall', linewidth=2)
    plt.plot(results_df['threshold'], results_df['f1'], 'g-', label='F1 Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall-F1 vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(results_df['recall'], results_df['precision'], 'o-', linewidth=2, markersize=4)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/threshold_tuning_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved to {results_dir}/threshold_tuning_results.csv")
    logger.info(f"Plots saved to {results_dir}/threshold_tuning_analysis.png")
    
    return results_df

if __name__ == '__main__':
    evaluate_model_with_thresholds() 