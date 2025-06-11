import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import morphology, measure

from ml.models.unet import UNet
from ml.training.dataset_patches import PatchDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_morphological_postprocessing(predictions, method='comprehensive', min_area=50):
    """
    Apply morphological post-processing to clean up predictions.
    
    Args:
        predictions: numpy array of binary predictions (H, W) with values 0 or 1
        method: 'basic', 'comprehensive', or 'aggressive'
        min_area: minimum area (in pixels) for connected components
    
    Returns:
        processed_predictions: cleaned binary predictions
    """
    processed = predictions.copy()
    
    if method == 'basic':
        # Basic morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Opening (erosion followed by dilation) - removes small noise
        processed = cv2.morphologyEx(processed.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Small area removal
        processed = remove_small_objects(processed, min_area=min_area)
        
    elif method == 'comprehensive':
        # Multi-scale morphological operations
        
        # 1. Small noise removal with opening
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(processed.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
        
        # 2. Fill small holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_fill)
        
        # 3. Remove small objects
        processed = remove_small_objects(processed, min_area=min_area)
        
        # 4. Smooth boundaries
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_smooth)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_smooth)
        
    elif method == 'aggressive':
        # Aggressive cleaning for high precision
        
        # 1. Larger opening to remove more noise
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed = cv2.morphologyEx(processed.astype(np.uint8), cv2.MORPH_OPEN, kernel_large)
        
        # 2. Remove small objects (larger threshold)
        processed = remove_small_objects(processed, min_area=min_area * 2)
        
        # 3. Convex hull approximation for very smooth regions
        processed = apply_convex_hull_smoothing(processed)
        
        # 4. Final opening
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_final)
    
    return processed.astype(np.float32)

def remove_small_objects(binary_img, min_area=50):
    """Remove connected components smaller than min_area."""
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_img.astype(np.uint8))
    
    # Create output image
    output = np.zeros_like(binary_img)
    
    # Keep only components with area >= min_area
    for label in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == label)
        if np.sum(component_mask) >= min_area:
            output[component_mask] = 1
    
    return output

def apply_convex_hull_smoothing(binary_img):
    """Apply convex hull to large connected components for smoothing."""
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_img.astype(np.uint8))
    
    output = np.zeros_like(binary_img)
    
    for label in range(1, num_labels):
        component_mask = (labels == label)
        area = np.sum(component_mask)
        
        # Apply convex hull only to larger components
        if area > 200:  # Threshold for applying convex hull
            # Find contours of this component
            component_img = component_mask.astype(np.uint8)
            contours, _ = cv2.findContours(component_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get convex hull
                hull = cv2.convexHull(contours[0])
                # Fill convex hull
                cv2.fillPoly(output, [hull], 1)
        else:
            # Keep small components as is
            output[component_mask] = 1
    
    return output

def gaussian_smoothing(predictions, sigma=0.5):
    """Apply Gaussian smoothing to predictions before thresholding."""
    return ndimage.gaussian_filter(predictions, sigma=sigma)

def calculate_metrics_with_postprocessing(preds_proba, labels, threshold, postprocess_methods):
    """Calculate metrics with different post-processing methods."""
    results = {}
    
    # Original predictions (no post-processing)
    preds_binary = (preds_proba > threshold).astype(np.float32)
    results['original'] = calculate_single_metrics(preds_binary, labels)
    
    # Apply different post-processing methods
    for method_name, method_config in postprocess_methods.items():
        processed_preds = preds_binary.copy()
        
        # Apply Gaussian smoothing if specified
        if 'gaussian_sigma' in method_config:
            processed_preds = gaussian_smoothing(preds_proba, sigma=method_config['gaussian_sigma'])
            processed_preds = (processed_preds > threshold).astype(np.float32)
        
        # Apply morphological operations
        if 'morph_method' in method_config:
            for i in range(processed_preds.shape[0]):  # Process each image in batch
                processed_preds[i, 0] = apply_morphological_postprocessing(
                    processed_preds[i, 0],
                    method=method_config['morph_method'],
                    min_area=method_config.get('min_area', 50)
                )
        
        results[method_name] = calculate_single_metrics(processed_preds, labels)
    
    return results

def calculate_single_metrics(preds, labels):
    """Calculate metrics for a single prediction method."""
    # Flatten for metric calculation
    preds_flat = preds.reshape(-1)
    labels_flat = (labels > 0.5).reshape(-1)
    
    # Calculate metrics
    intersection = (preds_flat * labels_flat).sum()
    
    tp = intersection
    fp = preds_flat.sum() - intersection
    fn = labels_flat.sum() - intersection
    tn = len(preds_flat) - tp - fp - fn
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
        'specificity': specificity
    }

def evaluate_model_with_postprocessing(
    model_path='ml/models/forest_cover_unet_focal_alpha_0.75.pth',
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    results_dir='ml/evaluation_results',
    batch_size=16,
    val_split=0.2,
    threshold=0.53,
    save_examples=True
):
    """
    Evaluate model with different post-processing methods.
    """
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

    # Define post-processing methods to test
    postprocess_methods = {
        'gaussian_light': {
            'gaussian_sigma': 0.5
        },
        'morph_basic': {
            'morph_method': 'basic',
            'min_area': 50
        },
        'morph_comprehensive': {
            'morph_method': 'comprehensive',
            'min_area': 50
        },
        'morph_aggressive': {
            'morph_method': 'aggressive',
            'min_area': 100
        },
        'gaussian_morph_combo': {
            'gaussian_sigma': 0.3,
            'morph_method': 'comprehensive',
            'min_area': 30
        },
        'precision_focused': {
            'morph_method': 'aggressive',
            'min_area': 150
        }
    }

    # Store all predictions and labels
    all_predictions = []
    all_labels = []
    all_images = []

    logger.info("Collecting predictions...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds_proba = torch.sigmoid(outputs)
            
            all_predictions.append(preds_proba.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(images.cpu().numpy())

    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    logger.info(f"Evaluating with threshold {threshold} and {len(postprocess_methods)} post-processing methods...")

    # Calculate metrics for all methods
    method_results = calculate_metrics_with_postprocessing(
        all_predictions, all_labels, threshold, postprocess_methods
    )

    # Print results
    logger.info("=== POST-PROCESSING EVALUATION RESULTS ===")
    logger.info(f"Threshold used: {threshold}")
    logger.info("")

    # Sort methods by F1 score
    sorted_methods = sorted(method_results.items(), key=lambda x: x[1]['f1'], reverse=True)

    for i, (method_name, metrics) in enumerate(sorted_methods):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        logger.info(f"{rank_emoji} {method_name.upper()}:")
        logger.info(f"   F1: {metrics['f1']:.6f} | Precision: {metrics['precision']:.6f} | Recall: {metrics['recall']:.6f}")
        logger.info(f"   IoU: {metrics['iou']:.6f} | Accuracy: {metrics['accuracy']:.6f} | Specificity: {metrics['specificity']:.6f}")
        
        if method_name != 'original':
            orig_f1 = method_results['original']['f1']
            f1_improvement = ((metrics['f1'] - orig_f1) / orig_f1) * 100
            orig_precision = method_results['original']['precision']
            precision_improvement = ((metrics['precision'] - orig_precision) / orig_precision) * 100
            logger.info(f"   Improvement: F1 {f1_improvement:+.2f}% | Precision {precision_improvement:+.2f}%")
        logger.info("")

    # Save detailed results
    Path(results_dir).mkdir(exist_ok=True)
    
    # Save metrics to CSV
    results_df = pd.DataFrame(method_results).T
    results_df['method'] = results_df.index
    results_df.to_csv(f"{results_dir}/postprocessing_evaluation_results.csv", index=False)

    # Save visual examples if requested
    if save_examples:
        save_postprocessing_examples(
            all_images, all_predictions, all_labels, 
            threshold, postprocess_methods, results_dir, n_examples=8
        )

    logger.info(f"Results saved to {results_dir}/postprocessing_evaluation_results.csv")
    
    return method_results

def save_postprocessing_examples(images, predictions, labels, threshold, postprocess_methods, 
                                results_dir, n_examples=8):
    """Save visual examples comparing different post-processing methods."""
    
    # Select random examples
    indices = np.random.choice(len(images), min(n_examples, len(images)), replace=False)
    
    for idx, img_idx in enumerate(indices):
        image = images[img_idx]
        pred_proba = predictions[img_idx, 0]
        label = labels[img_idx, 0]
        
        # Create RGB visualization (use bands 4,3,2 for natural color)
        rgb_image = image[[3, 2, 1], :, :]  # B04, B03, B02
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        # Create subplot
        n_methods = len(postprocess_methods) + 2  # +2 for original and ground truth
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(16, 8))
        axes = axes.flatten() if n_methods > 2 else [axes]
        
        # Show RGB image
        axes[0].imshow(rgb_image)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')
        
        # Show ground truth
        axes[1].imshow(label, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Show original prediction
        original_pred = (pred_proba > threshold).astype(np.float32)
        axes[2].imshow(original_pred, cmap='gray')
        axes[2].set_title("Original Prediction")
        axes[2].axis('off')
        
        # Show post-processed versions
        ax_idx = 3
        for method_name, method_config in postprocess_methods.items():
            if ax_idx >= len(axes):
                break
                
            processed_pred = original_pred.copy()
            
            # Apply Gaussian smoothing if specified
            if 'gaussian_sigma' in method_config:
                processed_pred = gaussian_smoothing(pred_proba, sigma=method_config['gaussian_sigma'])
                processed_pred = (processed_pred > threshold).astype(np.float32)
            
            # Apply morphological operations
            if 'morph_method' in method_config:
                processed_pred = apply_morphological_postprocessing(
                    processed_pred,
                    method=method_config['morph_method'],
                    min_area=method_config.get('min_area', 50)
                )
            
            axes[ax_idx].imshow(processed_pred, cmap='gray')
            axes[ax_idx].set_title(method_name.replace('_', ' ').title())
            axes[ax_idx].axis('off')
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/postprocessing_example_{idx}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved {len(indices)} post-processing examples to {results_dir}")

if __name__ == '__main__':
    evaluate_model_with_postprocessing() 