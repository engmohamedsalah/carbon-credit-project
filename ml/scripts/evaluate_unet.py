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

def evaluate_model(
    model_path='ml/models/forest_cover_unet.pth',
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    results_dir='ml/evaluation_results',
    batch_size=16,
    val_split=0.2,
    num_examples_to_save=10
):
    """
    Evaluates a trained U-Net model on the validation set.
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


    # Evaluation metrics
    total_iou = 0
    total_dice = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    
    # Create results directory
    Path(results_dir).mkdir(exist_ok=True)
    saved_examples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # Flatten for metric calculation
            preds_flat = preds.view(-1)
            labels_flat = (labels > 0.5).view(-1) # Threshold labels to binary and flatten

            # Move to CPU for metric calculation
            preds_flat = preds_flat.cpu()
            labels_flat = labels_flat.cpu()

            # Calculate metrics for the batch
            intersection = (preds_flat & labels_flat).sum().float()
            
            tp = intersection
            fp = (preds_flat & ~labels_flat).sum().float()
            fn = (~preds_flat & labels_flat).sum().float()
            tn = (~preds_flat & ~labels_flat).sum().float()

            total_iou += (tp / (tp + fp + fn + 1e-6)).item()
            total_dice += ((2. * tp) / (2. * tp + fp + fn + 1e-6)).item()
            total_accuracy += ((tp + tn) / (tp + tn + fp + fn + 1e-6)).item()
            total_precision += (tp / (tp + fp + 1e-6)).item()
            total_recall += (tp / (tp + fn + 1e-6)).item()

            # Save some visual examples
            if saved_examples < num_examples_to_save:
                for j in range(images.size(0)):
                    if saved_examples >= num_examples_to_save:
                        break
                    
                    # Select a representative RGB image (e.g., bands 4, 3, 2)
                    # The bands are B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
                    # So B04, B03, B02 are indices 3, 2, 1
                    img_rgb = images[j, [3, 2, 1], :, :].cpu().numpy()
                    # Normalize for visualization
                    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(np.transpose(img_rgb, (1, 2, 0)))
                    axes[0].set_title("Input Image (RGB)")
                    axes[0].axis('off')
                    
                    axes[1].imshow(labels[j, 0, :, :].cpu().numpy(), cmap='gray')
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis('off')

                    axes[2].imshow(preds[j, 0, :, :].cpu().numpy(), cmap='gray')
                    axes[2].set_title("Predicted Mask")
                    axes[2].axis('off')
                    
                    plt.savefig(f"{results_dir}/evaluation_example_{saved_examples}.png")
                    plt.close(fig)
                    saved_examples += 1

    # Calculate average metrics
    num_batches = len(val_loader)
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    logger.info("--- Evaluation Complete ---")
    logger.info(f"  Average IoU: {avg_iou:.4f}")
    logger.info(f"  Average Dice Coefficient: {avg_dice:.4f}")
    logger.info(f"  Average Pixel Accuracy: {avg_accuracy:.4f}")
    logger.info(f"  Average Precision: {avg_precision:.4f}")
    logger.info(f"  Average Recall: {avg_recall:.4f}")
    logger.info(f"---------------------------")
    logger.info(f"Saved {saved_examples} visual examples to '{results_dir}'")


if __name__ == '__main__':
    evaluate_model() 