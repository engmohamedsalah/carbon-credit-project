import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from ml.models.unet import UNet
from ml.training.dataset_patches import PatchDataset
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def iou_score(output, target):
    """
    Calculates the Intersection over Union (IoU) score for binary segmentation.
    """
    with torch.no_grad():
        output_ = torch.sigmoid(output) > 0.5
        target_ = target > 0.5
        intersection = (output_ & target_).float().sum((1, 2))
        union = (output_ | target_).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def evaluate_model(
    model_path='ml/models/forest_cover_unet_focal_5k_alpha_0.85.pth',
    csv_file='ml/data/forest_cover_patches_5k.csv',
    batch_size=16,
    eval_split=0.2
):
    """
    Evaluates the U-Net model on a subset of the data.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return

    # Check for device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load data
    df = pd.read_csv(csv_file)
    dataset = PatchDataset(df)

    # Split dataset
    dataset_size = len(dataset)
    eval_size = int(dataset_size * eval_split)
    train_size = dataset_size - eval_size # The rest is not used here
    _, eval_dataset = random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    logging.info(f"Evaluation set size: {len(eval_dataset)}")
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = UNet(n_channels=12, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    total_iou = 0

    eval_pbar = tqdm(eval_loader, desc="Evaluating")
    with torch.no_grad():
        for images, labels in eval_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # For metrics calculation
            preds = torch.sigmoid(outputs) > 0.5
            binary_labels = labels > 0.5
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(binary_labels.cpu().numpy().flatten())
            
            # Calculate IoU
            total_iou += iou_score(outputs, labels).item()

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_iou = total_iou / len(eval_loader)

    logging.info("Evaluation Complete")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1 Score:  {f1:.4f}")
    logging.info(f"Mean IoU:  {avg_iou:.4f}")

if __name__ == '__main__':
    evaluate_model() 