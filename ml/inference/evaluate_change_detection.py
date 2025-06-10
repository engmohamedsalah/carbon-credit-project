import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.training.dataset_patch_pairs import PatchPairDataset
from ml.models.siamese_unet import SiameseUNet

def evaluate_model(model_path, data_csv, batch_size=32):
    """
    Evaluates the change detection model.

    Args:
        model_path (str): Path to the saved model checkpoint.
        data_csv (str): Path to the CSV file with test data pairs.
        batch_size (int): Batch size for the DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Construct absolute path for data_csv
    if not os.path.isabs(data_csv):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_csv = os.path.join(project_root, data_csv)


    # Load test dataset
    try:
        df = pd.read_csv(data_csv)
    except FileNotFoundError:
        print(f"Error: Data CSV not found at {data_csv}")
        sys.exit(1)

    # For evaluation, we use the full balanced dataset, but without augmentation
    test_dataset = PatchPairDataset(df, augment=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load model
    model = SiameseUNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Evaluating on {len(test_dataset)} samples from {data_csv}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(test_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            output = torch.sigmoid(model(img1, img2)) # Apply sigmoid to get probabilities
            
            all_preds.append(output.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    print("\n--- Evaluation Results ---")
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_f1 = 0
    best_threshold = 0
    
    # Calculate initial stats at 0.5 for reference
    preds_binary_half = (all_preds > 0.5).astype(np.uint8)
    precision_half, recall_half, f1_half, _ = precision_recall_fscore_support(
        all_labels, preds_binary_half, average='binary', zero_division=0
    )
    print(f"\nStats at threshold 0.5 (for reference):")
    print(f"Precision: {precision_half:.4f}, Recall: {recall_half:.4f}, F1 Score: {f1_half:.4f}")


    for thresh in thresholds:
        preds_binary = (all_preds > thresh).astype(np.uint8)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds_binary, average='binary', zero_division=0
        )
        
        print(f"\nThreshold: {thresh:.1f} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print("\n--- Summary ---")
    print(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Change Detection Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model .pth file.')
    parser.add_argument('--data_csv', type=str, required=True, default='ml/data/sentinel2_annual_pairs_balanced.csv', help='Path to the balanced data CSV.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_csv, args.batch_size) 