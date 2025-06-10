# ml/training/train_change_detection.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import logging
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import random
import pandas as pd

from ml.models.siamese_unet import SiameseUNet
from ml.training.dataset_patch_pairs import PatchPairDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PATCH_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.005
NUM_EPOCHS = 20  # Increased from 2 to 20 for better training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CSV = "ml/data/sentinel2_annual_pairs_balanced.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "change_detection_siamese_unet.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# Example normalization (adapt as needed)
# The data is already converted to a tensor in the dataset, so ToTensor() is not needed here.
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.2])
])


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def train_change_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_samples = 0
        print(f"Epoch {epoch+1}/{num_epochs} - Training started", flush=True)
        for batch_idx, (img1, img2, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            print(f"Processing batch {batch_idx+1}", flush=True)
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            if label.dim() == 3:
                label = label.unsqueeze(1) # Ensure label has channel dim: [B, 1, H, W]
            optimizer.zero_grad()
            outputs = model(img1, img2)
            target = label # Use the real label mask
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img1.size(0)
            num_samples += img1.size(0)
        train_loss /= max(num_samples, 1)
        print(f"Epoch {epoch+1} training complete. Train Loss: {train_loss:.4f}", flush=True)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_samples = 0
        with torch.no_grad():
            for batch_idx, (img1, img2, label) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")):
                print(f"Validation batch {batch_idx+1}", flush=True)
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                if label.dim() == 3:
                    label = label.unsqueeze(1)
                outputs = model(img1, img2)
                target = label # Use the real label mask
                loss = criterion(outputs, target)
                val_loss += loss.item() * img1.size(0)
                num_val_samples += img1.size(0)
        val_loss /= max(num_val_samples, 1)
        print(f"Epoch {epoch+1} validation complete. Val Loss: {val_loss:.4f}", flush=True)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")


def main():
    logger.info(f"Using device: {DEVICE}")

    # --- Data Loading and Balancing ---
    full_df = pd.read_csv(DATA_CSV)
    
    change_indices = []
    nochange_indices = []

    print("Analyzing labels for balancing...")
    for i, row in tqdm(full_df.iterrows(), total=len(full_df)):
        label_path = row['label']
        if not isinstance(label_path, str) or not os.path.exists(label_path):
            continue
        label = np.load(label_path)
        if np.any(label > 0):
            change_indices.append(i)
        else:
            nochange_indices.append(i)

    n_change = len(change_indices)
    n_nochange = len(nochange_indices)
    
    print(f"Found {n_change} 'change' patches and {n_nochange} 'no change' patches.")
    n_balanced = min(n_change, n_nochange)
    print(f"Using {n_balanced} from each group for balanced training.")

    if n_balanced == 0:
        print("Not enough samples for balancing. Using all available data.")
        balanced_indices = list(range(len(full_df)))
    else:
        sampled_change_indices = random.sample(change_indices, n_balanced)
        sampled_nochange_indices = random.sample(nochange_indices, n_balanced)
        balanced_indices = sampled_change_indices + sampled_nochange_indices
    
    balanced_df = full_df.iloc[balanced_indices].sample(frac=1).reset_index(drop=True)

    print(f"Total balanced training patches: {len(balanced_df)}")
    
    # --- Dataset and DataLoader setup ---
    val_split = 0.2
    val_size = int(len(balanced_df) * val_split)
    train_size = len(balanced_df) - val_size
    
    train_df = balanced_df.iloc[:train_size]
    val_df = balanced_df.iloc[train_size:]

    # Create two dataset instances, one with and one without augmentation
    train_dataset = PatchPairDataset(train_df, augment=True, transform=transform, label_column='label')
    val_dataset = PatchPairDataset(val_df, augment=False, transform=transform, label_column='label')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model, loss, optimizer
    model = SiameseUNet(in_channels=4, out_channels=1).to(DEVICE)
    criterion = FocalLoss(alpha=0.5, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting training...")
    print("Starting training loop...", flush=True)
    train_change_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, MODEL_SAVE_PATH)
    logger.info(f"Training finished. Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()


