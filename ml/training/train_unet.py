import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from ml.models.unet import UNet
from ml.training.dataset_patches import PatchDataset
from ml.utils.losses import FocalLoss
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_unet(
    csv_file='ml/data/forest_cover_patches_balanced.csv',
    model_save_path='ml/models/forest_cover_unet_focal_balanced_augmented.pth',
    epochs=25,
    batch_size=16,
    learning_rate=1e-4,
    val_split=0.2
):
    """
    Train a U-Net model for forest cover classification with data augmentation.
    """
    # Create model save directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

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
    
    # Manually split indices for training and validation
    dataset_size = len(df)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.seed(42) # for reproducibility
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create separate datasets for training (with augmentation) and validation (without)
    train_dataset = PatchDataset(df.iloc[train_indices], augment=True)
    val_dataset = PatchDataset(df.iloc[val_indices], augment=False)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, optimizer, and loss function from our best performing run
    model = UNet(n_channels=12, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs} - Training started")
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'Train Loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix(Val_Loss=f'{val_loss / (val_pbar.n + 1):.4f}')
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model improved. Saved to {model_save_path}")

    logging.info("Training finished.")

if __name__ == '__main__':
    train_unet() 