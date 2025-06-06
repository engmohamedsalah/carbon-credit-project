# ml/training/train_time_series.py

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rasterio
from tqdm import tqdm
import glob
from datetime import datetime
import itertools

# Add project root to path if necessary (adjust based on your structure)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.convlstm_model import ConvLSTM
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# TODO: Adjust these paths and parameters as needed
DATA_DIR = "/home/ubuntu/carbon_credit_project/ml/data/prepared/change_detection" # Use data prepared for change detection for now, adapt dataset class
MODEL_SAVE_DIR = "/home/ubuntu/carbon_credit_project/ml/models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "convlstm_forest_predictor.pth")

# Training Parameters
INPUT_CHANNELS = 4 # R, G, B, NIR
HIDDEN_CHANNELS = [64, 64, 128] # Hidden channels per ConvLSTM layer
KERNEL_SIZES = [(3, 3), (3, 3), (3, 3)] # Kernel size per layer
NUM_LSTM_LAYERS = 3
OUTPUT_CHANNELS = 1 # Binary output (e.g., forest/non-forest or change/no-change for the last step)
SEQ_LENGTH = 5 # Number of time steps in the input sequence
BATCH_SIZE = 4
NUM_EPOCHS = 20 # Adjust as needed
LEARNING_RATE = 0.001
PATCH_SIZE = 64 # Size of the image patches to train on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Definition ---
class SatelliteTimeSeriesDataset(Dataset):
    """Dataset for loading sequences of satellite image patches and corresponding labels."""
    def __init__(self, data_dir, seq_length=5, patch_size=64, bands=["B02", "B03", "B04", "B08"]):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.bands = bands
        self.image_stack_dir = os.path.join(data_dir, "image_stacks")
        self.label_dir = os.path.join(data_dir, "change_labels") # Using change labels for now

        self.image_files = sorted(glob.glob(os.path.join(self.image_stack_dir, "*_stack.tif")))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, "*_change_label.tif")))

        if not self.image_files or not self.label_files:
            raise FileNotFoundError(f"No image stacks or labels found in {data_dir}")

        # Filter out sequences that don\"t have enough preceding images
        self.valid_indices = [i for i in range(len(self.image_files)) if i >= seq_length - 1]

        if not self.valid_indices:
             raise ValueError(f"Not enough images ({len(self.image_files)}) to form sequences of length {seq_length}")

        # Get dimensions from the first image
        with rasterio.open(self.image_files[0]) as src:
            self.height = src.height
            self.width = src.width

        # Calculate number of patches
        self.num_patches_h = self.height // self.patch_size
        self.num_patches_w = self.width // self.patch_size
        self.num_patches_per_image = self.num_patches_h * self.num_patches_w
        self.total_patches = len(self.valid_indices) * self.num_patches_per_image

        logger.info(f"Found {len(self.image_files)} images, {len(self.label_files)} labels.")
        logger.info(f"Creating sequences of length {self.seq_length}.")
        logger.info(f"Image dimensions: {self.height}x{self.width}")
        logger.info(f"Patch size: {self.patch_size}x{self.patch_size}")
        logger.info(f"Number of patches per image: {self.num_patches_per_image}")
        logger.info(f"Total valid sequences start indices: {len(self.valid_indices)}")
        logger.info(f"Total patches in dataset: {self.total_patches}")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Determine which image sequence and patch this index corresponds to
        seq_start_image_idx_in_valid = idx // self.num_patches_per_image
        patch_idx_in_image = idx % self.num_patches_per_image

        # Get the index of the *last* image in the sequence within the full image_files list
        last_image_idx = self.valid_indices[seq_start_image_idx_in_valid]

        # Calculate patch coordinates
        patch_row = (patch_idx_in_image // self.num_patches_w) * self.patch_size
        patch_col = (patch_idx_in_image % self.num_patches_w) * self.patch_size
        window = Window(patch_col, patch_row, self.patch_size, self.patch_size)

        sequence_patches = []
        # Load the sequence of image patches
        for i in range(self.seq_length):
            image_idx = last_image_idx - (self.seq_length - 1 - i)
            image_path = self.image_files[image_idx]
            try:
                with rasterio.open(image_path) as src:
                    patch = src.read(window=window)
                    # Normalize or preprocess patch if needed
                    # Assuming bands are already stacked correctly
                    patch = patch.astype(np.float32) / 65535.0 # Example normalization for uint16
                    sequence_patches.append(torch.from_numpy(patch))
            except Exception as e:
                logger.error(f"Error reading image patch from {image_path} at index {idx}: {e}")
                # Return dummy data or raise error
                return torch.zeros((self.seq_length, len(self.bands), self.patch_size, self.patch_size)), torch.zeros((1, self.patch_size, self.patch_size))

        # Stack sequence patches: (seq_len, channels, height, width)
        sequence_tensor = torch.stack(sequence_patches)

        # Load the corresponding label patch for the *last* image in the sequence
        label_path = self.label_files[last_image_idx] # Assuming labels correspond to images by index
        try:
            with rasterio.open(label_path) as src:
                label_patch = src.read(1, window=window) # Read first band
                label_patch = label_patch.astype(np.int64) # Ensure label is integer type for loss function
                label_tensor = torch.from_numpy(label_patch).unsqueeze(0) # Add channel dim: (1, height, width)
        except Exception as e:
            logger.error(f"Error reading label patch from {label_path} at index {idx}: {e}")
            # Return dummy data or raise error
            return torch.zeros((self.seq_length, len(self.bands), self.patch_size, self.patch_size)), torch.zeros((1, self.patch_size, self.patch_size))

        return sequence_tensor, label_tensor

# --- Training Function ---
def train_convlstm_model(model, dataloader, criterion, optimizer, num_epochs, device):
    """Trains the ConvLSTM model."""
    model.train()  # Set model to training mode
    best_loss = float('inf')

    logger.info(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for i, (sequences, labels) in enumerate(progress_bar):
            sequences = sequences.to(device)
            # Labels need shape (batch, height, width) for BCEWithLogitsLoss if output is (batch, 1, H, W)
            labels = labels.squeeze(1).float().to(device) # Remove channel dim, ensure float

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # Output is prediction for the last time step
            outputs, _ = model(sequences)

            # Ensure output and label shapes match for loss calculation
            # Output: (batch, 1, H, W), Label: (batch, H, W)
            loss = criterion(outputs.squeeze(1), labels) # Squeeze output channel dim

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": running_loss / (i + 1)})

        epoch_loss = running_loss / len(dataloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the model if loss has decreased (simple validation)
        # TODO: Implement proper validation set
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Model improved and saved to {MODEL_SAVE_PATH}")

    logger.info("Finished Training")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Initializing Time Series Training ---")

    # Create dataset and dataloader
    try:
        dataset = SatelliteTimeSeriesDataset(DATA_DIR, seq_length=SEQ_LENGTH, patch_size=PATCH_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to initialize dataset: {e}")
        sys.exit(1)

    # Initialize the model
    model = ConvLSTM(input_dim=INPUT_CHANNELS,
                     hidden_dim=HIDDEN_CHANNELS,
                     kernel_size=KERNEL_SIZES,
                     num_layers=NUM_LSTM_LAYERS,
                     output_dim=OUTPUT_CHANNELS,
                     batch_first=True,
                     bias=True).to(DEVICE)

    # Loss function - Use BCEWithLogitsLoss for binary segmentation with single output channel
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_convlstm_model(model, dataloader, criterion, optimizer, NUM_EPOCHS, DEVICE)

    logger.info("--- Time Series Training Script Finished ---")

