import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class PatchPairDataset(Dataset):
    def __init__(self, data, transform=None, label_column='label', augment=False):
        """
        Args:
            data (str or pd.DataFrame): Path to CSV or a pandas DataFrame.
            transform (callable, optional): Optional transform.
            label_column (str, optional): Name of the label column.
            augment (bool, optional): Whether to apply data augmentation.
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("data must be a filepath string or a pandas DataFrame")

        self.transform = transform
        self.label_column = label_column
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = np.load(row['patch1'])  # (C, H, W)
        img2 = np.load(row['patch2'])
        # Convert to torch tensors
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        label = None
        if self.label_column and self.label_column in row:
            label_path = row[self.label_column]
            if isinstance(label_path, str) and os.path.exists(label_path):
                label_arr = np.load(label_path)
                label = torch.from_numpy(label_arr).float()

        # The training loop expects a tuple, not a dictionary.
        # Let's return (img1, img2, label)
        # Ensure label is returned even if it's None, the loader will handle it.
        
        # Apply augmentation first
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img1 = torch.flip(img1, dims=[2])
                img2 = torch.flip(img2, dims=[2])
                if label is not None:
                    label = torch.flip(label, dims=[1])
            # Random vertical flip
            if random.random() > 0.5:
                img1 = torch.flip(img1, dims=[1])
                img2 = torch.flip(img2, dims=[1])
                if label is not None:
                    label = torch.flip(label, dims=[0])
            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img1 = torch.rot90(img1, k, dims=[1,2])
                img2 = torch.rot90(img2, k, dims=[1,2])
                if label is not None:
                    label = torch.rot90(label, k, dims=[0,1])
        
        # Apply transform after augmentation
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        if label is None:
            # Return a zero tensor if no label is found, to avoid issues in the loader
            # This assumes a fixed size for labels. Let's use the patch size.
            # This case should ideally not be hit in supervised training.
            label = torch.zeros((128, 128))

        return img1, img2, label 