# ml/scripts/balance_forest_cover_data.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def balance_dataset(
    input_csv='ml/data/forest_cover_patches.csv',
    output_csv='ml/data/forest_cover_patches_5k.csv',
    forest_threshold=10,  # Mean canopy cover percentage to be considered 'forest'
    target_size=5000, # None for fully balanced, or a number for target size
    force_include_all_minority=True
):
    """
    Balances or samples the dataset.
    - If target_size is None, it creates a perfectly balanced dataset by undersampling the majority class.
    - If target_size is a number, it creates a dataset of that size.
    - If force_include_all_minority is True, it ensures all minority class samples are included when target_size is set.
    """
    if not Path(input_csv).exists():
        logger.error(f"Input CSV not found at {input_csv}. Please generate it first.")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} entries from {input_csv}")

    forest_patches = []
    non_forest_patches = []

    logger.info("Classifying patches into 'forest' and 'non-forest'...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing patches"):
        label_path = Path(row['label_path'])
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path}. Skipping.")
            continue
        
        try:
            mask = np.load(label_path)
            
            # Take the first channel if multi-channel
            if mask.ndim == 3:
                mask = mask[0]
                
            if np.mean(mask) > forest_threshold:
                forest_patches.append(row)
            else:
                non_forest_patches.append(row)
        except Exception as e:
            logger.error(f"Could not process label {label_path}: {e}")


    logger.info(f"Found {len(forest_patches)} 'forest' patches.")
    logger.info(f"Found {len(non_forest_patches)} 'non-forest' patches.")

    minority_class = 'forest' if len(forest_patches) < len(non_forest_patches) else 'non_forest'
    
    if target_size is None:
        # Fully balanced undersampling
        num_minority = len(forest_patches) if minority_class == 'forest' else len(non_forest_patches)
        if num_minority == 0:
            logger.error(f"No {minority_class} patches found. Cannot create a balanced dataset. Exiting.")
            return

        forest_sample = random.sample(forest_patches, num_minority)
        non_forest_sample = random.sample(non_forest_patches, num_minority)
        
        balanced_df = pd.concat([pd.DataFrame(forest_sample), pd.DataFrame(non_forest_sample)])
        final_size = len(balanced_df)

    else:
        # Create a dataset of a specific size
        if force_include_all_minority:
            minority_patches = forest_patches if minority_class == 'forest' else non_forest_patches
            majority_patches = non_forest_patches if minority_class == 'forest' else forest_patches
            
            num_minority = len(minority_patches)
            num_majority_needed = target_size - num_minority

            if num_majority_needed < 0:
                logger.warning(f"Target size {target_size} is smaller than the number of minority samples {num_minority}. Using a random sample of minority class.")
                minority_sample = random.sample(minority_patches, target_size)
                majority_sample = []
            elif num_majority_needed > len(majority_patches):
                 logger.warning(f"Not enough majority patches to meet target size. Using all available patches.")
                 majority_sample = majority_patches
                 num_minority_needed = target_size - len(majority_sample)
                 minority_sample = random.sample(minority_patches, num_minority_needed)
            else:
                minority_sample = minority_patches
                majority_sample = random.sample(majority_patches, num_majority_needed)

            balanced_df = pd.concat([pd.DataFrame(minority_sample), pd.DataFrame(majority_sample)])
            final_size = len(balanced_df)
        else:
            # Simple random sample from the whole dataset
            balanced_df = df.sample(n=target_size, random_state=42)
            final_size = target_size


    # Concatenate and shuffle
    final_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(output_csv, index=False)
    logger.info(f"Successfully created dataset with {final_size} entries at {output_csv}")

if __name__ == '__main__':
    balance_dataset() 