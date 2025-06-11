# ml/scripts/check_data_balance.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_balance(
    input_csv='ml/data/forest_cover_patches_balanced.csv',
    forest_threshold=10
):
    """
    Checks the balance of 'forest' vs 'non-forest' patches in a given dataset CSV.
    """
    if not Path(input_csv).exists():
        logger.error(f"Input CSV not found at {input_csv}.")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} entries from {input_csv}")

    forest_count = 0
    non_forest_count = 0

    logger.info("Classifying patches...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing patches"):
        label_path = Path(row['label_path'])
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path}. Skipping.")
            continue
        
        try:
            mask = np.load(label_path)
            
            if mask.ndim == 3:
                mask = mask[0]
                
            if np.mean(mask) > forest_threshold:
                forest_count += 1
            else:
                non_forest_count += 1
        except Exception as e:
            logger.error(f"Could not process label {label_path}: {e}")

    logger.info("--- Dataset Balance ---")
    logger.info(f"'Forest' patches:     {forest_count}")
    logger.info(f"'Non-Forest' patches: {non_forest_count}")
    total_patches = forest_count + non_forest_count
    if total_patches > 0:
        forest_percent = (forest_count / total_patches) * 100
        non_forest_percent = (non_forest_count / total_patches) * 100
        logger.info(f"Total: {total_patches} patches")
        logger.info(f"Forest: {forest_percent:.2f}% | Non-Forest: {non_forest_percent:.2f}%")
    logger.info("-----------------------")


if __name__ == '__main__':
    check_balance(input_csv='ml/data/forest_cover_patches.csv') 