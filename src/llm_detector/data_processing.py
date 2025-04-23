# src/llm_detector/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data(competition_data_path, external_data_path=None, val_split=0.2, random_seed=42):
    """Loads competition data, optionally merges external data, handles NaNs, and splits."""
    logger.info("Loading data...")

    # Load the primary training data
    train_essays_path = f"{competition_data_path}/train_essays.csv"
    test_essays_path = f"{competition_data_path}/test_essays.csv"
    try:
        train_essays = pd.read_csv(train_essays_path)
        test_essays = pd.read_csv(test_essays_path) # Keep test_essays separate for prediction script
        logger.info(f"Loaded competition train data: {train_essays.shape}")
        logger.info(f"Loaded competition test data: {test_essays.shape}")
        # Rename 'generated' column to 'label' for consistency
        train_essays.rename(columns={'generated': 'label'}, inplace=True)
        train_essays = train_essays[['text', 'label']].copy() # Keep only necessary columns

    except FileNotFoundError:
        logger.error(f"ERROR: Could not find competition data files at {competition_data_path}")
        raise SystemExit("Competition data files not found.")

    # Load and combine external dataset (if path provided)
    if external_data_path:
        try:
            logger.info(f"Attempting to load external dataset from: {external_data_path}")
            external_df = pd.read_csv(external_data_path)

            # Basic validation and selection of external data columns
            if 'text' in external_df.columns and 'label' in external_df.columns:
                external_df = external_df[['text', 'label']].copy()
                logger.info(f"Loaded external dataset with {len(external_df)} rows.")

                # Combine with train_essays
                logger.info("Combining competition train data with external data...")
                train_essays = pd.concat([train_essays, external_df], ignore_index=True)
                logger.info(f"Combined training data shape before cleaning: {train_essays.shape}")

            else:
                 logger.warning(f"External dataset at {external_data_path} missing 'text' or 'label' columns. Skipping merge.")

        except FileNotFoundError:
             logger.warning(f"Optional external dataset not found at {external_data_path}. Proceeding without it.")
        except ValueError as e:
             logger.warning(f"Error processing external dataset: {e}. Skipping merge.")

    # Data Cleaning: Handle potential NaN values in the 'label' column AFTER merge
    nan_labels = train_essays['label'].isnull().sum()
    if nan_labels > 0:
        logger.warning(f"Found {nan_labels} rows with NaN in 'label' column. Dropping these rows.")
        train_essays.dropna(subset=['label'], inplace=True)
        logger.info(f"Shape after dropping NaN labels: {train_essays.shape}")

    # Ensure label column is integer type
    try:
        train_essays['label'] = train_essays['label'].astype(int)
        logger.info("Label column converted to integer type.")
    except Exception as e:
        logger.error(f"Could not convert 'label' column to integer. Error: {e}")
        raise SystemExit("Label conversion failed.")

    logger.info(f"Final combined training data shape for splitting: {train_essays.shape}")

    # Split training data for validation
    logger.info(f"Splitting data (Validation size: {val_split})...")
    if len(train_essays) < 2: # Check if enough data to split
        logger.error("Not enough data to perform train/validation split after processing.")
        raise SystemExit("Insufficient data for splitting.")
    if len(train_essays['label'].unique()) < 2:
        logger.warning("Only one class present in training data after processing. Stratification may fail or be meaningless.")
        # Consider alternative splitting if stratification isn't possible/needed
        train_df, val_df = train_test_split(
            train_essays,
            test_size=val_split,
            random_state=random_seed,
            # stratify=None # Cannot stratify with one class
        )
    else:
        train_df, val_df = train_test_split(
            train_essays,
            test_size=val_split,
            random_state=random_seed,
            stratify=train_essays['label'] # Important for classification
        )


    logger.info(f"Train split shape: {train_df.shape}")
    logger.info(f"Validation split shape: {val_df.shape}")

    # Convert pandas DataFrames to Hugging Face Datasets
    logger.info("Converting DataFrames to Hugging Face Datasets...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    # Test dataset is handled separately in predict.py

    logger.info("Dataset conversion complete.")

    return train_dataset, val_dataset # Return datasets for training

def load_test_data(competition_data_path):
    """Loads only the test data."""
    test_essays_path = f"{competition_data_path}/test_essays.csv"
    try:
        test_essays = pd.read_csv(test_essays_path)
        logger.info(f"Loaded competition test data: {test_essays.shape}")
        # Keep necessary columns ('id', 'text')
        if 'id' not in test_essays.columns or 'text' not in test_essays.columns:
            logger.error("Test data CSV must contain 'id' and 'text' columns.")
            raise SystemExit("Invalid test data format.")
        return test_essays[['id', 'text']]
    except FileNotFoundError:
        logger.error(f"ERROR: Could not find test data file at {test_essays_path}")
        raise SystemExit("Test data file not found.")