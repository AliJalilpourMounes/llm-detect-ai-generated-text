# scripts/predict.py
import argparse
import os
import logging
import sys
import pandas as pd
import torch

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import Trainer, TrainingArguments # Need Trainer for .predict()
from datasets import Dataset

from src.llm_detector.config import CFG # Import CFG for defaults if needed
from src.llm_detector.utils import check_gpu, clean_memory
from src.llm_detector.data_processing import load_test_data
from src.llm_detector.modeling import (
    load_tokenizer,
    load_model,
    tokenize_dataset,
    get_data_collator
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions using a trained model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory (containing pytorch_model.bin, config.json, etc.).")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Name or path of the tokenizer. If None, tries to load from model_path.")
    parser.add_argument("--competition_data_dir", type=str, default=CFG.DEFAULT_COMPETITION_DATA_PATH,
                        help="Directory containing test_essays.csv.")
    parser.add_argument("--output_csv", type=str, default="submission.csv",
                        help="Path to save the output submission CSV file.")
    parser.add_argument("--max_length", type=int, default=CFG.MAX_LENGTH, help="Max sequence length for tokenizer.")
    parser.add_argument("--batch_size", type=int, default=CFG.EVAL_BATCH_SIZE, help="Batch size for prediction.")
    parser.add_argument("--fp16", action='store_true', help="Enable FP16 inference (if GPU available).")
    parser.add_argument("--no_fp16", dest='fp16', action='store_false', help="Disable FP16 inference.")
    parser.set_defaults(fp16=CFG.FP16) # Default based on config/GPU check


    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. Setup ---
    logger.info("Starting prediction script...")
    device = check_gpu()
    use_fp16 = args.fp16 and device.type == 'cuda'

    # --- 2. Load Data ---
    test_df = load_test_data(args.competition_data_dir)
    test_dataset = Dataset.from_pandas(test_df)

    # --- 3. Load Model and Tokenizer ---
    tokenizer_path = args.tokenizer_name if args.tokenizer_name else args.model_path
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(args.model_path) # Assumes model saved with AutoModel.save_pretrained()
    data_collator = get_data_collator(tokenizer)

    # --- 4. Tokenize Test Data ---
    # Only remove 'text', keep 'id' for submission. '__index_level_0__' might be present.
    test_cols_to_remove = [col for col in ['text', '__index_level_0__'] if col in test_dataset.column_names]
    logger.info(f"Columns to remove from test_dataset: {test_cols_to_remove}")
    test_tokenized = tokenize_dataset(test_dataset, tokenizer, args.max_length, test_cols_to_remove)

    # --- 5. Setup Trainer for Prediction ---
    # We use Trainer's predict method for convenience, even without training args like LR etc.
    # Need minimal TrainingArguments. Output dir is temporary unless needed.
    predict_args = TrainingArguments(
        output_dir="./temp_predict_output", # Temporary directory
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=False,
        do_predict=True,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=predict_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. Generate Predictions ---
    logger.info("Generating predictions on the test set...")
    clean_memory()

    predictions_output = trainer.predict(test_tokenized)
    test_logits = predictions_output.predictions

    # Convert logits to probabilities (probability of being AI-generated, i.e., class 1)
    if isinstance(test_logits, tuple): # Handle potential tuple output
         test_logits = test_logits[0]

    test_probs = torch.tensor(test_logits, dtype=torch.float32).softmax(dim=-1)[:, 1].numpy()

    # --- 7. Create Submission File ---
    logger.info(f"Creating submission file: {args.output_csv}")
    # Ensure IDs are correctly aligned. Reloading test_df is safest.
    if 'id' not in test_df.columns:
         logger.error("Original test DataFrame lost 'id' column somehow.")
         # Fallback: try getting from dataset if 'id' wasn't removed during tokenization
         try:
             test_ids = test_tokenized['id'] # Requires 'id' was NOT in test_cols_to_remove
             logger.warning("Using 'id' column from tokenized dataset. Ensure it was not removed.")
         except KeyError:
             raise SystemExit("Cannot retrieve test IDs for submission file.")
    else:
        test_ids = test_df['id']


    submission_df = pd.DataFrame({
        'id': test_ids,
        'generated': test_probs
    })

    submission_df.to_csv(args.output_csv, index=False)
    logger.info("Submission file created successfully.")
    print("\nSubmission head:")
    print(submission_df.head())

    # --- 8. Clean up ---
    clean_memory()
    # Optionally remove temp dir: import shutil; shutil.rmtree("./temp_predict_output")
    logger.info("Prediction script finished.")

if __name__ == "__main__":
    main()