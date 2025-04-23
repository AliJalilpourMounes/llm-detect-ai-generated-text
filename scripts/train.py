# scripts/train.py
import argparse
import os
import logging
import sys

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

from src.llm_detector.config import CFG
from src.llm_detector.utils import seed_everything, check_gpu, clean_memory
from src.llm_detector.data_processing import load_and_prepare_data
from src.llm_detector.modeling import (
    load_tokenizer,
    load_model,
    tokenize_dataset,
    get_data_collator,
    compute_metrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for LLM text detection.")
    parser.add_argument("--competition_data_dir", type=str, default=CFG.DEFAULT_COMPETITION_DATA_PATH,
                        help="Directory containing train_essays.csv and test_essays.csv")
    parser.add_argument("--external_data_path", type=str, default=CFG.DEFAULT_EXTERNAL_DATA_PATH,
                        help="Path to the external CSV data file (optional). Set to empty string '' to disable.")
    parser.add_argument("--output_dir", type=str, default=CFG.DEFAULT_OUTPUT_DIR,
                        help="Directory to save model checkpoints and logs.")
    parser.add_argument("--model_name", type=str, default=CFG.MODEL_NAME, help="Hugging Face model name.")
    parser.add_argument("--max_length", type=int, default=CFG.MAX_LENGTH, help="Max sequence length.")
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=CFG.TRAIN_BATCH_SIZE, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=CFG.EVAL_BATCH_SIZE, help="Evaluation batch size per device.")
    parser.add_argument("--grad_accum_steps", type=int, default=CFG.GRAD_ACCUM_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=CFG.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=CFG.WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=CFG.RANDOM_SEED, help="Random seed.")
    parser.add_argument("--val_split", type=float, default=CFG.VAL_SPLIT, help="Validation split ratio.")
    parser.add_argument("--fp16", action='store_true', help="Enable mixed precision training (if GPU available).")
    parser.add_argument("--no_fp16", dest='fp16', action='store_false', help="Disable mixed precision training.")
    parser.set_defaults(fp16=CFG.FP16) # Default based on config/GPU check

    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. Setup ---
    logger.info("Starting training script...")
    seed_everything(args.seed)
    device = check_gpu()
    use_fp16 = args.fp16 and device.type == 'cuda'
    if use_fp16:
         logger.info("FP16/Mixed Precision Training Enabled.")
    else:
         logger.info("FP16/Mixed Precision Training Disabled.")


    # Handle optional external data path
    external_data_path = args.external_data_path if args.external_data_path else None

    # --- 2. Load and Prepare Data ---
    train_dataset, val_dataset = load_and_prepare_data(
        competition_data_path=args.competition_data_dir,
        external_data_path=external_data_path,
        val_split=args.val_split,
        random_seed=args.seed
    )
    clean_memory() # Clean up DataFrames inside load_and_prepare_data

    # --- 3. Tokenization ---
    tokenizer = load_tokenizer(args.model_name)
    data_collator = get_data_collator(tokenizer)

    # Define columns to remove AFTER splitting and converting to Dataset
    # Base columns: 'text'. Potentially '__index_level_0__' if pandas adds it.
    train_cols_to_remove = [col for col in ['text', '__index_level_0__'] if col in train_dataset.column_names]
    val_cols_to_remove = [col for col in ['text', '__index_level_0__'] if col in val_dataset.column_names]

    logger.info(f"Columns to remove from train_dataset: {train_cols_to_remove}")
    logger.info(f"Columns to remove from val_dataset: {val_cols_to_remove}")


    train_tokenized = tokenize_dataset(train_dataset, tokenizer, args.max_length, train_cols_to_remove)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, args.max_length, val_cols_to_remove)
    clean_memory() # Clean up original datasets

    # --- 4. Model Loading & Training Setup ---
    model = load_model(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=CFG.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        fp16=use_fp16,
        logging_strategy="steps",
        logging_steps=CFG.LOGGING_STEPS,
        report_to="none", # Change to "tensorboard", "wandb" etc. if needed
        save_total_limit=CFG.SAVE_TOTAL_LIMIT,
        seed=args.seed,
        # dataloader_num_workers=2, # Consider adding as an arg if needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=CFG.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=CFG.EARLY_STOPPING_THRESHOLD
        )]
    )

    # --- 5. Train the Model ---
    logger.info("Starting model training...")
    clean_memory() # Clean before training starts
    train_result = trainer.train()

    # --- 6. Save Metrics & Evaluate ---
    logger.info("Training complete.")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(os.path.join(args.output_dir, "best_model")) # Save final best model explicitly
    logger.info(f"Best model saved to {os.path.join(args.output_dir, 'best_model')}")
    logger.info(f"Best checkpoint was: {trainer.state.best_model_checkpoint}")


    logger.info("Evaluating on validation set using the best model...")
    eval_metrics = trainer.evaluate(eval_dataset=val_tokenized) # Ensure evaluation uses the best model
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    logger.info(f"Validation Metrics (Best Model): {eval_metrics}")

    # --- 7. Clean up ---
    clean_memory()
    logger.info("Training script finished.")

if __name__ == "__main__":
    main()