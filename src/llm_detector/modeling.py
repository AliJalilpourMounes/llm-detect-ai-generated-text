# src/llm_detector/modeling.py
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)

def load_tokenizer(model_name):
    """Loads the tokenizer for the specified model."""
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}'. Error: {e}")
        raise SystemExit("Tokenizer loading failed.")

def tokenize_function(examples, tokenizer, max_length):
    """Applies tokenizer to text data."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False, # Pad dynamically later with DataCollator
        max_length=max_length,
        return_token_type_ids=False, # Not needed for DeBERTa v3/RoBERTa
    )

def tokenize_dataset(dataset, tokenizer, max_length, remove_columns):
    """Tokenizes a Hugging Face Dataset."""
    logger.info(f"Tokenizing dataset. Columns to remove: {remove_columns}")
    try:
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, max_length),
            batched=True,
            remove_columns=remove_columns # Pass the specific columns to remove
        )
        logger.info("Tokenization complete.")
        return tokenized_dataset
    except KeyError as e:
         logger.error(f"Error during tokenization: Column {e} not found. Available columns: {dataset.column_names}")
         raise SystemExit("Tokenization failed due to missing column.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during tokenization: {e}")
         raise SystemExit("Tokenization failed.")


def load_model(model_name, num_labels=2):
    """Loads the sequence classification model."""
    logger.info(f"Loading model: {model_name}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}'. Error: {e}")
        raise SystemExit("Model loading failed.")

def get_data_collator(tokenizer):
    """Returns a data collator with padding."""
    return DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    """Computes AUC score for evaluation."""
    predictions, labels = eval_pred
    # Assuming predictions are logits
    if isinstance(predictions, tuple): # Handle potential tuple output
        logits = predictions[0]
    else:
        logits = predictions

    # Apply softmax to logits to get probabilities for class 1
    # Use torch for potential GPU acceleration if inputs are tensors, else numpy
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    else: # Assume numpy array
         # Manual softmax for numpy
         exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True)) # Improve numerical stability
         probs_all = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
         probs = probs_all[:, 1]


    try:
        auc = roc_auc_score(labels, probs)
    except ValueError as e:
        # This might happen if only one class is present in the eval batch
        logger.warning(f"roc_auc_score failed. This might happen if only one class is present in the batch. Error: {e}. Returning AUC=0.5")
        auc = 0.5
    return {'auc': auc}