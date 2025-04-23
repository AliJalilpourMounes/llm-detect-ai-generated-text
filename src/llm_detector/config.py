# src/llm_detector/config.py
import torch

class CFG:
    # Model Configuration
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512        # Max sequence length for tokenizer

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 8    # Adjust based on GPU VRAM
    EVAL_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2    # Effective batch size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
    LEARNING_RATE = 2e-5
    EPOCHS = 3              # Number of training epochs
    WEIGHT_DECAY = 0.01
    FP16 = torch.cuda.is_available() # Enable Mixed Precision if GPU available

    # Data Configuration
    VAL_SPLIT = 0.2         # Use 20% of training data for validation
    RANDOM_SEED = 42

    # Paths (These should ideally be passed as arguments or loaded from env/config file)
    # Default placeholder paths - override with script arguments
    DEFAULT_COMPETITION_DATA_PATH = "/kaggle/input/llm-detect-ai-generated-text/"
    DEFAULT_EXTERNAL_DATA_PATH = "/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv"
    DEFAULT_OUTPUT_DIR = "llm-detect-output" # Default output directory for model/results

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001

    # Evaluation
    METRIC_FOR_BEST_MODEL = "auc" # Metric to monitor

    # Logging
    LOGGING_STEPS = 50

    # Saving
    SAVE_TOTAL_LIMIT = 1 # Only keep the best checkpoint