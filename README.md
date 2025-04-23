# LLM - Detect AI Generated Text

This repository contains code to train and evaluate a model for the Kaggle competition "[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)". It uses a Transformer model (default: DeBERTa-v3-base) fine-tuned on the competition data, optionally augmented with external datasets.

## Repository Structure
```
llm-detect-ai-generated-text/

├── .gitignore       # Git ignore rules
├── LICENSE          # Project license
├── README.md        # This file
├── requirements.txt # Python dependencies
├── config/          # Configuration files (Optional alternative)
├── data/            # Data directory (see data/README.md for setup)
│   ├── README.md
│   └── .gitkeep
├── notebooks/       # Jupyter notebooks for exploration/prototyping
│   └── llm_detect_baseline_original.ipynb # Original notebook
├── scripts/         # Runnable Python scripts
│   ├── train.py     # Script for training the model
│   └── predict.py   # Script for generating predictions
├── src/             # Source code package
│   └── llm_detector/
│       ├── __init__.py
│       ├── config.py       # Configuration class
│       ├── data_processing.py # Data loading and preprocessing functions
│       ├── modeling.py     # Model/tokenizer loading, tokenization, metrics
│       └── utils.py        # Utility functions (seeding, GPU checks)
└── output/          # Default location for saved models, logs, submissions
    └── .gitkeep
```


## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AliJalilpourMounes/llm-detect-ai-generated-text.git
    cd llm-detect-ai-generated-text
    ```

2.  **Create a Python environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    Follow the instructions in `data/README.md` to download the necessary datasets from Kaggle and place them in the expected directory structure (or note the paths for script arguments).

## Usage

### Training

Run the `train.py` script to fine-tune the model. Key arguments:

*   `--competition_data_dir`: Path to the directory containing `train_essays.csv`.
*   `--external_data_path`: Path to the external dataset CSV (e.g., `train_v2_drcat_02.csv`). Set to `""` to disable external data.
*   `--output_dir`: Where to save model checkpoints, logs, and metrics.
*   `--model_name`: Hugging Face model identifier (e.g., `microsoft/deberta-v3-base`).
*   `--epochs`, `--lr`, `--train_batch_size`, etc.: Hyperparameters.

**Example:**

```bash
python scripts/train.py \
    --competition_data_dir ../input/llm-detect-ai-generated-text \
    --external_data_path ../input/daigt-v2-train-dataset/train_v2_drcat_02.csv \
    --output_dir ./output/deberta-v3-base-run1 \
    --epochs 3 \
    --lr 2e-5 \
    --train_batch_size 8 \
    --grad_accum_steps 2 \
    --fp16 # Add this flag if you have a capable GPU
```

See python scripts/train.py --help for all options.

### Prediction

Run the predict.py script to generate a submission.csv file using a trained model. Key arguments:

    --model_path: Path to the directory containing the fine-tuned model saved by train.py (e.g., ./output/deberta-v3-base-run1/best_model).

    --competition_data_dir: Path to the directory containing test_essays.csv.

    --output_csv: Name of the output submission file.

    --tokenizer_name: (Optional) Specify if tokenizer is different from model path.

    --batch_size: Prediction batch size.

**Example:**
```bash
python scripts/predict.py \
    --model_path ./output/deberta-v3-base-run1/best_model \
    --competition_data_dir ../input/llm-detect-ai-generated-text \
    --output_csv submission_deberta_run1.csv \
    --fp16 # Add this flag if you have a capable GPU
```

See python scripts/predict.py --help for all options.

### Configuration

Hyperparameters and paths can be adjusted via command-line arguments for train.py and predict.py. Default values are defined in src/llm_detector/config.py.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
