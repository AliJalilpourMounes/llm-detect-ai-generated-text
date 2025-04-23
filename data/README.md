# Data Acquisition

The data for this project comes from the Kaggle competition "[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)".

## Required Files:

*   `train_essays.csv`
*   `test_essays.csv`
*   `sample_submission.csv`

## Optional External Data:

*   `train_v2_drcat_02.csv` from the "[Daigt V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)" Kaggle Dataset.

## Setup:

1.  Download the competition data from the Kaggle competition page.
2.  Download the external dataset from its Kaggle page.
3.  Place the downloaded CSV files into a directory accessible by the scripts. The default expected location is a parent `input` directory structure relative to where you run the scripts, mimicking Kaggle:
    ```
    ../input/llm-detect-ai-generated-text/train_essays.csv
    ../input/llm-detect-ai-generated-text/test_essays.csv
    ../input/llm-detect-ai-generated-text/sample_submission.csv
    ../input/daigt-v2-train-dataset/train_v2_drcat_02.csv
    ```
4.  Alternatively, you can specify the paths to these files using command-line arguments when running `train.py` and `predict.py`. See `scripts/train.py --help` for details.