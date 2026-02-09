# BSPNN: Pathway-based stepforward neural network
## Overview

This package provides a three-step pipeline for pathway-based prediction:

1. **Step 1: Primary Prediction** - Trains individual pathway models and evaluates their performance
2. **Step 2: Level 1 Prediction** - Trains pathway models on N fold datasets and saves predictions
3. **Step 3: Level 2 Prediction** - Trains level 2 models using pathway predictions with stepwise forward selection and SHAP importance analysis

## Installation

```bash
pip install -e .
```

Or install from source:

```bash
git clone https://github.com/wioxio/BSPNN
cd BSPNN
pip install -e .
```

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.2.0
- tensorflow >= 2.4.0
- keras >= 2.4.0
- scikit-learn >= 0.24.0
- shap >= 0.39.0

## Usage

### As a Python Package

```python
from bspnn import (
    step1_primary_prediction,
    step2_prediction_level1,
    step3_prediction_level2
)

# Step 1: Train pathway models
step1_primary_prediction(
    train_dataN="path/to/train_data.pkl",
    test_dataN="path/to/test_data.pkl",
    pathwayN="path/to/pathways.csv",
    Nlayers=3,
    Nnodes=128,
    optimizer="adam",
    epoch_p=100,
    patience=10,
    batch_size_p=32,
    pathway_start_i=0,
    pathway_end_i=19,
    output_prefix="output",
    runN="results/run1"
)

# Step 2: Generate pathway predictions
step2_prediction_level1(
    train_dataNs=["train1.pkl", "train2.pkl"],
    val_dataNs=["val1.pkl", "val2.pkl"],
    test_dataNs=["test1.pkl", "test2.pkl"],
    pathwayN="path/to/pathways.csv",
    Nlayers=3,
    Nnodes=128,
    optimizer="adam",
    epoch_p=100,
    patience=10,
    batch_size_p=32,
    path_index_fileN="path/to/pathway_indices.csv",
    output_prefix="output",
    runN="results/run1"
)

# Step 3: Level 2 prediction with stepwise selection
step3_prediction_level2(
    cv_train_dataNs=["train1.pkl", "train2.pkl"],
    cv_val_dataNs=["val1.pkl", "val2.pkl"],
    cv_test_dataNs=["test1.pkl", "test2.pkl"],
    cv_train_pathway_prediction_dataNs=["train1", "train2"],
    cv_val_pathway_prediction_dataNs=["val1", "val2"],
    cv_test_pathway_prediction_dataNs=["test1", "test2"],
    pathwayN="path/to/pathways.csv",
    Nlayers=3,
    Nnodes=128,
    optimizer="adam",
    epoch_p=100,
    patience=10,
    batch_size_p=32,
    path_index_fileN="path/to/pathway_indices.csv",
    output_prefix="output",
    runN="results/run1",
    trial=1
)
```

### As Command-Line Scripts

After installation, you can use the command-line interfaces:

```bash
# Step 1
bspnn-step1 \
    --train_dataN path/to/train_data.pkl \
    --test_dataN path/to/test_data.pkl \
    --pathwayN path/to/pathways.csv \
    --Nlayers 3 \
    --Nnodes 128 \
    --optimizer adam \
    --epoch 100 \
    --patience 10 \
    --batch_size 32 \
    --pathway_start_i 0 \
    --pathway_end_i 19 \
    --output_prefix output \
    --runN results/run1

# Step 2
bspnn-step2 \
    --train_dataNs train1.pkl train2.pkl \
    --val_dataNs val1.pkl val2.pkl \
    --test_dataNs test1.pkl test2.pkl \
    --pathwayN path/to/pathways.csv \
    --Nlayers 3 \
    --Nnodes 128 \
    --optimizer adam \
    --epoch 100 \
    --patience 10 \
    --batch_size 32 \
    --path_index_fileN path/to/pathway_indices.csv \
    --output_prefix output \
    --runN results/run1

# Step 3
bspnn-step3 \
    --cv_train_dataNs train1.pkl train2.pkl \
    --cv_val_dataNs val1.pkl val2.pkl \
    --cv_test_dataNs test1.pkl test2.pkl \
    --cv_train_pathway_prediction_dataNs train1 train2 \
    --cv_val_pathway_prediction_dataNs val1 val2 \
    --cv_test_pathway_prediction_dataNs test1 test2 \
    --pathwayN path/to/pathways.csv \
    --Nlayers 3 \
    --Nnodes 128 \
    --optimizer adam \
    --epoch 100 \
    --patience 10 \
    --batch_size 32 \
    --path_index_fileN path/to/pathway_indices.csv \
    --output_prefix output \
    --runN results/run1 \
    --trial 1
```

## Package Structure

```
bspnn/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── model_builders.py      # Model architecture builders
├── callbacks/
│   ├── __init__.py
│   └── early_stopping.py      # Custom early stopping callback
├── utils/
│   ├── __init__.py
│   └── data_utils.py          # Data loading and preprocessing utilities
└── steps/
    ├── __init__.py
    ├── step1_primary_prediction.py    # Step 1 implementation
    ├── step2_prediction_level1.py     # Step 2 implementation
    └── step3_prediction_level2.py    # Step 3 implementation
```

## Key Components

### Models

- `make_pathway_model()`: Creates a pathway-specific models
- `make_original_model()`: Creates a standard fully-connected neural network
- `make_level2_model()`: Creates a model that takes pathway predictions as input

### Callbacks

- `EarlyStoppingAtMinLoss`: Custom early stopping that stops when loss reaches minimum or accuracy reaches threshold

### Utilities

- `pickle_data()`: Save data to pickle files
- `normalize_data()`: Normalize data by replacing NaN with 0
- `clean_file_list()`: Clean and split comma-separated file lists
- `split_comma_separated()`: Split comma-separated strings in argument lists

## Data Format

### Input Data

- Training/test data should be pickle files containing pandas DataFrames
- First column should be labels (binary classification)
- Remaining columns should be features (genes)

### Pathway Data

- CSV file with pathways as columns and genes as rows
- Values indicate pathway membership (typically 0 or 1)

## Output

The pipeline generates various output files:

- Pathway accuracy files (CSV)
- Model predictions (pickle files)
- SHAP importance scores (CSV)
- Evaluation metrics (accuracy, sensitivity, specificity, F1, Kappa)

## License

MIT License

## Citation

If you use this package in your research, please cite:

```
@software{bspnn,
  title={BSPNN: Pathway-based stepforward neural network},
  author={Shin, Min-Gyoung},
  year={2026},
  url={https://github.com/wioxio/BSPNN}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
