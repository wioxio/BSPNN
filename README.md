# BSPNN v2: Pathway-based stepforward neural network

## Overview

This package provides a three-step pathway-based prediction pipeline:

1. **Step 1: Primary Prediction**
   - Trains individual pathway models and evaluates their performance.
2. **Step 2: Level 1 Prediction**
   - Trains pathway models across fold datasets and saves pathway-level predictions.
3. **Step 3: Level 2 Prediction**
   - Trains level 2 models using pathway predictions with stepwise forward selection and SHAP-based analysis.

## What's New in v2

- Refactored to a modular package structure:
  - `bspnn_v2/models`
  - `bspnn_v2/callbacks`
  - `bspnn_v2/utils`
  - `bspnn_v2/steps`
- Shared functions are centralized (model builders, callbacks, and utilities).
- Stable CLI entry points are available:
  - `bspnn-v2-step1`
  - `bspnn-v2-step2`
  - `bspnn-v2-step3`

## Installation

Install from the package directory:

```bash
cd "github_scripts_v2"
pip install -e .
```

Or build/install as a standard package:

```bash
cd "github_scripts_v2"
pip install .
```

## Requirements

- Python >= 3.8
- numpy >= 1.19.0
- pandas >= 1.2.0
- tensorflow >= 2.8.0
- keras >= 2.8.0
- scikit-learn >= 0.24.0
- shap >= 0.39.0

## Package Structure

```text
bspnn_v2/
  models/
    model_builders.py
  callbacks/
    early_stopping.py
  utils/
    data_utils.py
  steps/
    step1_primary_prediction.py
    step2_prediction_level1.py
    step3_prediction_level2.py
  cli.py
```

## Usage

### CLI Commands

Run each pipeline step using installed entry points:

```bash
bspnn-v2-step1 --help
bspnn-v2-step2 --help
bspnn-v2-step3 --help
```

### As Python Modules

You can also run step scripts as modules:

```bash
python -m bspnn_v2.steps.step1_primary_prediction --help
python -m bspnn_v2.steps.step2_prediction_level1 --help
python -m bspnn_v2.steps.step3_prediction_level2 --help
```

### Run Examples

After installation, you can run the full pipeline from the command line.

```bash
# Step 1
bspnn-v2-step1 \
  --train_dataN path/to/train_data.pkl \
  --val_dataN path/to/val_data.pkl \
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
bspnn-v2-step2 \
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
bspnn-v2-step3 \
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

## Notes

- This v2 package is based on scripts in `prior_node2vec/scripts/v2`.
- Functionality is preserved while shared code is reorganized for easier maintenance.
