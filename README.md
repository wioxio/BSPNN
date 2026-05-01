# BSPNN: Pathway-based stepforward neural network

## Overview

This package provides a three-step pathway-based prediction pipeline:

1. **Step 1: Primary Prediction**
   - Trains individual pathway models and evaluates their performance in each fold.
   - Requires paired `--train_dataN` and `--val_dataN` (run it for each fold). Each argument is resolved to **`{runN}/data/<basename>`** (same layout as steps 2 and 3).
2. **Step 2: Level 1 Prediction**
   - Trains pathway models across fold datasets and saves pathway-level predictionsfor each fold.
3. **Step 3: Level 2 Prediction**
   - Trains level 2 models using pathway predictions with BNN, SPNN, BSPNN and SHAP-based analysis.


## Installation

Install from the package directory:

```bash
cd BSPNN-main
pip install -e .
```

Or build/install as a standard package:

```bash
cd BSPNN-main
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

## Run directory layout (`--runN`)

All steps use a single run root directory passed as `--runN`. For **step 2** and **step 3**, input pickles are resolved under that root as follows:

| Location | Contents |
|----------|----------|
| `{runN}/data/` | Per-fold expression matrices (pickles of `pandas.DataFrame`: label in column 0, features in columns 1+). **Steps 1, 2, and 3** take **basenames** (or paths) on the CLI; each is opened as `{runN}/data/<basename>` (only the filename is used). |
| `{runN}/prediction_level1/` | Pathway-level prediction outputs from **step 2**. Each file is named `pi<pathway_index>_<stem>.pkl`, where `<stem>` is the stem of the corresponding `{runN}/data/` pickle (e.g. `fold1_train.pkl` → stem `fold1_train`). **Step 3** reads these via `--cv_train_pathway_prediction_dataNs`, `--cv_val_pathway_prediction_dataNs`, and `--cv_test_pathway_prediction_dataNs`: pass the **same stems** (basenames; `.pkl` optional). Files are read as `{runN}/prediction_level1/pi<k>_<stem>.pkl`. |

Step 1 writes CSV summaries under `{runN}/` (not under `data/`).

Steps 1–3 create `{runN}/data/` and `{runN}/prediction_level1/` if missing (you still need the pickle files in place before running).

## Package Structure

```text
bspnn/
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
bspnn-step1 --help
bspnn-step2 --help
bspnn-step3 --help
```

### As Python Modules

You can also run step scripts as modules:

```bash
python -m bspnn.steps.step1_primary_prediction --help
python -m bspnn.steps.step2_prediction_level1 --help
python -m bspnn.steps.step3_prediction_level2 --help
```

### Run Examples

After installation, you can run the full pipeline from the command line. **Step 1** train/val pickles live under `{runN}/data/` (basename on CLI). For **step 3**, stems in `--cv_*_pathway_prediction_dataNs` must match the `pi<k>_<stem>.pkl` files under `{runN}/prediction_level1/` produced by step 2 (stem = basename of the corresponding `{runN}/data/` pickle without `.pkl`).

```bash
# Step 1 (place train/val pickles under {runN}/data/; pass basenames only)
# Data format: label iloc[:, 0] (float32, binary); features iloc[:, 1:]
bspnn-step1 \
  --train_dataN GSE254185_fold1_train.pkl \
  --val_dataN GSE254185_fold1_val.pkl \
  --pathwayN path/to/pathways.csv \
  --Nlayers 3 \
  --Nnodes 128 \
  --optimizer adam \
  --epoch 100 \
  --patience 10 \
  --batch_size 32 \
  --pathway_start_i 0 \
  --pathway_end_i 200 \
  --output_prefix output \
  --runN path/to/run_root

# Step 2 (place fold pickles under {runN}/data/ first; pass basenames only)
bspnn-step2 \
  --train_dataNs fold1_train.pkl fold2_train.pkl \
  --val_dataNs fold1_val.pkl fold2_val.pkl \
  --test_dataNs fold1_test.pkl fold2_test.pkl \
  --pathwayN path/to/pathways.csv \
  --Nlayers 3 \
  --Nnodes 128 \
  --optimizer adam \
  --epoch 100 \
  --patience 10 \
  --batch_size 32 \
  --path_index_fileN path/to/sorted_topN_pathways.csv \
  --output_prefix output \
  --runN path/to/run_root

# Step 3 (same {runN}/data/ basenames; pathway stems must match step 2 output names)
bspnn-step3 \
  --cv_train_dataNs fold1_train.pkl fold2_train.pkl \
  --cv_val_dataNs fold1_val.pkl fold2_val.pkl \
  --cv_test_dataNs fold1_test.pkl fold2_test.pkl \
  --cv_train_pathway_prediction_dataNs fold1_train_prediction_level1.pkl fold2_train_prediction_level1.pkl \
  --cv_val_pathway_prediction_dataNs fold1_val_prediction_level1.pkl fold2_val_prediction_level1.pkl \
  --cv_test_pathway_prediction_dataNs fold1_test_prediction_level1.pkl fold2_test_prediction_level1.pkl \
  --pathwayN path/to/pathways.csv \
  --Nlayers 3 \
  --Nnodes 128 \
  --optimizer adam \
  --epoch 100 \
  --patience 10 \
  --batch_size 32 \
  --path_index_fileN path/to/pathway_indices.csv \
  --output_prefix output \
  --runN path/to/run_root \
  --trial 1
```

