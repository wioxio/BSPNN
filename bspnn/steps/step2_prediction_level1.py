"""
Step 2: Level 1 prediction.

Trains pathway models on multiple datasets and saves predictions.
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import tensorflow as tf

from ..models import make_pathway_model
from ..callbacks import EarlyStoppingAtMinLoss
from ..utils import normalize_data, clean_file_list, pickle_data, configure_gpu


# Configure GPU (TensorFlow 2.x style - no InteractiveSession needed)
configure_gpu()


def step2_prediction_level1(
    train_dataNs,
    val_dataNs,
    test_dataNs,
    pathwayN,
    Nlayers,
    Nnodes,
    optimizer,
    epoch_p,
    patience,
    batch_size_p,
    path_index_fileN,
    output_prefix,
    runN
):
    """
    Train pathway models on multiple datasets and save predictions.
    
    Args:
        train_dataNs: List of training data file names
        val_dataNs: List of validation data file names
        test_dataNs: List of test data file names
        pathwayN: Path to pathway CSV file
        Nlayers: Number of layers (not used but kept for compatibility)
        Nnodes: Number of nodes (not used but kept for compatibility)
        optimizer: Optimizer name
        epoch_p: Maximum number of epochs
        patience: Patience for early stopping
        batch_size_p: Batch size
        path_index_fileN: Path to file containing pathway indices
        output_prefix: Prefix for output files
        runN: Output directory name
    """
    # Clean file lists
    train_dataNs = clean_file_list(train_dataNs)
    val_dataNs = clean_file_list(val_dataNs)
    test_dataNs = clean_file_list(test_dataNs)

    # Handle output_prefix if None or list
    if output_prefix is None:
        output_prefix = "output"
    elif isinstance(output_prefix, list):
        output_prefix = output_prefix[0] if len(output_prefix) > 0 else "output"

    print("Arguments:")
    print("Train data: " + str(train_dataNs))
    print("Val data: " + str(val_dataNs))
    print("Test data: " + str(test_dataNs))
    print("Pathway: " + pathwayN)
    print("Path index file: " + path_index_fileN)
    print("Run: " + runN)
    print(f"Nlayers: {Nlayers}")
    print(f"Nnodes: {Nnodes}")
    print("Optimizer: " + optimizer)
    print(f"epoch_p: {epoch_p}")
    print(f"patience: {patience}")
    print('\n')

    # Create output directory if it doesn't exist
    output_dir = runN + "/prediction_by_fold1/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")

    # Read pathways
    pathways = pd.read_csv(pathwayN, header=0, index_col=0)
    print(f"Loaded {len(pathways.columns)} pathways from {pathwayN}")
    
    # Read pathway indices, no header and index
    pathway_indices = pd.read_csv(path_index_fileN, header=None, index_col=None)
    pathway_indices = pathway_indices.iloc[range(0, 20), 0].values
    pathway_indices = pathway_indices.astype(int)
    print(f"Processing {len(pathway_indices)} pathway indices: {pathway_indices[:5]}...")

    for dataC in range(len(train_dataNs)):
        train_file_path = os.path.join(runN, 'data', train_dataNs[dataC])
        val_file_path = os.path.join(runN, 'data', val_dataNs[dataC])
        test_file_path = os.path.join(runN, 'data', test_dataNs[dataC])
        
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Train data file not found: {train_file_path}")
        if not os.path.exists(val_file_path):
            raise FileNotFoundError(f"Validation data file not found: {val_file_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test data file not found: {test_file_path}")
        
        with open(train_file_path, 'rb') as file:
            train_data_step1 = pickle.load(file)
        with open(val_file_path, 'rb') as file:
            val_data_step1 = pickle.load(file)
        with open(test_file_path, 'rb') as file:
            test_data_step1 = pickle.load(file)

        x_train_step1 = train_data_step1.iloc[:, 1:].values
        x_train_step1 = normalize_data(x_train_step1)
        y_train_step1 = train_data_step1.iloc[:, 0].values
        y_train_step1 = y_train_step1.astype("float32")

        x_val_step1 = val_data_step1.iloc[:, 1:].values
        x_val_step1 = normalize_data(x_val_step1)
        y_val_step1 = val_data_step1.iloc[:, 0].values
        y_val_step1 = y_val_step1.astype("float32")

        x_test_step1 = test_data_step1.iloc[:, 1:].values
        x_test_step1 = normalize_data(x_test_step1)
        y_test_step1 = test_data_step1.iloc[:, 0].values
        y_test_step1 = y_test_step1.astype("float32")

        print(f'########## Processing dataset {dataC+1}/{len(test_dataNs)}: {test_dataNs[dataC]}')

        inner_cv_test_pathway_accuracy = []
        inner_cv_val_pathway_accuracy = []

        for pi in pathway_indices:
            # Train a pathway model
            pathways_sub = pathways.iloc[:, pi]

            # Weights for the first layer
            diag_self = np.zeros((sum(pathways_sub > 0), sum(pathways_sub > 0)), int)
            np.fill_diagonal(diag_self, 1)
            diag_self = diag_self * 0.5

            # Subset pathway genes
            pathway_gene_indices = np.where(pathways_sub > 0)[0]
            X_train_sub = x_train_step1[:, pathway_gene_indices]
            X_val_sub = x_val_step1[:, pathway_gene_indices]
            X_test_sub = x_test_step1[:, pathway_gene_indices]

            model1_0 = make_pathway_model(
                X_train_sub.shape[1], 1, Nnodes, Nlayers, optimizer, pi,
                pathways_sub[pathways_sub > 0], diag_self
            )

            model1_0.fit(
                X_train_sub, y_train_step1, epochs=epoch_p, batch_size=batch_size_p,
                verbose=0, callbacks=[EarlyStoppingAtMinLoss(patience=patience)],
                validation_split=0.2, shuffle=True
            )

            # Evaluate the model
            loss, accuracy = model1_0.evaluate(X_val_sub, y_val_step1, verbose=0)
            print(f'########## {pi}th pathway in {val_dataNs[dataC]} val accuracy: {accuracy*100:.2f}%')
            inner_cv_val_pathway_accuracy.append(accuracy)

            loss, accuracy = model1_0.evaluate(X_test_sub, y_test_step1, verbose=0)
            print(f'########## {pi}th pathway in {test_dataNs[dataC]} test accuracy: {accuracy*100:.2f}%')
            inner_cv_test_pathway_accuracy.append(accuracy)

            # Save predictions
            train_pred_file = runN + "/prediction_by_fold1/pi" + str(pi) + "_" + train_dataNs[dataC].replace(".pkl", "_prediction_by_fold1_pi" + str(pi) + ".pkl")
            val_pred_file = runN + "/prediction_by_fold1/pi" + str(pi) + "_" + val_dataNs[dataC].replace(".pkl", "_prediction_by_fold1_pi" + str(pi) + ".pkl")
            test_pred_file = runN + "/prediction_by_fold1/pi" + str(pi) + "_" + test_dataNs[dataC].replace(".pkl", "_prediction_by_fold1_pi" + str(pi) + ".pkl")
            
            pickle_data(train_pred_file, model1_0.predict(X_train_sub))
            pickle_data(val_pred_file, model1_0.predict(X_val_sub))
            pickle_data(test_pred_file, model1_0.predict(X_test_sub))
            
            print(f'  Saved predictions for pathway {pi} to: {test_pred_file}')

    print("Step 2 completed successfully!")
    print(f"Predictions saved to: {runN}/prediction_by_fold1/")


def main():
    """Entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser()

    # Read params
    parser.add_argument('--train_dataNs', type=str, nargs='*')
    parser.add_argument('--val_dataNs', type=str, nargs='*')
    parser.add_argument('--test_dataNs', type=str, nargs='*')
    parser.add_argument('--pathwayN', type=str)
    parser.add_argument('--Nlayers', default=[2, 3, 4, 5], type=int, nargs='*')
    parser.add_argument('--Nnodes', default=[32, 64, 128, 256, 512, 1024, 2048], type=int, nargs='*')
    parser.add_argument('--optimizer', default=['adam', 'sgd', 'rmsprop'], type=str, nargs='*')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--path_index_fileN', type=str)
    parser.add_argument('--runN', type=str)

    args = parser.parse_args()

    # Extract single values from lists
    Nlayers = args.Nlayers[0] if isinstance(args.Nlayers, list) else args.Nlayers
    Nnodes = args.Nnodes[0] if isinstance(args.Nnodes, list) else args.Nnodes
    optimizer = args.optimizer[0] if isinstance(args.optimizer, list) else args.optimizer

    step2_prediction_level1(
        train_dataNs=args.train_dataNs,
        val_dataNs=args.val_dataNs,
        test_dataNs=args.test_dataNs,
        pathwayN=args.pathwayN,
        Nlayers=Nlayers,
        Nnodes=Nnodes,
        optimizer=optimizer,
        epoch_p=args.epoch,
        patience=args.patience,
        batch_size_p=args.batch_size,
        path_index_fileN=args.path_index_fileN,
        output_prefix=args.output_prefix,
        runN=args.runN
    )


if __name__ == "__main__":
    main()
