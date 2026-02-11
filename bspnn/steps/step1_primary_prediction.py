"""
Step 1: Primary pathway prediction.

Trains pathway-specific models and evaluates their performance.
"""

import pandas as pd
import numpy as np
import os
import pickle
import csv
import tensorflow as tf

from ..models import make_pathway_model
from ..callbacks import EarlyStoppingAtMinLoss
from ..utils import normalize_data, configure_gpu


# Configure GPU (TensorFlow 2.x style - no InteractiveSession needed)
configure_gpu()


def step1_primary_prediction(
    train_dataN,
    test_dataN,
    pathwayN,
    Nlayers,
    Nnodes,
    optimizer,
    epoch_p,
    patience,
    batch_size_p,
    pathway_start_i,
    pathway_end_i,
    output_prefix,
    runN
):
    """
    Train pathway-specific models and evaluate their performance.
    
    Args:
        train_dataN: Path to training data pickle file
        test_dataN: Path to test data pickle file
        pathwayN: Path to pathway CSV file
        Nlayers: Number of layers (not used but kept for compatibility)
        Nnodes: Number of nodes (not used but kept for compatibility)
        optimizer: Optimizer name
        epoch_p: Maximum number of epochs
        patience: Patience for early stopping
        batch_size_p: Batch size
        pathway_start_i: Starting pathway index
        pathway_end_i: Ending pathway index
        output_prefix: Prefix for output files
        runN: Output directory name
    """
    print("Arguments:")
    print("Pathway: " + pathwayN)
    print(f"Nlayers: {Nlayers}")
    print(f"Nnodes: {Nnodes}")
    print("Optimizer: " + optimizer)
    print(f"epoch_p: {epoch_p}")
    print(f"patience: {patience}")
    print('\n')

    # Make a directory for the run
    os.makedirs(runN, exist_ok=True)

    # Read pathways
    pathways = pd.read_csv(pathwayN, header=0, index_col=0)

    # Number of pathways
    npath = pathways.shape[1]
    print(f"Number of pathways: {npath}")

    # Pathway names for lime
    pathway_names = ['_' + element + '_' for element in pathways.columns.tolist()]
    # Create a mapping of substrings in B to their indices
    substring_to_index = {substring: index for index, substring in enumerate(pathway_names)}

    # Gene names for lime
    gene_names = ['_' + str(element) + '_' for element in pathways.index.tolist()]
    gene_substring_to_index = {substring: index for index, substring in enumerate(gene_names)}

    outter_cv_pathway_accuracy_writerN = open(runN + '/' + output_prefix + '_primary.csv', 'a', newline='')
    outter_cv_pathway_accuracy_writer = csv.writer(outter_cv_pathway_accuracy_writerN)

    outter_cv_pathway_accuracy_sorted_writerN = open(runN + '/' + output_prefix + '_primary_sorted.csv', 'a', newline='')
    outter_cv_pathway_accuracy_sorted_writer = csv.writer(outter_cv_pathway_accuracy_sorted_writerN)

    with open(train_dataN, 'rb') as file:
        train_data = pickle.load(file)

    x_train = train_data.iloc[:, 1:].values
    x_train = normalize_data(x_train)
    y_train = train_data.iloc[:, 0].values
    y_train = y_train.astype("float32")

    with open(test_dataN, 'rb') as file:
        test_data = pickle.load(file)

    x_test = test_data.iloc[:, 1:].values
    x_test = normalize_data(x_test)
    y_test = test_data.iloc[:, 0].values
    y_test = y_test.astype("float32")

    pathway_accuracies = []

    for pi in range(pathway_start_i, pathway_end_i + 1):
        # Train a pathway model
        pathways_sub = pathways.iloc[:, pi]

        # Weights for the first layer
        diag_self = np.zeros((sum(pathways_sub > 0), sum(pathways_sub > 0)), int)
        np.fill_diagonal(diag_self, 1)
        diag_self = diag_self * 0.5

        # Subset pathway genes
        X_train_sub = x_train[:, np.where(pathways_sub > 0)[0]]
        X_test_sub = x_test[:, np.where(pathways_sub > 0)[0]]

        model1_0 = make_pathway_model(X_train_sub.shape[1], 1, Nnodes, Nlayers, optimizer, pi, pathways_sub[pathways_sub > 0], diag_self)

        model1_0.fit(X_train_sub, y_train, epochs=epoch_p, batch_size=batch_size_p, verbose=0, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split=0.2, shuffle=True)

        # Evaluate the model
        loss, accuracy = model1_0.evaluate(X_test_sub, y_test, verbose=0)
        print(f'########## pathway {pi} accuracy: {accuracy*100:.2f}%')
        pathway_accuracies.append(accuracy)

    # Write pathway accuracies with pathway index
    outter_cv_pathway_accuracy_writer.writerows(
        [list(item) for item in zip(range(pathway_start_i, pathway_end_i + 1), pathway_accuracies)]
    )
    outter_cv_pathway_accuracy_writerN.close()

    # Write pathway accuracies sorted by accuracy
    sorted_indices = sorted(
        zip(range(pathway_start_i, pathway_end_i + 1), pathway_accuracies),
        key=lambda x: x[1],
        reverse=True
    )
    outter_cv_pathway_accuracy_sorted_writer.writerows([list(item) for item in sorted_indices])
    outter_cv_pathway_accuracy_sorted_writerN.close()


def main():
    """Entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser()

    # Read params
    parser.add_argument('--train_dataN', type=str)
    parser.add_argument('--test_dataN', type=str)
    parser.add_argument('--pathwayN', type=str)
    parser.add_argument('--Nlayers', default=[2, 3, 4, 5], type=int, nargs='*',
                        help='number of nodes in the last layer')
    parser.add_argument('--Nnodes', default=[32, 64, 128, 256, 512, 1024, 2048], type=int, nargs='*',
                        help='number of nodes in the last layer')
    parser.add_argument('--optimizer', default=['adam', 'sgd', 'rmsprop'], type=str, nargs='*')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--pathway_start_i', type=int)
    parser.add_argument('--pathway_end_i', type=int)
    parser.add_argument('--runN', type=str)

    args = parser.parse_args()

    # Extract single values from lists
    Nlayers = args.Nlayers[0] if isinstance(args.Nlayers, list) else args.Nlayers
    Nnodes = args.Nnodes[0] if isinstance(args.Nnodes, list) else args.Nnodes
    optimizer = args.optimizer[0] if isinstance(args.optimizer, list) else args.optimizer

    step1_primary_prediction(
        train_dataN=args.train_dataN,
        test_dataN=args.test_dataN,
        pathwayN=args.pathwayN,
        Nlayers=Nlayers,
        Nnodes=Nnodes,
        optimizer=optimizer,
        epoch_p=args.epoch,
        patience=args.patience,
        batch_size_p=args.batch_size,
        pathway_start_i=args.pathway_start_i,
        pathway_end_i=args.pathway_end_i,
        output_prefix=args.output_prefix,
        runN=args.runN
    )


if __name__ == "__main__":
    main()
