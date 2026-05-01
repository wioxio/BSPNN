import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
#from keras.utils import np_utils
from keras import utils   
from sklearn.utils import shuffle
import numpy as np
from datetime import datetime
import os
import pickle
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras
from datetime import datetime
import argparse
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle
import csv
import random
from ..models import make_pathway_model
from ..callbacks import EarlyStoppingAtMinLoss
from ..utils import split_comma_separated


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


parser = argparse.ArgumentParser()

# read params
parser.add_argument('--train_dataN',
                    type=str,
                    help='Train pickle(s): basename(s) resolved under RUN/data/ (comma-separated ok).')
parser.add_argument('--val_dataN',
                    type=str,
                    help='Validation pickle(s): basename(s) under RUN/data/, same order as --train_dataN.')
parser.add_argument('--pathwayN',
                        type=str)
parser.add_argument('--Nlayers', default=[2,3,4,5], 
                            type=int,
                            nargs='*',
                            help='number of nodes in the last layer')
parser.add_argument('--Nnodes', default=[32,64,128,256,512,1024,2048], 
                            type=int,
                            nargs='*',
                            help='number of nodes in the last layer')
parser.add_argument('--optimizer', default=['adam', 'sgd', 'rmsprop'], 
                        type=str,
                        nargs='*')
parser.add_argument('--epoch',
                        type=int)
parser.add_argument('--patience', default = 5,
                        type=int)
parser.add_argument('--batch_size', default = 10,
                        type=int)
parser.add_argument('--output_prefix',
                        type=str)
parser.add_argument('--pathway_start_i',
                        type=int)
parser.add_argument('--pathway_end_i',
                        type=int)
parser.add_argument('--runN',
                        type=str)

args = parser.parse_args()

runN = args.runN


def _under_run_data(runN, paths):
    if paths is None:
        return None
    return [os.path.normpath(os.path.join(runN, 'data', os.path.basename(p))) for p in paths]


train_dataN = split_comma_separated(args.train_dataN) if args.train_dataN else None
val_dataN = split_comma_separated(args.val_dataN) if args.val_dataN else None
train_dataN = _under_run_data(runN, train_dataN)
val_dataN = _under_run_data(runN, val_dataN)

if train_dataN is None:
    parser.error("--train_dataN is required.")
if val_dataN is None:
    parser.error(
        f"--val_dataN is required (one pickle per train fold under {runN}/data/, same order as --train_dataN)."
    )
if len(train_dataN) != len(val_dataN):
    parser.error(
        f"--train_dataN and --val_dataN must list the same number of files "
        f"(got {len(train_dataN)} train vs {len(val_dataN)} val)."
    )

print(train_dataN)
print(val_dataN)

pathwayN = args.pathwayN
Nlayers = args.Nlayers
Nnodes = args.Nnodes
optimizer = args.optimizer
epoch_p = args.epoch
output_prefix = args.output_prefix 
patience = args.patience
batch_size_p = args.batch_size
pathway_start_i = args.pathway_start_i
pathway_end_i = args.pathway_end_i


############### This should be updated
Nlayers = Nlayers[0]
Nnodes = Nnodes[0]
optimizer = optimizer[0]


print("Arguments:")
print("Pathway: "+pathwayN)
print(f"Nlayers: {Nlayers}")
print(f"Nnodes: {Nnodes}")
print("Optimizer: "+optimizer)
print(f"epoch_p: {epoch_p}")
print(f"patience: {patience}")
print('\n')


# make a directory for the run and data layout
os.makedirs(runN, exist_ok=True)
os.makedirs(os.path.join(runN, 'data'), exist_ok=True)

# read pathways
pathways = pd.read_csv(pathwayN, header=0, index_col=0)

# number of pathways
npath = pathways.shape[1]
print(f"Number of pathways: {npath}")

# pathway names for lime
pathway_names = ['_' + element + '_' for element in pathways.columns.tolist()]
# Create a mapping of substrings in B to their indices
substring_to_index = {substring: index for index, substring in enumerate(pathway_names)}

# gene names for lime
gene_names = ['_' + str(element) + '_' for element in pathways.index.tolist()]
gene_substring_to_index = {substring: index for index, substring in enumerate(gene_names)}

for dataC in range(len(train_dataN)):
    #
    outter_cv_pathway_accuracy_writerN = open(runN + '/' + output_prefix + '_primary_cv' + str(dataC) + '.csv', 'a', newline='')
    outter_cv_pathway_accuracy_writer = csv.writer(outter_cv_pathway_accuracy_writerN)
    #
    outter_cv_pathway_accuracy_sorted_writerN = open(runN + '/' + output_prefix + '_primary_sorted_cv' + str(dataC) + '.csv', 'a', newline='')
    outter_cv_pathway_accuracy_sorted_writer = csv.writer(outter_cv_pathway_accuracy_sorted_writerN)
    #
    with open(train_dataN[dataC], 'rb') as file:
        train_data = pickle.load(file)
    #
    x_train = train_data.iloc[:, 1:].values
    # substitute missing value with 0
    x_train = np.nan_to_num(x_train, 0)
    y_train = train_data.iloc[:, 0].values
    y_train = y_train.astype("float32")
    #
    with open(val_dataN[dataC], 'rb') as file:
        val_data = pickle.load(file)
    #
    x_val = val_data.iloc[:, 1:].values
    # substitute missing value with 0
    x_val = np.nan_to_num(x_val, 0)
    y_val = val_data.iloc[:, 0].values
    y_val = y_val.astype("float32")    
    #
    pathway_accuracies = []
    #
    for pi in range(pathway_start_i, pathway_end_i+1):      
        #
        ########## Train a pathway model
        pathways_sub = pathways.iloc[:,pi]
        #
        # weights for the first layer
        diag_self = np.zeros((sum(pathways_sub>0), sum(pathways_sub>0)), int)  
        np.fill_diagonal(diag_self, 1) 
        diag_self = diag_self * 0.5
        #
        # subset pathway genes
        X_train_sub=x_train[:,np.where(pathways_sub>0)[0]]
        X_val_sub=x_val[:,np.where(pathways_sub>0)[0]]
        #
        model1_0 = make_pathway_model(X_train_sub.shape[1], 1, Nnodes, Nlayers, optimizer, pi, pathways_sub[pathways_sub>0], diag_self)
        #
        model1_0.fit(X_train_sub, y_train, epochs=epoch_p, batch_size=batch_size_p, verbose=0, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
        #
        ########## Evaluate the model
        loss, accuracy = model1_0.evaluate(X_val_sub, y_val, verbose=0)
        print(f'########## pathway {pi} accuracy: {accuracy*100:.2f}%')
        pathway_accuracies.append(accuracy)
    #
    # Write pathway accuracies with pathway index
    outter_cv_pathway_accuracy_writer.writerows(
        [list(item) for item in zip(range(pathway_start_i, pathway_end_i+1), pathway_accuracies)]
    )
    outter_cv_pathway_accuracy_writerN.close()
    #
    # Write pathway accuracies sorted by accuracy
    sorted_indices = sorted(
        zip(range(pathway_start_i, pathway_end_i+1), pathway_accuracies),
        key=lambda x: x[1],
        reverse=True
    )
    outter_cv_pathway_accuracy_sorted_writer.writerows([list(item) for item in sorted_indices])
    outter_cv_pathway_accuracy_sorted_writerN.close()
