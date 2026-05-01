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
from pathlib import Path
from ..models import make_pathway_model
from ..callbacks import EarlyStoppingAtMinLoss
from ..utils import clean_file_list, pickle_data, format_pathway_pred_path_for_display


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


parser = argparse.ArgumentParser()

# read params
parser.add_argument('--train_dataNs',
                    type=str, nargs='*')
parser.add_argument('--val_dataNs',
                    type=str, nargs='*')                 
parser.add_argument('--test_dataNs',
                    type=str, nargs='*')
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
parser.add_argument('--path_index_fileN',
                        type=str)
parser.add_argument('--runN',
                        type=str)

args = parser.parse_args()


train_dataNs=args.train_dataNs
val_dataNs=args.val_dataNs
test_dataNs=args.test_dataNs

# Split comma-separated strings and clean filenames
train_dataNs = clean_file_list(train_dataNs)
val_dataNs = clean_file_list(val_dataNs)
test_dataNs = clean_file_list(test_dataNs)

pathwayN = args.pathwayN
Nlayers = args.Nlayers
Nnodes = args.Nnodes
optimizer = args.optimizer
epoch_p = args.epoch
output_prefix = args.output_prefix 
patience = args.patience
batch_size_p = args.batch_size
path_index_fileN = args.path_index_fileN
runN = args.runN


def _under_run_data(runN, names):
    return [os.path.normpath(os.path.join(runN, 'data', os.path.basename(n))) for n in names]


train_dataNs = _under_run_data(runN, train_dataNs)
val_dataNs = _under_run_data(runN, val_dataNs)
test_dataNs = _under_run_data(runN, test_dataNs)

os.makedirs(os.path.join(runN, 'data'), exist_ok=True)
os.makedirs(os.path.join(runN, 'prediction_level1'), exist_ok=True)

# Handle output_prefix if None or list
if output_prefix is None:
    output_prefix = "output"
elif isinstance(output_prefix, list):
    output_prefix = output_prefix[0] if len(output_prefix) > 0 else "output"


Nlayers = Nlayers[0]
Nnodes = Nnodes[0]
optimizer = optimizer[0]


print("Arguments:")
print("Train data: "+str(train_dataNs))
print("Val data: "+str(val_dataNs))
print("Test data: "+str(test_dataNs))
print("Pathway: "+pathwayN)
print("Path index file: "+path_index_fileN)
print("Run: "+runN)
print(f"Nlayers: {Nlayers}")
print(f"Nnodes: {Nnodes}")
print("Optimizer: "+optimizer)
print(f"epoch_p: {epoch_p}")
print(f"patience: {patience}")
print('\n')


# read pathways
pathways = pd.read_csv(pathwayN, header=0, index_col=0)
# read pathway indices, no header and index 
pathway_indices = pd.read_csv(path_index_fileN, header=None, index_col=None)
pathway_indices = pathway_indices.iloc[range(0,20), 0].values
pathway_indices = pathway_indices.astype(int)


for dataC in range(len(train_dataNs)):
    with open(train_dataNs[dataC], 'rb') as file:
        train_data_step1 = pickle.load(file)
    with open(val_dataNs[dataC], 'rb') as file:
        val_data_step1 = pickle.load(file)
    with open(test_dataNs[dataC], 'rb') as file:
        test_data_step1 = pickle.load(file)
    #
    x_train_step1 = train_data_step1.iloc[:, 1:].values
    # substitute missing value with 0
    x_train_step1 = np.nan_to_num(x_train_step1, 0)
    y_train_step1 = train_data_step1.iloc[:, 0].values
    y_train_step1 = y_train_step1.astype("float32")
    #
    x_val_step1 = val_data_step1.iloc[:, 1:].values
    # substitute missing value with 0
    x_val_step1 = np.nan_to_num(x_val_step1, 0)
    y_val_step1 = val_data_step1.iloc[:, 0].values
    y_val_step1 = y_val_step1.astype("float32")
    #
    x_test_step1 = test_data_step1.iloc[:, 1:].values
    # substitute missing value with 0
    x_test_step1 = np.nan_to_num(x_test_step1, 0)
    y_test_step1 = test_data_step1.iloc[:, 0].values
    y_test_step1 = y_test_step1.astype("float32")
    #
    print(f'########## Processing dataset {dataC+1}/{len(test_dataNs)}: {test_dataNs[dataC]}') 
    #        
    inner_cv_test_pathway_accuracy = [] # X_train based pathway accuracy for the current test fold test_index size:  pathway
    inner_cv_val_pathway_accuracy = [] # X_val based pathway accuracy for the current test fold test_index size:  pathway
    #
    for pi in pathway_indices:      
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
        pathway_gene_indices = np.where(pathways_sub>0)[0]
        X_train_sub=x_train_step1[:,pathway_gene_indices]
        X_val_sub=x_val_step1[:,pathway_gene_indices]
        X_test_sub=x_test_step1[:,pathway_gene_indices]
        #
        model1_0 = make_pathway_model(X_train_sub.shape[1], 1, Nnodes, Nlayers, optimizer, pi, pathways_sub[pathways_sub>0], diag_self)
        #
        model1_0.fit(X_train_sub, y_train_step1, epochs=epoch_p, batch_size=batch_size_p, verbose=0, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
        #
        ########## Evaluate the model
        loss, accuracy = model1_0.evaluate(X_val_sub, y_val_step1, verbose=0)
        print(f'########## {pi}th pathway in {val_dataNs[dataC]} val accuracy: {accuracy*100:.2f}%')
        inner_cv_val_pathway_accuracy.append(accuracy) # all path for the current inner cv
        loss, accuracy = model1_0.evaluate(X_test_sub, y_test_step1, verbose=0)
        print(f'########## {pi}th pathway in {test_dataNs[dataC]} test accuracy: {accuracy*100:.2f}%')
        inner_cv_test_pathway_accuracy.append(accuracy) # all path for the current inner cv
        #
        # Create directory if it doesn't exist
        Path(os.path.join(runN, 'prediction_level1')).mkdir(parents=True, exist_ok=True)
        stem_tr = os.path.splitext(os.path.basename(train_dataNs[dataC]))[0]
        stem_va = os.path.splitext(os.path.basename(val_dataNs[dataC]))[0]
        stem_te = os.path.splitext(os.path.basename(test_dataNs[dataC]))[0]
        train_pred_file = os.path.join(runN, 'prediction_level1', f'pi{pi}_{stem_tr}.pkl')
        val_pred_file = os.path.join(runN, 'prediction_level1', f'pi{pi}_{stem_va}.pkl')
        test_pred_file = os.path.join(runN, 'prediction_level1', f'pi{pi}_{stem_te}.pkl')
        pickle_data(train_pred_file, model1_0.predict(X_train_sub))
        pickle_data(val_pred_file, model1_0.predict(X_val_sub))
        pickle_data(test_pred_file, model1_0.predict(X_test_sub))
        print(
            f'  Saved pathway {pi} predictions: '
            f'{format_pathway_pred_path_for_display(train_pred_file)} | '
            f'{format_pathway_pred_path_for_display(val_pred_file)} | '
            f'{format_pathway_pred_path_for_display(test_pred_file)}'
        )


