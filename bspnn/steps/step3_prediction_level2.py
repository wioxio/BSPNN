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
import shap
#from shap import shap_tabular
from tensorflow.keras.models import load_model
import keras
from datetime import datetime
import argparse
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle
import csv
import random
import shap
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score
import os
from ..models import make_level2_model, make_original_model
from ..callbacks import EarlyStoppingAtMinLoss
from ..utils import split_comma_separated, pickle_data


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# read params
parser = argparse.ArgumentParser()


parser.add_argument('--cv_train_dataNs', # pickled, balanced assay
                    type=str, nargs='*',)
parser.add_argument('--cv_val_dataNs', # pickled, balanced assay
                    type=str, nargs='*',)
parser.add_argument('--cv_test_dataNs', # pickled, balanced assay
                    type=str, nargs='*',)
parser.add_argument('--cv_train_pathway_prediction_dataNs', # list of pickled pathway prediction by fold1
                    type=str, nargs='*',)
parser.add_argument('--cv_val_pathway_prediction_dataNs', # list of pickled pathway prediction by fold1
                    type=str, nargs='*',)
parser.add_argument('--cv_test_pathway_prediction_dataNs', # list of pickled pathway prediction by fold1
                    type=str, nargs='*',)
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
parser.add_argument('--trial',
                        type=int)

args = parser.parse_args()


cv_train_dataNs = split_comma_separated(args.cv_train_dataNs) if args.cv_train_dataNs else args.cv_train_dataNs
cv_val_dataNs = split_comma_separated(args.cv_val_dataNs) if args.cv_val_dataNs else args.cv_val_dataNs
cv_test_dataNs = split_comma_separated(args.cv_test_dataNs) if args.cv_test_dataNs else args.cv_test_dataNs

cv_train_pathway_prediction_dataNs = split_comma_separated(args.cv_train_pathway_prediction_dataNs) if args.cv_train_pathway_prediction_dataNs else args.cv_train_pathway_prediction_dataNs
cv_val_pathway_prediction_dataNs = split_comma_separated(args.cv_val_pathway_prediction_dataNs) if args.cv_val_pathway_prediction_dataNs else args.cv_val_pathway_prediction_dataNs
cv_test_pathway_prediction_dataNs = split_comma_separated(args.cv_test_pathway_prediction_dataNs) if args.cv_test_pathway_prediction_dataNs else args.cv_test_pathway_prediction_dataNs
cv_train_pathway_prediction_dataNs = [s.replace('.pkl', '') for s in cv_train_pathway_prediction_dataNs]
cv_val_pathway_prediction_dataNs = [s.replace('.pkl', '') for s in cv_val_pathway_prediction_dataNs]
cv_test_pathway_prediction_dataNs = [s.replace('.pkl', '') for s in cv_test_pathway_prediction_dataNs]

pathwayN = args.pathwayN
Nlayers = args.Nlayers
Nnodes = args.Nnodes
optimizer = args.optimizer
epoch_p = args.epoch
output_prefix = args.output_prefix 
patience = args.patience
batch_size_p = args.batch_size
path_index_fileN=args.path_index_fileN
runN = args.runN
trial = args.trial


Nlayers = Nlayers[0]
Nnodes = Nnodes[0]
optimizer = optimizer[0]


def stepwise_forward(target_pathways_fixed_p, target_pathways_testing_p, train_dataN, test_dataN, y_train_p, y_test_p, Nnodes_p, Nlayers_p, optimizer_p, runN, epoch_p, batch_size_p, patience):
    accuracies = []
    sensitivities = []
    specificities = []
    f1s = []
    kappas = []
    # initialize accuracies
    print(f'########## stepwise forward prediction level2: {target_pathways_fixed_p}th pathway in {test_dataN}')
    model2_0 = make_level2_model(len(target_pathways_fixed_p), 1, Nnodes_p, Nlayers_p, optimizer_p)
    #
    train_pathway_predictions_by_step1 = []
    test_pathway_predictions_by_step1 = []
    #
    for pi in target_pathways_fixed_p:
        with open( runN + 
                "/prediction_level1/pi"+str(pi)+"_" + train_dataN+"_pi"+str(pi)+".pkl", 'rb') as file:
            train_pathway_predictions_by_step1.append(pickle.load(file))
        with open(runN + 
                "/prediction_level1/pi"+str(pi)+"_" + test_dataN+"_pi"+str(pi)+".pkl", 'rb') as file:
            test_pathway_predictions_by_step1.append(pickle.load(file))
    #
    train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
    train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:,:,0].T
    #
    test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
    test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:,:,0].T
    #
    model2_0.fit(train_pathway_predictions_by_step1, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
    #
    loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, y_test_p, verbose=0)
    y_pred = model2_0.predict(test_pathway_predictions_by_step1)
    y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions# Convert probabilities to binary predictions
    cm = confusion_matrix(y_test_p, y_pred)
    sensitivity = recall_score(y_test_p, y_pred, average='binary')  # 'binary' if binary classification, 'macro' for multiclass
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
    f1 = f1_score(y_test_p, y_pred, average='binary')  # 'binary' for binary, 'macro' for multiclass
    kappa = cohen_kappa_score(y_test_p, y_pred)
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    f1s.append(f1)
    kappas.append(kappa)
    # for each pathway in the testing set
    for test_pi in range(0, len(target_pathways_testing_p)):
        model2_0 = make_level2_model(len(target_pathways_fixed_p)+1, 1, Nnodes_p, Nlayers_p, optimizer_p)
        #
        train_pathway_predictions_by_step1 = []
        test_pathway_predictions_by_step1 = []
        #
        for pi in target_pathways_fixed_p + [target_pathways_testing_p[test_pi]]:
            with open( runN + 
                    "/prediction_level1/pi"+str(pi)+"_" + train_dataN+"_pi"+str(pi)+".pkl", 'rb') as file:
                train_pathway_predictions_by_step1.append(pickle.load(file))
            with open(runN + 
                    "/prediction_level1/pi"+str(pi)+"_" + test_dataN+"_pi"+str(pi)+".pkl", 'rb') as file:
                test_pathway_predictions_by_step1.append(pickle.load(file))
        #
        train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
        train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:,:,0].T
        #
        test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
        test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:,:,0].T
        #
        model2_0.fit(train_pathway_predictions_by_step1, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
        #
        loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, y_test_p, verbose=0)
        y_pred = model2_0.predict(test_pathway_predictions_by_step1)
        y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
        cm = confusion_matrix(y_test_p, y_pred)
        sensitivities.append(recall_score(y_test_p, y_pred, average='binary'))  # 'binary' if binary classification, 'macro' for multiclass
        specificities.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))  # TN / (TN + FP)
        f1s.append(f1_score(y_test_p, y_pred, average='binary'))  # 'binary' for binary, 'macro' for multiclass
        kappas.append(cohen_kappa_score(y_test_p, y_pred))
        accuracies.append(accuracy)
        choosen_index=accuracies.index(max(accuracies))
    if(choosen_index==0):
        return None, None, None, None, None, None, None, None, None, None, None
    else:
        choosen_index=choosen_index-1
        return target_pathways_testing_p[choosen_index], accuracies[choosen_index], accuracies[0], sensitivities[choosen_index], sensitivities[0], specificities[choosen_index], specificities[0], f1s[choosen_index], f1s[0], kappas[choosen_index], kappas[0]


def stepwise_forward_v2(target_pathways_fixed_p, target_pathways_testing_p, train_dataN, test_dataN, y_train_p, y_test_p, Nnodes_p, Nlayers_p, optimizer_p, runN, epoch_p, batch_size_p, patience):
    accuracies = []
    sensitivities = []
    specificities = []
    f1s = []
    kappas = []
    # for each pathway in the testing set
    for test_pi in range(0, len(target_pathways_testing_p)):
        model2_0 = make_level2_model(len(target_pathways_fixed_p)+1, 1, Nnodes_p, Nlayers_p, optimizer_p)
        #
        train_pathway_predictions_by_step1 = []
        test_pathway_predictions_by_step1 = []
        #
        for pi in target_pathways_fixed_p + [target_pathways_testing_p[test_pi]]:
            with open( runN + 
                    "/prediction_level1/pi"+str(pi)+"_" + train_dataN+"_pi"+str(pi)+".pkl", 'rb') as file:
                train_pathway_predictions_by_step1.append(pickle.load(file))
            with open(runN + 
                    "/prediction_level1/pi"+str(pi)+"_" + test_dataN+"_pi"+str(pi)+".pkl", 'rb') as file:
                test_pathway_predictions_by_step1.append(pickle.load(file))
        #
        train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
        train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:,:,0].T
        #
        test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
        test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:,:,0].T
        #
        model2_0.fit(train_pathway_predictions_by_step1, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
        #
        loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, y_test_p, verbose=0)
        y_pred = model2_0.predict(test_pathway_predictions_by_step1)
        y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
        cm = confusion_matrix(y_test_p, y_pred)
        sensitivities.append(recall_score(y_test_p, y_pred, average='binary'))  # 'binary' if binary classification, 'macro' for multiclass
        specificities.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))  # TN / (TN + FP)
        f1s.append(f1_score(y_test_p, y_pred, average='binary'))  # 'binary' for binary, 'macro' for multiclass
        kappas.append(cohen_kappa_score(y_test_p, y_pred))
        accuracies.append(accuracy)
        choosen_index=accuracies.index(max(accuracies))
    return target_pathways_testing_p[choosen_index], accuracies[choosen_index], sensitivities[choosen_index], specificities[choosen_index], f1s[choosen_index], kappas[choosen_index]
    

def csv_filename(base_name):
    return base_name.replace('.csv', f'_trial{trial}.csv')


print("Arguments:")

# print each element of cv data as a list
print("CV train data: "+str(cv_train_dataNs))
print("CV test data: "+str(cv_test_dataNs))
print("CV val data: "+str(cv_val_dataNs))
print("CV train pathway prediction data: "+str(cv_train_pathway_prediction_dataNs))
print("CV test pathway prediction data: "+str(cv_test_pathway_prediction_dataNs))
print("CV val pathway prediction data: "+str(cv_val_pathway_prediction_dataNs))
print("Pathway: "+pathwayN)
print("Path index file: "+path_index_fileN)
print("Run: "+runN)
print(f"Nlayers: {Nlayers}")
print(f"Nnodes: {Nnodes}")
print("Optimizer: "+optimizer)
print(f"epoch_p: {epoch_p}")
print(f"patience: {patience}")
print(f"batch_size: {batch_size_p}")
print("Output prefix: "+output_prefix)
print('\n')


# read pathways index
path_index=pd.read_csv(path_index_fileN, header=None, index_col=False, sep=",") 
target_pathways = path_index[0].tolist()
target_pathways=target_pathways[0:20]

pathways = pd.read_csv(pathwayN, header=0, index_col=0) #747
path_names = pathways.columns


gene_name=pathways.index.tolist()


########## original accuracies
# run model for each data and calculate the average accuracy from the vanilla model
with open(csv_filename(runN+"/original_accuracy_per_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Accuracy"]])

with open(csv_filename(runN+"/original_sensitivity_per_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Sensitivity"]])

with open(csv_filename(runN+"/original_specificity_per_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Specificity"]])

with open(csv_filename(runN+"/original_f1_per_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "F1"]])

with open(csv_filename(runN+"/original_kappa_per_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Kappa"]])

with open(csv_filename(runN+"/gene_importance_per_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Feature", "SHAP_importance"]])


original_accuracy = []
original_sensitivity = []
original_specificity = []
original_f1 = []
original_kappa = []
for dataC in range(len(cv_train_dataNs)):
    with open(cv_test_dataNs[dataC], 'rb') as file:
        test_data = pickle.load(file)
        test_x = test_data.iloc[:, 1:].values
        #substitute missing value with 0
        test_x = np.nan_to_num(test_x, 0)
        test_y = test_data.iloc[:, 0].values.astype("float32")
    print(f'test data size: ({test_x.shape[0]}, {test_x.shape[1]})')
    with open(cv_train_dataNs[dataC], 'rb') as file:
        train_data = pickle.load(file)
        train_x = train_data.iloc[:, 1:].values
        #substitute missing value with 0
        train_x = np.nan_to_num(train_x, 0)
        train_y = train_data.iloc[:, 0].values.astype("float32")
    print(f'train data size: ({train_x.shape[0]}, {train_x.shape[1]})')
    #
    # Train the model
    model0 = make_original_model(train_x.shape[1], 1, Nnodes, Nlayers, optimizer)
    model0.fit(train_x, train_y, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
    loss, accuracy = model0.evaluate(test_x, test_y, verbose=0)
    original_accuracy.append(accuracy)
    y_pred = model0.predict(test_x)
    y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
    cm = confusion_matrix(test_y, y_pred)
    original_sensitivity.append(recall_score(test_y, y_pred, average='binary'))  # 'binary' if binary classification, 'macro' for multiclass
    original_specificity.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))  # TN / (TN + FP)
    original_f1.append(f1_score(test_y, y_pred, average='binary'))  # 'binary' for binary, 'macro' for multiclass
    original_kappa.append(cohen_kappa_score(test_y, y_pred))
    print(f'########## Original Accuracy in the {dataC}th data: {accuracy*100:.2f}%')
    with open(csv_filename(runN+"/original_accuracy_per_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[dataC, accuracy]])
    with open(csv_filename(runN+"/original_sensitivity_per_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[dataC, recall_score(test_y, y_pred, average='binary')]])
    with open(csv_filename(runN+"/original_specificity_per_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[dataC, cm[0, 0] / (cm[0, 0] + cm[0, 1])]])
    with open(csv_filename(runN+"/original_f1_per_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[dataC, f1_score(test_y, y_pred, average='binary')]])
    with open(csv_filename(runN+"/original_kappa_per_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[dataC, cohen_kappa_score(test_y, y_pred)]])
    # shap
    #
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(train_x.shape[0], size=min(100, train_x.shape[0]), replace=False)
    background = train_x[bg_idx]
    #
    explainer = shap.DeepExplainer(model0, background) #set 1
    #
    shap_values_pathway = explainer.shap_values(test_x, check_additivity=False) #set 0
    #
    shap_values_pathway = np.nanmean(np.abs(shap_values_pathway), axis=0)
    # remove brackets from np.mean(np.abs(shap_values_pathway), axis=0) when writing to the file
    shap_values_pathway = np.asarray(shap_values_pathway, dtype=float).ravel()
    shap_importance = pd.DataFrame(list(zip([dataC]*test_x.shape[1], gene_name, shap_values_pathway)),
                                columns=['CV', 'Feature', 'SHAP_importance'])
    shap_importance = shap_importance.sort_values(by="SHAP_importance", ascending=False)
    # top 100 genes
    shap_importance = shap_importance.head(100)
    # write to the file, do not write column names
    shap_importance.to_csv(csv_filename(runN+"/gene_importance_per_cv_v2.csv"), index=False, mode='a', header=False)

with open(csv_filename(runN+"/original_accuracy_average_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1",  "Kappa"]])
    writer.writerows([[sum(original_accuracy)/len(original_accuracy), sum(original_sensitivity)/len(original_sensitivity), sum(original_specificity)/len(original_specificity), sum(original_f1)/len(original_f1), sum(original_kappa)/len(original_kappa)]])

del test_x, test_y, train_x, train_y, original_accuracy, original_sensitivity, original_specificity, original_f1, original_kappa


########## stepforward
# run model for each data and calculate the average accuracy from the stepwise forward model
with open(csv_filename(runN+"/stepfoward_accuracy_stepwise_history_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Pathway_index", "Accuracy"]])

with open(csv_filename(runN+"/stepfoward_sensitivity_stepwise_history_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Pathway_index", "Sensitivity"]])

with open(csv_filename(runN+"/stepfoward_specificity_stepwise_history_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Pathway_index", "Specificity"]])

with open(csv_filename(runN+"/stepfoward_f1_stepwise_history_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Pathway_index", "F1"]])

with open(csv_filename(runN+"/stepfoward_kappa_stepwise_history_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Pathway_index", "Kappa"]])


stepforward_sensitivity_ave = []
stepforward_specificity_ave = []
stepforward_f1_ave = []
stepforward_kappa_ave = []
stepforward_accuracy_ave = []
# dictionary of pathway index and cvi:
# empty dictionary
selected_pathways = {}
for cvi in range(len(cv_train_dataNs)):
    print(f'########## stepforward prediction level2: {cvi}th data')
    with open(cv_train_dataNs[cvi], 'rb') as file:
        train_data = pickle.load(file)
        #train_x = train_data.iloc[:, 1:].values
        train_y = train_data.iloc[:, 0].values.astype("float32")
        print(f'train data size: ({train_data.shape[0]}, {train_data.shape[1]})')
    with open(cv_val_dataNs[cvi], 'rb') as file:
        val_data = pickle.load(file)
        #val_x = val_data.iloc[:, 1:].values
        val_y = val_data.iloc[:, 0].values.astype("float32")
        print(f'val data size: ({val_data.shape[0]}, {val_data.shape[1]})')
    with open(cv_test_dataNs[cvi], 'rb') as file:
        test_data = pickle.load(file)
        #test_x = test_data.iloc[:, 1:].values
        test_y = test_data.iloc[:, 0].values.astype("float32")  
        print(f'test data size: ({test_data.shape[0]}, {test_data.shape[1]})')
    #
    past_acc = []
    fixed_pi=[]
    testing_pi = list(target_pathways)
    testing_pi_original = list(target_pathways)
    stepforward_sensitivity_cv = []
    stepforward_specificity_cv = []
    stepforward_f1_cv = []
    stepforward_kappa_cv = []
    stepforward_accuracy_cv = []
    last_best_accuracy = 0
    failed_count = 0
    last_max_index = -1
    tolerance = 0
    for tpi, current_pi in enumerate(testing_pi_original):
        print(f'########## stepforward prediction level2: {cvi}th data, {current_pi}th pathway, fixed_pi: {fixed_pi}')
        max_index, max_acc, sensitivity_max, specificity_max, f1_max, kappa_max = stepwise_forward_v2(target_pathways_fixed_p=fixed_pi, target_pathways_testing_p=testing_pi, 
        train_dataN=cv_train_pathway_prediction_dataNs[cvi], test_dataN=cv_val_pathway_prediction_dataNs[cvi], y_train_p=train_y, y_test_p=val_y, Nnodes_p=Nnodes, Nlayers_p=Nlayers, optimizer_p=optimizer, runN=runN, epoch_p=1, batch_size_p=batch_size_p, patience=patience)
        if(tpi >= 1 and tolerance == 2): # pass first iteration
            if(max_acc < last_best_accuracy and failed_count == 0 ):
                failed_count = failed_count + 1
            elif (max_acc < last_best_accuracy and failed_count == 1):
                break
            else:
                last_best_accuracy = max_acc
                failed_count = 0
        if tolerance == 0:
            if(max_acc < last_best_accuracy):
                if(tpi == 0):
                   last_max_index = max_index
                   past_acc.append(max_acc)
                   fixed_pi.append(max_index)
                   testing_pi.remove(max_index)
                   if(max_index not in selected_pathways):
                        selected_pathways[max_index] = 1
                   else:
                        selected_pathways[max_index] = selected_pathways[max_index] + 1
                   stepforward_accuracy_cv.append(max_acc)
                   stepforward_sensitivity_cv.append(sensitivity_max)
                   stepforward_specificity_cv.append(specificity_max)
                   stepforward_f1_cv.append(f1_max)
                   stepforward_kappa_cv.append(kappa_max)
                   break
                else:
                   break
            else:
                last_best_accuracy = max_acc
        last_max_index = max_index
        past_acc.append(max_acc)
        fixed_pi.append(max_index)
        testing_pi.remove(max_index)
        if(max_index not in selected_pathways):
            selected_pathways[max_index] = 1
        else:
            selected_pathways[max_index] = selected_pathways[max_index] + 1
        stepforward_accuracy_cv.append(max_acc)
        stepforward_sensitivity_cv.append(sensitivity_max)
        stepforward_specificity_cv.append(specificity_max)
        stepforward_f1_cv.append(f1_max)
        stepforward_kappa_cv.append(kappa_max)
    #
    if tolerance == 2:
        if(len(past_acc) > 0 and ((failed_count == 1 and max_acc < last_best_accuracy) or (failed_count == 1 and tpi == len(testing_pi_original)-1))): # if failed twice or last pathway was semi failed, drop last element
            # drop last element from past_acc, fixed_pi, stepforward_accuracy_cv, stepforward_sensitivity_cv, stepforward_specificity_cv, stepforward_f1_cv, stepforward_kappa_cv
            past_acc.pop()
            fixed_pi.pop()
            stepforward_accuracy_cv.pop()
            stepforward_sensitivity_cv.pop()
            stepforward_specificity_cv.pop()
            stepforward_f1_cv.pop()
            stepforward_kappa_cv.pop()
            selected_pathways[last_max_index] = 0
        if(len(fixed_pi) > 0 and len(testing_pi_original) > 1 and fixed_pi[-1] == testing_pi_original[1] and failed_count == 1): # it went upto the second pathway but failed, drop the second pathway
            # drop last element from past_acc, fixed_pi, stepforward_accuracy_cv, stepforward_sensitivity_cv, stepforward_specificity_cv, stepforward_f1_cv, stepforward_kappa_cv
            past_acc.pop()
            fixed_pi.pop()
            stepforward_accuracy_cv.pop()
            stepforward_sensitivity_cv.pop()
            stepforward_specificity_cv.pop()
            stepforward_f1_cv.pop()
            stepforward_kappa_cv.pop()
            selected_pathways[testing_pi_original[1]] = 0
    if(len(stepforward_sensitivity_cv) > 0):
        stepforward_sensitivity_ave.append(stepforward_sensitivity_cv[len(stepforward_sensitivity_cv)-1])
        stepforward_specificity_ave.append(stepforward_specificity_cv[len(stepforward_specificity_cv)-1])
        stepforward_f1_ave.append(stepforward_f1_cv[len(stepforward_f1_cv)-1])
        stepforward_kappa_ave.append(stepforward_kappa_cv[len(stepforward_kappa_cv)-1])
        stepforward_accuracy_ave.append(stepforward_accuracy_cv[len(stepforward_accuracy_cv)-1])
    # write the accuracy history append to the file
    with open(csv_filename(runN+"/stepfoward_accuracy_stepwise_history_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, fixed_pi[i],past_acc[i]] for i in range(len(past_acc))])
    with open(csv_filename(runN+"/stepfoward_sensitivity_stepwise_history_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, fixed_pi[i], stepforward_sensitivity_cv[i]] for i in range(len(stepforward_sensitivity_cv))])
    with open(csv_filename(runN+"/stepfoward_specificity_stepwise_history_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, fixed_pi[i], stepforward_specificity_cv[i]] for i in range(len(stepforward_specificity_cv))])
    with open(csv_filename(runN+"/stepfoward_f1_stepwise_history_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, fixed_pi[i], stepforward_f1_cv[i]] for i in range(len(stepforward_f1_cv))])
    with open(csv_filename(runN+"/stepfoward_kappa_stepwise_history_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, fixed_pi[i], stepforward_kappa_cv[i]] for i in range(len(stepforward_kappa_cv))])

with open(csv_filename(runN+"/stepfoward_cv_accuracy_average_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1",  "Kappa"]])
    writer.writerows([[sum(stepforward_accuracy_ave)/len(stepforward_accuracy_ave), sum(stepforward_sensitivity_ave)/len(stepforward_sensitivity_ave), sum(stepforward_specificity_ave)/len(stepforward_specificity_ave), sum(stepforward_f1_ave)/len(stepforward_f1_ave), sum(stepforward_kappa_ave)/len(stepforward_kappa_ave)]])


# from selected_pathways, sort by value and get the pathway with value larger than 1, and get the pathway index, 
pathway_index_selected_multi = [key for key, value in selected_pathways.items() if value > 0]
pathway_index_selected_multi


# stepforward shap cv
pd.DataFrame([['CV', 'Feature', 'SHAP_importance']]).to_csv(csv_filename(runN+"/pathway_importance_v2.csv"), index=False, header=False)

with open(csv_filename(runN+"/stepfoward_final_accuracy_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Accuracy"]])

with open(csv_filename(runN+"/stepfoward_final_sensitivity_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Sensitivity"]])

with open(csv_filename(runN+"/stepfoward_final_specificity_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Specificity"]])

with open(csv_filename(runN+"/stepfoward_final_f1_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "F1"]])

with open(csv_filename(runN+"/stepfoward_final_kappa_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Kappa"]])

final_accuracy = []
final_sensitivity = []
final_specificity = []
final_f1 = []
final_kappa = []
for cvi in range(len(cv_train_dataNs)):
    print(f'########## stepforward prediction level2: {cvi}th data')
    with open(cv_train_dataNs[cvi], 'rb') as file:
        train_data = pickle.load(file)
        #train_x = train_data.iloc[:, 1:].values
        train_y = train_data.iloc[:, 0].values.astype("float32")
        print(f'train data size: ({train_data.shape[0]}, {train_data.shape[1]})')
        del train_data
    with open(cv_val_dataNs[cvi], 'rb') as file:
        val_data = pickle.load(file)
        #val_x = val_data.iloc[:, 1:].values
        val_y = val_data.iloc[:, 0].values.astype("float32")
        print(f'val data size: ({val_data.shape[0]}, {val_data.shape[1]})')
        del val_data
    with open(cv_test_dataNs[cvi], 'rb') as file:
        test_data = pickle.load(file)
        #test_x = test_data.iloc[:, 1:].values
        test_y = test_data.iloc[:, 0].values.astype("float32")  
        print(f'test data size: ({test_data.shape[0]}, {test_data.shape[1]})')
        del test_data
    #
    train_pathway_predictions_by_step1 = []
    test_pathway_predictions_by_step1 = []
    #
    for pi in pathway_index_selected_multi:
        with open( runN + 
                "/prediction_level1/pi"+str(pi)+"_" + cv_train_pathway_prediction_dataNs[cvi]+"_pi"+str(pi)+".pkl", 'rb') as file:
            train_pathway_predictions_by_step1.append(pickle.load(file))
        with open(runN + 
                "/prediction_level1/pi"+str(pi)+"_" + cv_test_pathway_prediction_dataNs[cvi]+"_pi"+str(pi)+".pkl", 'rb') as file:
            test_pathway_predictions_by_step1.append(pickle.load(file))
    #
    train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
    train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:,:,0].T
    #
    test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
    test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:,:,0].T
    #
    model2_0 = make_level2_model(len(pathway_index_selected_multi), 1, Nnodes, Nlayers, optimizer)
    model2_0.fit(train_pathway_predictions_by_step1, train_y, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
    #
    loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, test_y, verbose=0)
    final_accuracy.append(accuracy)
    y_pred = model2_0.predict(test_pathway_predictions_by_step1)
    y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
    cm = confusion_matrix(test_y, y_pred)
    sensitivity = recall_score(test_y, y_pred, average='binary')  # 'binary' if binary classification, 'macro' for multiclass
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
    f1 = f1_score(test_y, y_pred, average='binary')  # 'binary' for binary, 'macro' for multiclass
    kappa = cohen_kappa_score(test_y, y_pred)
    final_sensitivity.append(sensitivity)
    final_specificity.append(specificity)
    final_f1.append(f1)
    final_kappa.append(kappa)
    with open(csv_filename(runN+"/stepfoward_final_accuracy_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, accuracy]])
    with open(csv_filename(runN+"/stepfoward_final_sensitivity_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, sensitivity]])
    with open(csv_filename(runN+"/stepfoward_final_specificity_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, specificity]])
    with open(csv_filename(runN+"/stepfoward_final_f1_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, f1]])
    with open(csv_filename(runN+"/stepfoward_final_kappa_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, kappa]])
    # shap
    #
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(train_pathway_predictions_by_step1.shape[0], size=min(100, train_pathway_predictions_by_step1.shape[0]), replace=False)
    background = train_pathway_predictions_by_step1[bg_idx]
    #
    explainer = shap.DeepExplainer(model2_0, background) #set 1
    #
    shap_values_pathway = explainer.shap_values(test_pathway_predictions_by_step1) #set 0
    #
    shap_values_pathway = np.array(shap_values_pathway)
    # remove brackets from np.mean(np.abs(shap_values_pathway), axis=0) when writing to the file
    shap_values_pathway = np.nanmean(np.abs(shap_values_pathway), axis=0)
    shap_values_pathway = np.asarray(shap_values_pathway, dtype=float).ravel()
    shap_importance = pd.DataFrame(list(zip([cvi]*len(pathway_index_selected_multi), path_names[pathway_index_selected_multi], shap_values_pathway)),
                                columns=['CV', 'Feature', 'SHAP_importance'])
    shap_importance = shap_importance.sort_values(by="SHAP_importance", ascending=False)
    # write to the file, do not write column names
    shap_importance.to_csv(csv_filename(runN+"/pathway_importance_v2.csv"), index=False, mode='a', header=False)

with open(csv_filename(runN+"/stepfoward_final_cv_accuracy_average_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1",  "Kappa"]])
    writer.writerows([[sum(final_accuracy)/len(final_accuracy), sum(final_sensitivity)/len(final_sensitivity), sum(final_specificity)/len(final_specificity), sum(final_f1)/len(final_f1), sum(final_kappa)/len(final_kappa)]])


# stepforward + original model with shap
pd.DataFrame([['CV', 'Feature', 'SHAP_importance']]).to_csv(csv_filename(runN+"/pathway_gene_importance_v2.csv"), index=False, header=False)

with open(csv_filename(runN+"/stepfoward_original_accuracy_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Accuracy"]])

with open(csv_filename(runN+"/stepfoward_original_sensitivity_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Sensitivity"]])

with open(csv_filename(runN+"/stepfoward_original_specificity_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Specificity"]])

with open(csv_filename(runN+"/stepfoward_original_f1_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "F1"]])

with open(csv_filename(runN+"/stepfoward_original_kappa_cv_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Kappa"]])

with open(csv_filename(runN+"/pathway_gene_importance_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["CV", "Feature", "SHAP_importance"]])

final_accuracy = []
final_sensitivity = []
final_specificity = []
final_f1 = []
final_kappa = []
for cvi in range(len(cv_train_dataNs)):
    print(f'########## stepforward prediction level2: {cvi}th data')
    with open(cv_train_dataNs[cvi], 'rb') as file:
        train_data = pickle.load(file)
        train_x = train_data.iloc[:, 1:].values
        #substitute missing value with 0
        train_x = np.nan_to_num(train_x, 0)
        train_y = train_data.iloc[:, 0].values.astype("float32")
        print(f'train data size: ({train_data.shape[0]}, {train_data.shape[1]})')
        del train_data
    with open(cv_val_dataNs[cvi], 'rb') as file:
        val_data = pickle.load(file)
        val_x = val_data.iloc[:, 1:].values
        #substitute missing value with 0
        val_x = np.nan_to_num(val_x, 0)
        val_y = val_data.iloc[:, 0].values.astype("float32")
        print(f'val data size: ({val_data.shape[0]}, {val_data.shape[1]})')
        del val_data
    with open(cv_test_dataNs[cvi], 'rb') as file:
        test_data = pickle.load(file)
        test_x = test_data.iloc[:, 1:].values
        #substitute missing value with 0
        test_x = np.nan_to_num(test_x, 0)
        test_y = test_data.iloc[:, 0].values.astype("float32")  
        print(f'test data size: ({test_data.shape[0]}, {test_data.shape[1]})')
        del test_data
    #
    train_pathway_predictions_by_step1 = []
    test_pathway_predictions_by_step1 = []
    #
    for pi in pathway_index_selected_multi:
        with open( runN + 
                "/prediction_level1/pi"+str(pi)+"_" + cv_train_pathway_prediction_dataNs[cvi]+"_pi"+str(pi)+".pkl", 'rb') as file:
            train_pathway_predictions_by_step1.append(pickle.load(file))
        with open(runN + 
                "/prediction_level1/pi"+str(pi)+"_" + cv_test_pathway_prediction_dataNs[cvi]+"_pi"+str(pi)+".pkl", 'rb') as file:
            test_pathway_predictions_by_step1.append(pickle.load(file))
    #
    train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
    train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:,:,0].T
    #
    test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
    test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:,:,0].T
    #
    train_combined_x=np.hstack((train_pathway_predictions_by_step1, train_x)) #this is on 1
    test_combined_x=np.hstack((test_pathway_predictions_by_step1, test_x)) # this is on 0
    #
    del test_pathway_predictions_by_step1, train_pathway_predictions_by_step1, train_x, test_x
    #
    model2_0 = make_level2_model(train_combined_x.shape[1], 1, Nnodes, Nlayers, optimizer)
    model2_0.fit(train_combined_x, train_y, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split = 0.2, shuffle=True)
    #
    loss, accuracy = model2_0.evaluate(test_combined_x, test_y, verbose=0)
    final_accuracy.append(accuracy)
    y_pred = model2_0.predict(test_combined_x)
    y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions
    cm = confusion_matrix(test_y, y_pred)
    sensitivity = recall_score(test_y, y_pred, average='binary')  # 'binary' if binary classification, 'macro' for multiclass
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
    f1 = f1_score(test_y, y_pred, average='binary')  # 'binary' for binary, 'macro' for multiclass
    kappa = cohen_kappa_score(test_y, y_pred)
    final_sensitivity.append(sensitivity)
    final_specificity.append(specificity)
    final_f1.append(f1)
    final_kappa.append(kappa)
    with open(csv_filename(runN+"/stepfoward_original_accuracy_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, accuracy]])
    with open(csv_filename(runN+"/stepfoward_original_sensitivity_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, sensitivity]])
    with open(csv_filename(runN+"/stepfoward_original_specificity_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, specificity]])
    with open(csv_filename(runN+"/stepfoward_original_f1_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, f1]])
    with open(csv_filename(runN+"/stepfoward_original_kappa_cv_v2.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([[cvi, kappa]])
    # shap
    #
    #check missing vlaue in train_combined_x and test_combined_x
    print(f'train_combined_x shape: {train_combined_x.shape}')
    print(f'test_combined_x shape: {test_combined_x.shape}')
    print(f'train_combined_x missing value: {np.isnan(train_combined_x).sum()}')
    print(f'test_combined_x missing value: {np.isnan(test_combined_x).sum()}')
    #
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(train_combined_x.shape[0], size=min(100, train_combined_x.shape[0]), replace=False)
    background = train_combined_x[bg_idx]
    #
    explainer = shap.DeepExplainer(model2_0, background) #set 1
    shap_values_pathway = explainer.shap_values(test_combined_x, check_additivity=False) #set 0
    #
    shap_values_pathway = np.array(shap_values_pathway)
    # remove brackets from np.mean(np.abs(shap_values_pathway), axis=0) when writing to the file
    shap_values_pathway = np.nanmean(np.abs(shap_values_pathway), axis=0)
    shap_values_pathway = np.asarray(shap_values_pathway, dtype=float).ravel()
    shap_importance = pd.DataFrame(list(zip([cvi]*(len(pathway_index_selected_multi)+len(gene_name)), path_names[pathway_index_selected_multi].tolist() + gene_name, shap_values_pathway)),
                                columns=['CV', 'Feature', 'SHAP_importance'])
    shap_importance = shap_importance.sort_values(by="SHAP_importance", ascending=False)
    # write to the file, do not write column names
    shap_importance.to_csv(csv_filename(runN+"/pathway_gene_importance_v2.csv"), index=False, mode='a', header=False)

with open(csv_filename(runN+"/stepfoward_original_final_cv_accuracy_average_v2.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1",  "Kappa"]])
    writer.writerows([[sum(final_accuracy)/len(final_accuracy), sum(final_sensitivity)/len(final_sensitivity), sum(final_specificity)/len(final_specificity), sum(final_f1)/len(final_f1), sum(final_kappa)/len(final_kappa)]])


