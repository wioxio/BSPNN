"""
Step 3: Level 2 prediction with stepwise forward selection.

Trains level 2 models using pathway predictions, performs stepwise forward selection,
and evaluates with SHAP importance analysis.
"""

import pandas as pd
import numpy as np
import os
import pickle
import csv
import warnings
import shap
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score

from ..models import make_level2_model, make_original_model
from ..callbacks import EarlyStoppingAtMinLoss
from ..utils import normalize_data, split_comma_separated, pickle_data, configure_gpu


# Configure GPU (TensorFlow 2.x style - no InteractiveSession needed)
configure_gpu()

# Suppress SHAP warnings about tensor structure (harmless warning in newer TensorFlow versions)
warnings.filterwarnings('ignore', message='.*structure of `inputs`.*', category=UserWarning)


def stepwise_forward(
    target_pathways_fixed_p,
    target_pathways_testing_p,
    train_dataN,
    test_dataN,
    y_train_p,
    y_test_p,
    Nnodes_p,
    Nlayers_p,
    optimizer_p,
    runN,
    epoch_p,
    batch_size_p,
    patience
):
    """
    Perform stepwise forward selection to find the best pathway combination.
    
    Args:
        target_pathways_fixed_p: List of fixed pathway indices
        target_pathways_testing_p: List of pathway indices to test
        train_dataN: Training data file name (without .pkl)
        test_dataN: Test data file name (without .pkl)
        y_train_p: Training labels
        y_test_p: Test labels
        Nnodes_p: Number of nodes per layer
        Nlayers_p: Number of layers
        optimizer_p: Optimizer name
        runN: Output directory
        epoch_p: Number of epochs
        batch_size_p: Batch size
        patience: Early stopping patience
        
    Returns:
        Tuple of (max_index, max_acc, original_acc, sensitivity_max, original_sensitivity,
                 specificity_max, original_specificity, f1_max, original_f1,
                 kappa_max, original_kappa) or None values if no improvement
    """
    accuracies = []
    sensitivities = []
    specificities = []
    f1s = []
    kappas = []
    
    print(f'########## stepwise forward prediction level2: {target_pathways_fixed_p}th pathway in {test_dataN}')
    model2_0 = make_level2_model(len(target_pathways_fixed_p), 1, Nnodes_p, Nlayers_p, optimizer_p)
    
    train_pathway_predictions_by_step1 = []
    test_pathway_predictions_by_step1 = []
    
    for pi in target_pathways_fixed_p:
        with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + train_dataN + "_pi" + str(pi) + ".pkl", 'rb') as file:
            train_pathway_predictions_by_step1.append(pickle.load(file))
        with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + test_dataN + "_pi" + str(pi) + ".pkl", 'rb') as file:
            test_pathway_predictions_by_step1.append(pickle.load(file))
    
    train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
    train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:, :, 0].T
    
    test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
    test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:, :, 0].T
    
    model2_0.fit(train_pathway_predictions_by_step1, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split=0.2, shuffle=True)
    
    loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, y_test_p, verbose=0)
    y_pred = model2_0.predict(test_pathway_predictions_by_step1)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test_p, y_pred)
    sensitivity = recall_score(y_test_p, y_pred, average='binary')
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_test_p, y_pred, average='binary')
    kappa = cohen_kappa_score(y_test_p, y_pred)
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    f1s.append(f1)
    kappas.append(kappa)
    
    # For each pathway in the testing set
    for test_pi in range(0, len(target_pathways_testing_p)):
        model2_0 = make_level2_model(len(target_pathways_fixed_p) + 1, 1, Nnodes_p, Nlayers_p, optimizer_p)
        
        train_pathway_predictions_by_step1 = []
        test_pathway_predictions_by_step1 = []
        
        for pi in target_pathways_fixed_p + [target_pathways_testing_p[test_pi]]:
            with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + train_dataN + "_pi" + str(pi) + ".pkl", 'rb') as file:
                train_pathway_predictions_by_step1.append(pickle.load(file))
            with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + test_dataN + "_pi" + str(pi) + ".pkl", 'rb') as file:
                test_pathway_predictions_by_step1.append(pickle.load(file))
        
        train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
        train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:, :, 0].T
        
        test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
        test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:, :, 0].T
        
        model2_0.fit(train_pathway_predictions_by_step1, y_train_p, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split=0.2, shuffle=True)
        
        loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, y_test_p, verbose=0)
        y_pred = model2_0.predict(test_pathway_predictions_by_step1)
        y_pred = (y_pred > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_test_p, y_pred)
        sensitivities.append(recall_score(y_test_p, y_pred, average='binary'))
        specificities.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
        f1s.append(f1_score(y_test_p, y_pred, average='binary'))
        kappas.append(cohen_kappa_score(y_test_p, y_pred))
        accuracies.append(accuracy)
        choosen_index = accuracies.index(max(accuracies))
    
    if choosen_index == 0:
        return None, None, None, None, None, None, None, None, None, None, None
    else:
        choosen_index = choosen_index - 1
        return (target_pathways_testing_p[choosen_index], accuracies[choosen_index], accuracies[0],
                sensitivities[choosen_index], sensitivities[0], specificities[choosen_index], specificities[0],
                f1s[choosen_index], f1s[0], kappas[choosen_index], kappas[0])


def step3_prediction_level2(
    cv_train_dataNs,
    cv_val_dataNs,
    cv_test_dataNs,
    cv_train_pathway_prediction_dataNs,
    cv_val_pathway_prediction_dataNs,
    cv_test_pathway_prediction_dataNs,
    pathwayN,
    Nlayers,
    Nnodes,
    optimizer,
    epoch_p,
    patience,
    batch_size_p,
    path_index_fileN,
    output_prefix,
    runN,
    trial=None
):
    """
    Main function for step 3: Level 2 prediction with stepwise forward selection.
    
    This function performs:
    1. Original model evaluation with SHAP
    2. Stepwise forward selection
    3. Final model evaluation with selected pathways
    4. Combined pathway + gene model evaluation
    """
    # Split comma-separated strings
    cv_train_dataNs = split_comma_separated(cv_train_dataNs) if cv_train_dataNs else cv_train_dataNs
    cv_val_dataNs = split_comma_separated(cv_val_dataNs) if cv_val_dataNs else cv_val_dataNs
    cv_test_dataNs = split_comma_separated(cv_test_dataNs) if cv_test_dataNs else cv_test_dataNs

    cv_train_pathway_prediction_dataNs = split_comma_separated(cv_train_pathway_prediction_dataNs) if cv_train_pathway_prediction_dataNs else cv_train_pathway_prediction_dataNs
    cv_val_pathway_prediction_dataNs = split_comma_separated(cv_val_pathway_prediction_dataNs) if cv_val_pathway_prediction_dataNs else cv_val_pathway_prediction_dataNs
    cv_test_pathway_prediction_dataNs = split_comma_separated(cv_test_pathway_prediction_dataNs) if cv_test_pathway_prediction_dataNs else cv_test_pathway_prediction_dataNs

    cv_train_pathway_prediction_dataNs = [s.replace('.pkl', '') for s in cv_train_pathway_prediction_dataNs]
    cv_val_pathway_prediction_dataNs = [s.replace('.pkl', '') for s in cv_val_pathway_prediction_dataNs]
    cv_test_pathway_prediction_dataNs = [s.replace('.pkl', '') for s in cv_test_pathway_prediction_dataNs]

    print("Arguments:")
    print("CV train data: " + str(cv_train_dataNs))
    print("CV test data: " + str(cv_test_dataNs))
    print("CV val data: " + str(cv_val_dataNs))
    print("CV train pathway prediction data: " + str(cv_train_pathway_prediction_dataNs))
    print("CV test pathway prediction data: " + str(cv_test_pathway_prediction_dataNs))
    print("CV val pathway prediction data: " + str(cv_val_pathway_prediction_dataNs))
    print("Pathway: " + pathwayN)
    print("Path index file: " + path_index_fileN)
    print("Run: " + runN)
    print(f"Nlayers: {Nlayers}")
    print(f"Nnodes: {Nnodes}")
    print("Optimizer: " + optimizer)
    print(f"epoch_p: {epoch_p}")
    print(f"patience: {patience}")
    print(f"batch_size: {batch_size_p}")
    print("Output prefix: " + output_prefix)
    print('\n')

    # Helper function to add trial suffix to CSV filenames
    def csv_filename(base_name):
        if trial is not None:
            return base_name.replace('.csv', f'_trial{trial}.csv')
        return base_name

    # Read pathways index
    # header yes index no
    path_index = pd.read_csv(path_index_fileN, header=0, index_col=False, sep=",")
    # read the first column
    target_pathways = path_index.iloc[:, 0].tolist()
    target_pathways = target_pathways[0:20]

    pathways = pd.read_csv(pathwayN, header=0, index_col=0)
    path_names = pathways.columns
    gene_name = pathways.index.tolist()

    ########## original accuracies
    # Run model for each data and calculate the average accuracy from the vanilla model
    with open(csv_filename(runN + "/original_accuracy_per_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Accuracy"]])

    with open(csv_filename(runN + "/original_sensitivity_per_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Sensitivity"]])

    with open(csv_filename(runN + "/original_specificity_per_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Specificity"]])

    with open(csv_filename(runN + "/original_f1_per_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "F1"]])

    with open(csv_filename(runN + "/original_kappa_per_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Kappa"]])

    with open(csv_filename(runN + "/gene_importance_per_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Feature", "SHAP_importance"]])

    original_accuracy = []
    original_sensitivity = []
    original_specificity = []
    original_f1 = []
    original_kappa = []
    
    # Validate that all data lists have the same length
    if len(cv_train_dataNs) != len(cv_test_dataNs):
        raise ValueError(f"Mismatch in data list lengths: cv_train_dataNs has {len(cv_train_dataNs)} items, but cv_test_dataNs has {len(cv_test_dataNs)} items")
    
    for dataC in range(len(cv_train_dataNs)):
        # Construct file paths
        test_file_path = os.path.join(runN, 'data', cv_test_dataNs[dataC])
        train_file_path = os.path.join(runN, 'data', cv_train_dataNs[dataC])
        
        # Check if files exist before opening
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test data file not found: {test_file_path}\n"
                                  f"Looking for file: {cv_test_dataNs[dataC]}\n"
                                  f"Data directory: {os.path.join(runN, 'data')}\n"
                                  f"Directory exists: {os.path.exists(os.path.join(runN, 'data'))}\n"
                                  f"If directory exists, files in it: {os.listdir(os.path.join(runN, 'data')) if os.path.exists(os.path.join(runN, 'data')) else 'N/A'}")
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Train data file not found: {train_file_path}\n"
                                  f"Looking for file: {cv_train_dataNs[dataC]}")
        
        with open(test_file_path, 'rb') as file:
            test_data = pickle.load(file)
            test_x = test_data.iloc[:, 1:].values
            test_x = normalize_data(test_x)
            test_y = test_data.iloc[:, 0].values.astype("float32")
        print(f'test data size: ({test_x.shape[0]}, {test_x.shape[1]})')
        with open(train_file_path, 'rb') as file:
            train_data = pickle.load(file)
            train_x = train_data.iloc[:, 1:].values
            train_x = normalize_data(train_x)
            train_y = train_data.iloc[:, 0].values.astype("float32")
        print(f'train data size: ({train_x.shape[0]}, {train_x.shape[1]})')
        
        # Train the model
        model0 = make_original_model(train_x.shape[1], 1, Nnodes, Nlayers, optimizer)
        model0.fit(train_x, train_y, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split=0.2, shuffle=True)
        loss, accuracy = model0.evaluate(test_x, test_y, verbose=0)
        original_accuracy.append(accuracy)
        y_pred = model0.predict(test_x)
        y_pred = (y_pred > 0.5).astype(int).flatten()
        cm = confusion_matrix(test_y, y_pred)
        original_sensitivity.append(recall_score(test_y, y_pred, average='binary'))
        original_specificity.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
        original_f1.append(f1_score(test_y, y_pred, average='binary'))
        original_kappa.append(cohen_kappa_score(test_y, y_pred))
        print(f'########## Original Accuracy in the {dataC}th data: {accuracy*100:.2f}%')
        with open(csv_filename(runN + "/original_accuracy_per_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[dataC, accuracy]])
        with open(csv_filename(runN + "/original_sensitivity_per_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[dataC, recall_score(test_y, y_pred, average='binary')]])
        with open(csv_filename(runN + "/original_specificity_per_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[dataC, cm[0, 0] / (cm[0, 0] + cm[0, 1])]])
        with open(csv_filename(runN + "/original_f1_per_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[dataC, f1_score(test_y, y_pred, average='binary')]])
        with open(csv_filename(runN + "/original_kappa_per_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[dataC, cohen_kappa_score(test_y, y_pred)]])
        
        # SHAP analysis
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(train_x.shape[0], size=min(100, train_x.shape[0]), replace=False)
        background = np.asarray(train_x[bg_idx], dtype=np.float32)
        
        explainer = shap.DeepExplainer(model0, background)
        shap_values_pathway = explainer.shap_values(np.asarray(test_x, dtype=np.float32), check_additivity=False)
        shap_values_pathway = np.nanmean(np.abs(shap_values_pathway), axis=0)
        shap_values_pathway = np.asarray(shap_values_pathway, dtype=float).ravel()
        shap_importance = pd.DataFrame(list(zip([dataC] * test_x.shape[1], gene_name, shap_values_pathway)),
                                    columns=['CV', 'Feature', 'SHAP_importance'])
        shap_importance = shap_importance.sort_values(by="SHAP_importance", ascending=False)
        # Top 100 genes
        shap_importance = shap_importance.head(100)
        shap_importance.to_csv(csv_filename(runN + "/gene_importance_per_cv.csv"), index=False, mode='a', header=False)

    with open(csv_filename(runN + "/original_accuracy_average.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1", "Kappa"]])
        writer.writerows([[
            sum(original_accuracy) / len(original_accuracy) if original_accuracy else 0,
            sum(original_sensitivity) / len(original_sensitivity) if original_sensitivity else 0,
            sum(original_specificity) / len(original_specificity) if original_specificity else 0,
            sum(original_f1) / len(original_f1) if original_f1 else 0,
            sum(original_kappa) / len(original_kappa) if original_kappa else 0
        ]])

    del test_x, test_y, train_x, train_y, original_accuracy, original_sensitivity, original_specificity, original_f1, original_kappa

    ########## stepforward
    # Run model for each data and calculate the average accuracy from the stepwise forward model
    with open(csv_filename(runN + "/stepfoward_accuracy_stepwise_history.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Pathway_index", "Accuracy"]])

    with open(csv_filename(runN + "/stepfoward_sensitivity_stepwise_history.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Pathway_index", "Sensitivity"]])

    with open(csv_filename(runN + "/stepfoward_specificity_stepwise_history.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Pathway_index", "Specificity"]])

    with open(csv_filename(runN + "/stepfoward_f1_stepwise_history.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Pathway_index", "F1"]])

    with open(csv_filename(runN + "/stepfoward_kappa_stepwise_history.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Pathway_index", "Kappa"]])

    stepforward_sensitivity_ave = []
    stepforward_specificity_ave = []
    stepforward_f1_ave = []
    stepforward_kappa_ave = []
    stepforward_accuracy_ave = []
    # Dictionary of pathway index and cvi:
    selected_pathways = {target_pathways[0]: 2}
    
    for cvi in range(len(cv_train_dataNs)):
        print(f'########## stepforward prediction level2: {cvi}th data')
        train_file_path = os.path.join(runN, 'data', cv_train_dataNs[cvi])
        val_file_path = os.path.join(runN, 'data', cv_val_dataNs[cvi])
        test_file_path = os.path.join(runN, 'data', cv_test_dataNs[cvi])
        
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Train data file not found: {train_file_path}")
        if not os.path.exists(val_file_path):
            raise FileNotFoundError(f"Validation data file not found: {val_file_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test data file not found: {test_file_path}")
        
        with open(train_file_path, 'rb') as file:
            train_data = pickle.load(file)
            train_y = train_data.iloc[:, 0].values.astype("float32")
            print(f'train data size: ({train_data.shape[0]}, {train_data.shape[1]})')
            del train_data
        with open(val_file_path, 'rb') as file:
            val_data = pickle.load(file)
            val_y = val_data.iloc[:, 0].values.astype("float32")
            print(f'val data size: ({val_data.shape[0]}, {val_data.shape[1]})')
            del val_data
        with open(test_file_path, 'rb') as file:
            test_data = pickle.load(file)
            test_y = test_data.iloc[:, 0].values.astype("float32")
            print(f'test data size: ({test_data.shape[0]}, {test_data.shape[1]})')
            del test_data
        
        past_acc = [0]
        fixed_pi = [target_pathways[0]]
        testing_pi = target_pathways[1:]
        stepforward_sensitivity_cv = []
        stepforward_specificity_cv = []
        stepforward_f1_cv = []
        stepforward_kappa_cv = []
        stepforward_accuracy_cv = []
        
        for tpi in testing_pi:
            print(f'########## stepforward prediction level2: {cvi}th data, {tpi}th pathway, fixed_pi: {fixed_pi}')
            max_index, max_acc, original_acc, sensitivity_max, original_sensitivity, specificity_max, original_specificity, f1_max, original_f1, kappa_max, original_kappa = stepwise_forward(
                target_pathways_fixed_p=fixed_pi,
                target_pathways_testing_p=testing_pi,
                train_dataN=cv_train_pathway_prediction_dataNs[cvi],
                test_dataN=cv_val_pathway_prediction_dataNs[cvi],
                y_train_p=train_y,
                y_test_p=val_y,
                Nnodes_p=Nnodes,
                Nlayers_p=Nlayers,
                optimizer_p=optimizer,
                runN=runN,
                epoch_p=1,
                batch_size_p=batch_size_p,
                patience=patience
            )
            if max_index == None:
                break
            if max_acc < past_acc[-1]:
                break
            past_acc[0] = original_acc
            past_acc.append(max_acc)
            fixed_pi.append(max_index)
            testing_pi.remove(max_index)
            if max_index not in selected_pathways:
                selected_pathways[max_index] = 1
            else:
                selected_pathways[max_index] = selected_pathways[max_index] + 1
            stepforward_accuracy_cv.append(max_acc)
            stepforward_sensitivity_cv.append(sensitivity_max)
            stepforward_specificity_cv.append(specificity_max)
            stepforward_f1_cv.append(f1_max)
            stepforward_kappa_cv.append(kappa_max)
        
        if len(stepforward_sensitivity_cv) > 0:
            stepforward_sensitivity_ave.append(stepforward_sensitivity_cv[len(stepforward_sensitivity_cv) - 1])
            stepforward_specificity_ave.append(stepforward_specificity_cv[len(stepforward_specificity_cv) - 1])
            stepforward_f1_ave.append(stepforward_f1_cv[len(stepforward_f1_cv) - 1])
            stepforward_kappa_ave.append(stepforward_kappa_cv[len(stepforward_kappa_cv) - 1])
            stepforward_accuracy_ave.append(stepforward_accuracy_cv[len(stepforward_accuracy_cv) - 1])
        
        # Write the accuracy history append to the file
        with open(csv_filename(runN + "/stepfoward_accuracy_stepwise_history.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, fixed_pi[i], past_acc[i]] for i in range(len(past_acc))])
        with open(csv_filename(runN + "/stepfoward_sensitivity_stepwise_history.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, fixed_pi[i], stepforward_sensitivity_cv[i]] for i in range(len(stepforward_sensitivity_cv))])
        with open(csv_filename(runN + "/stepfoward_specificity_stepwise_history.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, fixed_pi[i], stepforward_specificity_cv[i]] for i in range(len(stepforward_specificity_cv))])
        with open(csv_filename(runN + "/stepfoward_f1_stepwise_history.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, fixed_pi[i], stepforward_f1_cv[i]] for i in range(len(stepforward_f1_cv))])
        with open(csv_filename(runN + "/stepfoward_kappa_stepwise_history.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, fixed_pi[i], stepforward_kappa_cv[i]] for i in range(len(stepforward_kappa_cv))])

    with open(csv_filename(runN + "/stepfoward_cv_accuracy_average.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1", "Kappa"]])
        writer.writerows([[
            sum(stepforward_accuracy_ave) / len(stepforward_accuracy_ave) if stepforward_accuracy_ave else 0,
            sum(stepforward_sensitivity_ave) / len(stepforward_sensitivity_ave) if stepforward_sensitivity_ave else 0,
            sum(stepforward_specificity_ave) / len(stepforward_specificity_ave) if stepforward_specificity_ave else 0,
            sum(stepforward_f1_ave) / len(stepforward_f1_ave) if stepforward_f1_ave else 0,
            sum(stepforward_kappa_ave) / len(stepforward_kappa_ave) if stepforward_kappa_ave else 0
        ]])

    # Get pathways selected multiple times
    pathway_index_selected_multi = [key for key, value in selected_pathways.items() if value > 1]
    # Convert to numpy array of integers for proper indexing
    # Convert keys to integers in case they're strings or other types
    if len(pathway_index_selected_multi) > 0:
        pathway_index_selected_multi = np.array([int(key) for key in pathway_index_selected_multi], dtype=int)
    else:
        pathway_index_selected_multi = np.array([], dtype=int)

    # Stepforward SHAP CV
    pd.DataFrame([['CV', 'Feature', 'SHAP_importance']]).to_csv(csv_filename(runN + "/pathway_importance.csv"), index=False, header=False)

    with open(csv_filename(runN + "/stepfoward_final_accuracy_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Accuracy"]])

    with open(csv_filename(runN + "/stepfoward_final_sensitivity_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Sensitivity"]])

    with open(csv_filename(runN + "/stepfoward_final_specificity_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Specificity"]])

    with open(csv_filename(runN + "/stepfoward_final_f1_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "F1"]])

    with open(csv_filename(runN + "/stepfoward_final_kappa_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Kappa"]])

    final_accuracy = []
    final_sensitivity = []
    final_specificity = []
    final_f1 = []
    final_kappa = []
    
    for cvi in range(len(cv_train_dataNs)):
        print(f'########## stepforward prediction level2: {cvi}th data')
        train_file_path = os.path.join(runN, 'data', cv_train_dataNs[cvi])
        val_file_path = os.path.join(runN, 'data', cv_val_dataNs[cvi])
        test_file_path = os.path.join(runN, 'data', cv_test_dataNs[cvi])
        
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Train data file not found: {train_file_path}")
        if not os.path.exists(val_file_path):
            raise FileNotFoundError(f"Validation data file not found: {val_file_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test data file not found: {test_file_path}")
        
        with open(train_file_path, 'rb') as file:
            train_data = pickle.load(file)
            train_y = train_data.iloc[:, 0].values.astype("float32")
            print(f'train data size: ({train_data.shape[0]}, {train_data.shape[1]})')
            del train_data
        with open(val_file_path, 'rb') as file:
            val_data = pickle.load(file)
            val_y = val_data.iloc[:, 0].values.astype("float32")
            print(f'val data size: ({val_data.shape[0]}, {val_data.shape[1]})')
            del val_data
        with open(test_file_path, 'rb') as file:
            test_data = pickle.load(file)
            test_y = test_data.iloc[:, 0].values.astype("float32")
            print(f'test data size: ({test_data.shape[0]}, {test_data.shape[1]})')
            del test_data
        
        train_pathway_predictions_by_step1 = []
        test_pathway_predictions_by_step1 = []
        
        for pi in pathway_index_selected_multi:
            with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + cv_train_pathway_prediction_dataNs[cvi] + "_pi" + str(pi) + ".pkl", 'rb') as file:
                train_pathway_predictions_by_step1.append(pickle.load(file))
            with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + cv_test_pathway_prediction_dataNs[cvi] + "_pi" + str(pi) + ".pkl", 'rb') as file:
                test_pathway_predictions_by_step1.append(pickle.load(file))
        
        train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
        train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:, :, 0].T
        
        test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
        test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:, :, 0].T
        
        model2_0 = make_level2_model(len(pathway_index_selected_multi), 1, Nnodes, Nlayers, optimizer)
        model2_0.fit(train_pathway_predictions_by_step1, train_y, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split=0.2, shuffle=True)
        
        loss, accuracy = model2_0.evaluate(test_pathway_predictions_by_step1, test_y, verbose=0)
        final_accuracy.append(accuracy)
        y_pred = model2_0.predict(test_pathway_predictions_by_step1)
        y_pred = (y_pred > 0.5).astype(int).flatten()
        cm = confusion_matrix(test_y, y_pred)
        sensitivity = recall_score(test_y, y_pred, average='binary')
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        f1 = f1_score(test_y, y_pred, average='binary')
        kappa = cohen_kappa_score(test_y, y_pred)
        final_sensitivity.append(sensitivity)
        final_specificity.append(specificity)
        final_f1.append(f1)
        final_kappa.append(kappa)
        with open(csv_filename(runN + "/stepfoward_final_accuracy_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, accuracy]])
        with open(csv_filename(runN + "/stepfoward_final_sensitivity_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, sensitivity]])
        with open(csv_filename(runN + "/stepfoward_final_specificity_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, specificity]])
        with open(csv_filename(runN + "/stepfoward_final_f1_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, f1]])
        with open(csv_filename(runN + "/stepfoward_final_kappa_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, kappa]])
        
        # SHAP analysis
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(train_pathway_predictions_by_step1.shape[0], size=min(100, train_pathway_predictions_by_step1.shape[0]), replace=False)
        background = np.asarray(train_pathway_predictions_by_step1[bg_idx], dtype=np.float32)
        
        explainer = shap.DeepExplainer(model2_0, background)
        shap_values_pathway = explainer.shap_values(np.asarray(test_pathway_predictions_by_step1, dtype=np.float32))
        shap_values_pathway = np.array(shap_values_pathway)
        shap_values_pathway = np.nanmean(np.abs(shap_values_pathway), axis=0)
        shap_values_pathway = np.asarray(shap_values_pathway, dtype=float).ravel()
        shap_importance = pd.DataFrame(
            list(zip([cvi] * len(pathway_index_selected_multi), path_names[pathway_index_selected_multi], shap_values_pathway)),
            columns=['CV', 'Feature', 'SHAP_importance']
        )
        shap_importance = shap_importance.sort_values(by="SHAP_importance", ascending=False)
        shap_importance.to_csv(csv_filename(runN + "/pathway_importance.csv"), index=False, mode='a', header=False)

    with open(csv_filename(runN + "/stepfoward_final_cv_accuracy_average.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1", "Kappa"]])
        writer.writerows([[
            sum(final_accuracy) / len(final_accuracy) if final_accuracy else 0,
            sum(final_sensitivity) / len(final_sensitivity) if final_sensitivity else 0,
            sum(final_specificity) / len(final_specificity) if final_specificity else 0,
            sum(final_f1) / len(final_f1) if final_f1 else 0,
            sum(final_kappa) / len(final_kappa) if final_kappa else 0
        ]])

    # Stepforward + original model with SHAP
    pd.DataFrame([['CV', 'Feature', 'SHAP_importance']]).to_csv(csv_filename(runN + "/pathway_gene_importance.csv"), index=False, header=False)

    with open(csv_filename(runN + "/stepfoward_original_accuracy_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Accuracy"]])

    with open(csv_filename(runN + "/stepfoward_original_sensitivity_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Sensitivity"]])

    with open(csv_filename(runN + "/stepfoward_original_specificity_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Specificity"]])

    with open(csv_filename(runN + "/stepfoward_original_f1_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "F1"]])

    with open(csv_filename(runN + "/stepfoward_original_kappa_cv.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Kappa"]])

    with open(csv_filename(runN + "/pathway_gene_importance.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["CV", "Feature", "SHAP_importance"]])

    final_accuracy = []
    final_sensitivity = []
    final_specificity = []
    final_f1 = []
    final_kappa = []
    
    for cvi in range(len(cv_train_dataNs)):
        print(f'########## stepforward prediction level2: {cvi}th data')
        train_file_path = os.path.join(runN, 'data', cv_train_dataNs[cvi])
        val_file_path = os.path.join(runN, 'data', cv_val_dataNs[cvi])
        test_file_path = os.path.join(runN, 'data', cv_test_dataNs[cvi])
        
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Train data file not found: {train_file_path}")
        if not os.path.exists(val_file_path):
            raise FileNotFoundError(f"Validation data file not found: {val_file_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test data file not found: {test_file_path}")
        
        with open(train_file_path, 'rb') as file:
            train_data = pickle.load(file)
            train_x = train_data.iloc[:, 1:].values
            train_x = normalize_data(train_x)
            train_y = train_data.iloc[:, 0].values.astype("float32")
            print(f'train data size: ({train_data.shape[0]}, {train_data.shape[1]})')
            del train_data
        with open(val_file_path, 'rb') as file:
            val_data = pickle.load(file)
            val_x = val_data.iloc[:, 1:].values
            val_x = normalize_data(val_x)
            val_y = val_data.iloc[:, 0].values.astype("float32")
            print(f'val data size: ({val_data.shape[0]}, {val_data.shape[1]})')
            del val_data
        with open(test_file_path, 'rb') as file:
            test_data = pickle.load(file)
            test_x = test_data.iloc[:, 1:].values
            test_x = normalize_data(test_x)
            test_y = test_data.iloc[:, 0].values.astype("float32")
            print(f'test data size: ({test_data.shape[0]}, {test_data.shape[1]})')
            del test_data
        
        train_pathway_predictions_by_step1 = []
        test_pathway_predictions_by_step1 = []
        
        for pi in pathway_index_selected_multi:
            with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + cv_train_pathway_prediction_dataNs[cvi] + "_pi" + str(pi) + ".pkl", 'rb') as file:
                train_pathway_predictions_by_step1.append(pickle.load(file))
            with open(runN + "/prediction_by_fold1/pi" + str(pi) + "_" + cv_test_pathway_prediction_dataNs[cvi] + "_pi" + str(pi) + ".pkl", 'rb') as file:
                test_pathway_predictions_by_step1.append(pickle.load(file))
        
        train_pathway_predictions_by_step1 = np.array(train_pathway_predictions_by_step1)
        train_pathway_predictions_by_step1 = train_pathway_predictions_by_step1[:, :, 0].T
        
        test_pathway_predictions_by_step1 = np.array(test_pathway_predictions_by_step1)
        test_pathway_predictions_by_step1 = test_pathway_predictions_by_step1[:, :, 0].T
        
        train_combined_x = np.hstack((train_pathway_predictions_by_step1, train_x))
        test_combined_x = np.hstack((test_pathway_predictions_by_step1, test_x))
        
        del test_pathway_predictions_by_step1, train_pathway_predictions_by_step1, train_x, test_x
        
        model2_0 = make_level2_model(train_combined_x.shape[1], 1, Nnodes, Nlayers, optimizer)
        model2_0.fit(train_combined_x, train_y, epochs=epoch_p, batch_size=batch_size_p, verbose=1, callbacks=[EarlyStoppingAtMinLoss(patience=patience)], validation_split=0.2, shuffle=True)
        
        loss, accuracy = model2_0.evaluate(test_combined_x, test_y, verbose=0)
        final_accuracy.append(accuracy)
        y_pred = model2_0.predict(test_combined_x)
        y_pred = (y_pred > 0.5).astype(int).flatten()
        cm = confusion_matrix(test_y, y_pred)
        sensitivity = recall_score(test_y, y_pred, average='binary')
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        f1 = f1_score(test_y, y_pred, average='binary')
        kappa = cohen_kappa_score(test_y, y_pred)
        final_sensitivity.append(sensitivity)
        final_specificity.append(specificity)
        final_f1.append(f1)
        final_kappa.append(kappa)
        with open(csv_filename(runN + "/stepfoward_original_accuracy_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, accuracy]])
        with open(csv_filename(runN + "/stepfoward_original_sensitivity_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, sensitivity]])
        with open(csv_filename(runN + "/stepfoward_original_specificity_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, specificity]])
        with open(csv_filename(runN + "/stepfoward_original_f1_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, f1]])
        with open(csv_filename(runN + "/stepfoward_original_kappa_cv.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows([[cvi, kappa]])
        
        # SHAP analysis
        print(f'train_combined_x shape: {train_combined_x.shape}')
        print(f'test_combined_x shape: {test_combined_x.shape}')
        print(f'train_combined_x missing value: {np.isnan(train_combined_x).sum()}')
        print(f'test_combined_x missing value: {np.isnan(test_combined_x).sum()}')
        
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(train_combined_x.shape[0], size=min(100, train_combined_x.shape[0]), replace=False)
        background = np.asarray(train_combined_x[bg_idx], dtype=np.float32)
        
        explainer = shap.DeepExplainer(model2_0, background)
        shap_values_pathway = explainer.shap_values(np.asarray(test_combined_x, dtype=np.float32), check_additivity=False)
        shap_values_pathway = np.array(shap_values_pathway)
        shap_values_pathway = np.nanmean(np.abs(shap_values_pathway), axis=0)
        shap_values_pathway = np.asarray(shap_values_pathway, dtype=float).ravel()
        shap_importance = pd.DataFrame(
            list(zip([cvi] * (len(pathway_index_selected_multi) + len(gene_name)), 
                     path_names[pathway_index_selected_multi].tolist() + gene_name, shap_values_pathway)),
            columns=['CV', 'Feature', 'SHAP_importance']
        )
        shap_importance = shap_importance.sort_values(by="SHAP_importance", ascending=False)
        shap_importance.to_csv(csv_filename(runN + "/pathway_gene_importance.csv"), index=False, mode='a', header=False)

    with open(csv_filename(runN + "/stepfoward_original_final_cv_accuracy_average.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows([["Accuracy", "Sensitivity", "Specificity", "F1", "Kappa"]])
        writer.writerows([[
            sum(final_accuracy) / len(final_accuracy) if final_accuracy else 0,
            sum(final_sensitivity) / len(final_sensitivity) if final_sensitivity else 0,
            sum(final_specificity) / len(final_specificity) if final_specificity else 0,
            sum(final_f1) / len(final_f1) if final_f1 else 0,
            sum(final_kappa) / len(final_kappa) if final_kappa else 0
        ]])

    print("Step 3 completed successfully!")


def main():
    """Entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--cv_train_dataNs', type=str, nargs='*')
    parser.add_argument('--cv_val_dataNs', type=str, nargs='*')
    parser.add_argument('--cv_test_dataNs', type=str, nargs='*')
    parser.add_argument('--cv_train_pathway_prediction_dataNs', type=str, nargs='*')
    parser.add_argument('--cv_val_pathway_prediction_dataNs', type=str, nargs='*')
    parser.add_argument('--cv_test_pathway_prediction_dataNs', type=str, nargs='*')
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
    parser.add_argument('--trial', type=int)

    args = parser.parse_args()

    # Extract single values from lists
    Nlayers = args.Nlayers[0] if isinstance(args.Nlayers, list) else args.Nlayers
    Nnodes = args.Nnodes[0] if isinstance(args.Nnodes, list) else args.Nnodes
    optimizer = args.optimizer[0] if isinstance(args.optimizer, list) else args.optimizer

    step3_prediction_level2(
        cv_train_dataNs=args.cv_train_dataNs,
        cv_val_dataNs=args.cv_val_dataNs,
        cv_test_dataNs=args.cv_test_dataNs,
        cv_train_pathway_prediction_dataNs=args.cv_train_pathway_prediction_dataNs,
        cv_val_pathway_prediction_dataNs=args.cv_val_pathway_prediction_dataNs,
        cv_test_pathway_prediction_dataNs=args.cv_test_pathway_prediction_dataNs,
        pathwayN=args.pathwayN,
        Nlayers=Nlayers,
        Nnodes=Nnodes,
        optimizer=optimizer,
        epoch_p=args.epoch,
        patience=args.patience,
        batch_size_p=args.batch_size,
        path_index_fileN=args.path_index_fileN,
        output_prefix=args.output_prefix,
        runN=args.runN,
        trial=args.trial
    )


if __name__ == "__main__":
    main()
