"""
Utility helpers used by BSPNN v2 step scripts.
"""

import pickle
import numpy as np


def split_comma_separated(arg_list):
    if arg_list is None:
        return None
    if isinstance(arg_list, str):
        arg_list = [arg_list]
    result = []
    for item in arg_list:
        split_items = [s.strip() for s in str(item).split(",")]
        split_items = [s for s in split_items if s]
        result.extend(split_items)
    return result if result else None


def clean_file_list(file_list):
    if file_list is None:
        return []
    result = []
    for item in file_list:
        items = item.split(",") if isinstance(item, str) else [item]
        for fname in items:
            cleaned = fname.strip().rstrip(",")
            if cleaned:
                result.append(cleaned)
    return result


def pickle_data(fileN_p, dt_p):
    file = open(fileN_p, "wb")
    pickle.dump(dt_p, file)
    file.close()


def get_importance_index_flag(y_test_p, sampleC_for_importance_p, dataset_name=""):
    y_test_sum = sum(y_test_p)
    importance_sampleC_1 = int(sampleC_for_importance_p / 2)
    importance_sampleC_2 = int(sampleC_for_importance_p / 2)
    if y_test_sum[0] < importance_sampleC_1 and y_test_sum[1] >= importance_sampleC_2:
        print(f"Importance sample count error\nTest data in {dataset_name} has only {y_test_sum[0]} counts in the first label")
        importance_sampleC_1 = y_test_sum[0]
        importance_sampleC_2 = int(sampleC_for_importance_p / 2) + importance_sampleC_1 - y_test_sum[0]
    if y_test_sum[1] < importance_sampleC_2 and y_test_sum[0] >= importance_sampleC_1:
        print(f"Importance sample count error\nTest data in {dataset_name} has only {y_test_sum[1]} counts in the second label")
        importance_sampleC_1 = int(sampleC_for_importance_p / 2) + importance_sampleC_2 - y_test_sum[1]
        importance_sampleC_2 = y_test_sum[1]
    if y_test_sum[1] < importance_sampleC_2 and y_test_sum[0] < importance_sampleC_1:
        print(
            f"Importance sample count error\nTest data in {dataset_name} has only {y_test_sum[0]} counts in the first label and {y_test_sum[1]} counts in the second label"
        )
        importance_sampleC_1 = y_test_sum[0]
        importance_sampleC_2 = y_test_sum[1]
    condition_indices = np.where(y_test_p[:, 0] == 1)[0]
    np.random.seed(609)
    sampled_indices = np.random.choice(condition_indices, size=importance_sampleC_1, replace=False)
    condition_indices = np.where(y_test_p[:, 1] == 1)[0]
    sampled_indices = np.concatenate((np.random.choice(condition_indices, size=importance_sampleC_2, replace=False), sampled_indices), axis=0)
    return sampled_indices
