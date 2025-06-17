# utils.py content with ALS implementation

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import nnls
import warnings
import matplotlib.gridspec as gs

# Suppress the Specific RuntimeWarning from nnls if it arises due to all zeros in b
warnings.filterwarnings("ignore", message="The algorithm terminated successfully.", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Iteration limit reached.", category=RuntimeWarning)


def evaluate_model(R_true, R_pred, indices):
    y_true = np.array([R_true[tuple(idx)] for idx in indices])
    y_pred = np.array([R_pred[tuple(idx)] for idx in indices])
    y_true_bin = y_true > 0
    
    # Check if y_true_bin contains both True and False values
    if len(np.unique(y_true_bin)) == 2:
        # Check if y_pred contains non-NaN values for AUROC
        # This is primarily to handle potential NaN issues if nnls returns them for some reason
        if not np.any(np.isnan(y_pred)):
            auroc = roc_auc_score(y_true_bin, y_pred)
        else:
            auroc = np.nan # If R_pred contains NaNs, AUROC cannot be computed
    else:
        auroc = np.nan # AUROC requires at least one positive and one negative sample

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return rmse, auroc


def predicted_frequency_class(x):
    thresholds = [0.42, 1.26, 2.43, 3.25, 3.93]
    if x < thresholds[0]:
        return 0
    elif x <= thresholds[1]:
        return 1
    elif x <= thresholds[2]:
        return 2
    elif x <= thresholds[3]:
        return 3
    elif x <= thresholds[4]:
        return 4
    else:
        return 5
    
def evaluate_prediction_classes(R_true, R_pred, indices):
    y_true = []
    y_pred = []

    for idx in indices:
        true_val = int(R_true[tuple(idx)])
        # Ensure R_pred[tuple(idx)] is not NaN before passing to predicted_frequency_class
        pred_score = R_pred[tuple(idx)]
        if np.isnan(pred_score):
            # Handle NaN predictions, e.g., by assigning to class 0 or skipping
            # For this context, assuming NaN means unobserved, so we can map to 0 (zero class)
            # Or perhaps better, simply skip if we only want to evaluate known frequencies
            # The original paper's context is that '0' implies unobserved, but also a class.
            # If a prediction is NaN, it's an invalid prediction for a specific frequency class.
            # Let's skip it to avoid polluting results, or handle as an error if it occurs.
            # Given the problem, we're evaluating known frequencies (`if true_val > 0:`),
            # so if a prediction for a known frequency is NaN, it's a model failure.
            # For robustness, we could assign a default class like 0 or treat it as an error.
            # For now, let's let `predicted_frequency_class` handle values directly, 
            # as `nnls` is expected to return non-negative floats.
            pred_val = 0 # Default to 'zero' if prediction is NaN (shouldn't happen with nnls)
        else:
            pred_val = predicted_frequency_class(pred_score)

        # Only consider as a real class if R_true > 0 (as per the paper's evaluation)
        if true_val > 0:
            y_true.append(true_val)
            y_pred.append(pred_val)

    labels_true = [1, 2, 3, 4, 5]
    labels_pred = [0, 1, 2, 3, 4, 5]

    # Matriz de confusão
    # filter y_true and y_pred to only include labels present in labels_true (1-5) and labels_pred (0-5)
    # This addresses the ValueError if labels are missing
    y_true_filtered = [y for y in y_true if y in labels_true]
    y_pred_filtered = [y for y in y_pred if y in labels_pred]

    # Ensure y_true_filtered and y_pred_filtered are not empty
    if len(y_true_filtered) == 0:
        return np.zeros((len(labels_true), len(labels_pred))), "No true labels to evaluate."

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels_pred)

    report = classification_report(y_true_filtered, y_pred_filtered, labels=labels_pred, zero_division=0)

    return cm, report

    
