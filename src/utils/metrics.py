from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import numpy as np 

def calc_bacc(y_true, y_pred):
    """
    Calculate balanced accuracy.
    Args:
        y_true: Ground truth, value 0 or 1
        y_pred: Predict
    """ 

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    score = balanced_accuracy_score(y_true, y_pred)

    return score


def calc_macro_f1(y_true, y_pred):
    """
    Calculate Macro-F1 score.
    Args:
        y_true: Ground truth, value 0 or 1
        y_pred: Predictions, value 0 or 1
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()

    score = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return score

