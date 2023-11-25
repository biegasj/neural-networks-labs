import numpy as np


def classification_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if len(y_pred.shape) == 2:
        return np.mean(np.argmax(y_pred, axis=1) == y_true)
    return np.mean(y_pred == y_true)


def regression_accuracy(
    y_pred: np.ndarray, y_true: np.ndarray, precision: float
) -> float:
    return np.mean(np.abs(y_pred - y_true) < precision)
