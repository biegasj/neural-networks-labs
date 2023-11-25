from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    def __init__(self):
        self.d_inputs: np.ndarray | None = None

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        outputs = self.forward(y_pred, y_true)
        return np.mean(outputs)

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
        pass

    @abstractmethod
    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
        pass


class MeanAbsoluteError(LossFunction):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.abs(y_true - y_pred)

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        num_of_samples = len(d_values)
        num_of_outputs_in_sample = len(d_values[0])

        self.d_inputs = np.sign(y_true - d_values) / num_of_outputs_in_sample
        self.d_inputs = self.d_inputs / num_of_samples
        return self.d_inputs


class MeanSquaredError(LossFunction):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
        return (y_true - y_pred) ** 2

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
        num_of_samples = len(d_values)
        num_of_outputs_in_sample = len(d_values[0])

        self.d_inputs = -2 * (y_true - d_values) / num_of_outputs_in_sample
        self.d_inputs = self.d_inputs / num_of_samples
        return self.d_inputs


class CategoricalCrossEntropyLoss(LossFunction):
    epsilon: float = 1e-7

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
        num_of_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(num_of_samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
        num_of_samples = len(d_values)
        if len(y_true.shape) == 1:
            num_of_outputs_in_sample = len(d_values[0])
            y_true = np.eye(num_of_outputs_in_sample)[y_true]
        self.d_inputs = (-y_true / d_values) / num_of_samples
        return self.d_inputs
