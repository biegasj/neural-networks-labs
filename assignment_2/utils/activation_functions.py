from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    def __init__(self):
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None
        self.d_inputs: np.ndarray | None = None

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, d_values: np.ndarray) -> np.ndarray:
        pass


class LinearActivation(ActivationFunction):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = inputs
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        self.d_inputs = d_values.copy()
        return self.d_inputs


class SigmoidActivation(ActivationFunction):
    # For regression tasks
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        self.d_inputs = d_values * self.outputs * (1 - self.outputs)
        return self.d_inputs


class ReLUActivation(ActivationFunction):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0
        return self.d_inputs


class SoftmaxActivation(ActivationFunction):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        powers = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = powers / np.sum(powers, axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        self.d_inputs = np.empty_like(d_values)
        for i, (output, d_value) in enumerate(zip(self.outputs, d_values)):
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            self.d_inputs[i] = np.dot(jacobian_matrix, d_value)
        return self.d_inputs
