import numpy as np

from assignment_3.layers.base import Layer


class DropoutLayer(Layer):
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        self.mask = None
        # Attributes related to forward pass
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None
        # Attributes related to backward pass
        self.d_inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.mask = np.random.binomial(
            1, 1 - self.dropout_rate, size=self.inputs.shape
        ) / (1 - self.dropout_rate)
        self.outputs = self.inputs * self.mask
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        self.d_inputs = d_values * self.mask
        return self.d_inputs
