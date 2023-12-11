import numpy as np

from assignment_3.layers.base import Layer


class ReshapeLayer(Layer):
    def __init__(self):
        self.shape: np.ndarray | None = None
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None
        self.d_inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.shape = inputs.shape
        self.outputs = self.inputs.reshape(inputs.shape[0], -1)
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        self.d_inputs = d_values.reshape(self.shape)
        return self.d_inputs
