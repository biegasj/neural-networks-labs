import numpy as np

from assignment_3.layers.base import Layer


class InputLayer(Layer):
    def __init__(self):
        self.outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs = inputs
        return self.outputs
