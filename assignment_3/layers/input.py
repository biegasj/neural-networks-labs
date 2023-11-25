import numpy as np


class InputLayer:
    def __init__(self):
        self.outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs = inputs
        return self.outputs
