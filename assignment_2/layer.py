import numpy as np

from assignment_2.utils.activation_functions import ActivationFunction
from assignment_2.utils.weights import WeightInitializer


class InputLayer:
    def __init__(self):
        self.outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs = inputs
        return self.outputs


class Layer(InputLayer):
    def __init__(
        self,
        input_dim: int,
        num_of_neurons: int,
        *,
        weight_initializer: WeightInitializer,
        activation_function: ActivationFunction,
    ):
        self.input_dim = input_dim
        self.num_of_neurons = num_of_neurons
        self.weights_initializer = weight_initializer
        self.weights = self.weights_initializer.initialize(input_dim, num_of_neurons)
        self.biases = np.zeros((1, num_of_neurons))
        self.activation_function = activation_function
        # Attributes related to forward pass
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None
        # Attributes related to backward pass
        self.d_inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.outputs = self.activation_function.forward(self.outputs)
        return self.outputs

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        d_values = self.activation_function.backward(d_values)
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_values, self.weights.T)
        return self.d_inputs


class DropoutLayer:
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
