import numpy as np

from assignment_1.layer import InputLayer, Layer
from assignment_1.utils.activation_functions import ActivationFunction
from assignment_1.utils.weights import WeightInitializer


class NeuralNetwork:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_of_hidden_layers: int,
        weight_initializer: WeightInitializer,
        activation_function: ActivationFunction,
        output_activation_function: ActivationFunction
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_of_hidden_layers
        self.weight_initializer = weight_initializer
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        # Layers
        self.input_layer = InputLayer()
        self.hidden_layers = self._create_hidden_layers()
        self.output_layer = Layer(
            self.hidden_dim,
            self.output_dim,
            weight_initializer=self.weight_initializer,
            activation_function=self.output_activation_function,
        )

    def _create_hidden_layers(self) -> list[Layer]:
        hidden_layers = [
            Layer(
                self.input_dim,
                self.hidden_dim,
                weight_initializer=self.weight_initializer,
                activation_function=self.activation_function,
            )
        ]
        for _ in range(self.num_hidden_layers - 1):
            hidden_layers.append(
                Layer(
                    self.hidden_dim,
                    self.hidden_dim,
                    weight_initializer=self.weight_initializer,
                    activation_function=self.activation_function,
                )
            )
        return hidden_layers

    @property
    def all_layers(self) -> list[Layer]:
        return self.hidden_layers + [self.output_layer]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self.input_layer.forward(inputs)
        for layer in self.hidden_layers:
            outputs = layer.forward(outputs)
        return self.output_layer.forward(outputs)

    def backward(self, d_loss: np.ndarray) -> None:
        d_values = self.output_layer.backward(d_loss)
        for layer in reversed(self.hidden_layers):
            d_values = layer.backward(d_values)
