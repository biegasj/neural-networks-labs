import numpy as np

from assignment_2.layer import InputLayer, Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer] | None = None):
        self.input_layer = InputLayer()
        self.hidden_layers: list[Layer] = []
        self._output_layer: Layer | None = None

        if layers:
            self._initialize_layers(layers)

    def _initialize_layers(self, layers: list[Layer]) -> None:
        self.hidden_layers = layers[:-1]
        self._output_layer = layers[-1]

    def add_layer(self, layer: Layer) -> None:
        self.hidden_layers.append(layer)

    @property
    def output_layer(self) -> Layer:
        return self._output_layer

    @output_layer.setter
    def output_layer(self, layer: Layer) -> None:
        self._output_layer = layer

    @property
    def all_layers(self) -> list[Layer]:
        return self.hidden_layers + [self.output_layer]

    @property
    def output_dim(self) -> int:
        return self._output_layer.num_of_neurons or 0

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_layers)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        outputs = self.input_layer.forward(inputs)
        for layer in self.hidden_layers:
            if not training and not isinstance(layer, Layer):
                continue
            outputs = layer.forward(outputs)
        return self.output_layer.forward(outputs)

    def backward(self, d_loss: np.ndarray) -> None:
        d_values = self.output_layer.backward(d_loss)
        for layer in reversed(self.hidden_layers):
            d_values = layer.backward(d_values)
