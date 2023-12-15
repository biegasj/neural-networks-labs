import numpy as np

from assignment_3.layers import Layer
from assignment_3.networks.core import NeuralNetwork


class Autoencoder(NeuralNetwork):
    @property
    def encoder(self) -> list[Layer]:
        return self.hidden_layers[: self.num_hidden_layers // 2 + 1]

    @property
    def decoder(self) -> list[Layer]:
        return self.hidden_layers[self.num_hidden_layers // 2 :]

    @property
    def code(self) -> np.ndarray:
        return self.hidden_layers[self.num_hidden_layers // 2].outputs

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        self.forward(inputs)
        return self.code

    def decode(self, inputs: np.ndarray) -> np.ndarray:
        self.forward(inputs)
        return self.output_layer.outputs
