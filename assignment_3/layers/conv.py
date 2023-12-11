import numpy as np
from numpy.lib.stride_tricks import as_strided

from assignment_3.layers.base import Layer
from assignment_3.utils.activation_functions import ActivationFunction, ReLUActivation


class ConvLayer(Layer):
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size,
        stride=1,
        padding=0,
        activation_function: ActivationFunction = ReLUActivation,
    ):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(
            self.channels_out, self.channels_in, self.kernel_size, self.kernel_size
        ) * np.sqrt(2.0 / (self.channels_in * self.kernel_size * self.kernel_size))
        self.stride = stride
        self.padding = padding

        self.weights = self.kernels
        self.biases = np.zeros(channels_out)
        self.activation_function = activation_function

        # Attributes related to forward pass
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None
        # Attributes related to backward pass
        self.d_weights: np.ndarray | None = None
        self.d_inputs: np.ndarray | None = None
        self.d_biases: np.ndarray | None = None
        # Attributes related to both forward and backward pass
        self.inputs_as_strided: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = np.pad(
            inputs,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )
        batch_size, channels_in, input_height, input_width = self.inputs.shape
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        self.outputs = np.zeros(
            (batch_size, self.channels_out, output_height, output_width)
        )
        self.inputs_as_strided = as_strided(
            self.inputs,
            shape=(
                batch_size,
                channels_in,
                output_height,
                output_width,
                self.kernel_size,
                self.kernel_size,
            ),
            strides=(
                self.inputs.strides[0],
                self.inputs.strides[1],
                self.inputs.strides[2] * self.stride,
                self.inputs.strides[3] * self.stride,
                self.inputs.strides[2],
                self.inputs.strides[3],
            ),
        )
        for channel_out_index in range(self.channels_out):
            self.outputs[:, channel_out_index, :, :] = (
                np.tensordot(
                    self.inputs_as_strided,
                    self.kernels[channel_out_index, :, :, :],
                    axes=((1, 4, 5), (0, 1, 2)),
                )
                + self.biases[channel_out_index]
            )
        return self.activation_function.forward(self.outputs)

    def backward(self, d_values):
        _, _, output_height, output_width = d_values.shape
        d_values = self.activation_function.backward(d_values)
        batch_size, channels_in, input_height, input_width = self.inputs.shape

        self.d_weights = np.zeros_like(self.kernels)
        self.d_biases = np.zeros_like(self.biases)
        self.d_inputs = np.zeros_like(self.inputs)

        for channel_out_index in range(self.channels_out):
            d_values_reshaped = d_values[:, channel_out_index, :, :].reshape(
                batch_size, 1, output_height, output_width, 1, 1
            )
            self.d_weights[channel_out_index] = np.sum(
                self.inputs_as_strided * d_values_reshaped, axis=(0, 2, 3)
            )
            self.d_biases[channel_out_index] = np.sum(
                d_values[:, channel_out_index, :, :], axis=(0, 1, 2)
            )

        rotated_kernels = self.kernels[:, :, ::-1, ::-1]
        d_values_padded = np.pad(
            d_values,
            (
                (0, 0),
                (0, 0),
                (self.kernel_size - 1, self.kernel_size - 1),
                (self.kernel_size - 1, self.kernel_size - 1),
            ),
            mode="constant",
        )
        d_values_as_strided = as_strided(
            d_values_padded,
            shape=(
                batch_size,
                self.channels_out,
                input_height,
                input_width,
                self.kernel_size,
                self.kernel_size,
            ),
            strides=(
                d_values_padded.strides[0],
                d_values_padded.strides[1],
                d_values_padded.strides[2] * self.stride,
                d_values_padded.strides[3] * self.stride,
                d_values_padded.strides[2],
                d_values_padded.strides[3],
            ),
        )
        self.d_inputs = np.tensordot(
            d_values_as_strided, rotated_kernels, axes=((1, 4, 5), (0, 2, 3))
        )

        if self.padding:
            self.d_inputs = self.d_inputs[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        return self.d_inputs
