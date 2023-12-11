import numpy as np
from numpy.lib.stride_tricks import as_strided

from assignment_3.layers.base import Layer


class MaxPoolLayer(Layer):
    def __init__(
        self,
        pool_size: int = 2,
        stride: int = 2,
    ):
        self.pool_size = pool_size
        self.stride = stride

        # Attributes related to forward pass
        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None
        # Attributes related to backward pass
        self.d_inputs: np.ndarray | None = None
        # Attributes related to both forward and backward pass
        self.inputs_as_strided: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        batch_size, pools, input_height, input_width = inputs.shape

        output_h = (input_height - self.pool_size) // self.stride + 1
        output_w = (input_width - self.pool_size) // self.stride + 1

        self.inputs_as_strided = as_strided(
            self.inputs,
            shape=(
                batch_size,
                pools,
                output_h,
                output_w,
                self.pool_size,
                self.pool_size,
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
        self.outputs = np.max(self.inputs_as_strided, axis=(4, 5))
        return self.outputs

    def backward(self, d_values: np.ndarray) -> None:
        batch_size, pools, input_height, input_width = self.inputs.shape
        _, _, output_height, output_width = d_values.shape
        self.d_inputs = np.zeros_like(self.inputs)

        for row_index in range(output_height):
            for column_index in range(output_width):
                max_val_indices = np.argmax(
                    self.inputs_as_strided[:, :, row_index, column_index].reshape(
                        batch_size, pools, -1
                    ),
                    axis=2,
                )
                max_val_coordinates = np.unravel_index(
                    max_val_indices, (self.pool_size, self.pool_size)
                )

                for sample_index in range(batch_size):
                    for pool_index in range(pools):
                        row_begin = row_index * self.stride
                        column_begin = column_index * self.stride

                        max_x = max_val_coordinates[0][sample_index, pool_index]
                        max_y = max_val_coordinates[1][sample_index, pool_index]

                        current_pool_d_values = d_values[
                            sample_index, pool_index, row_index, column_index
                        ]
                        self.d_inputs[
                            sample_index,
                            pool_index,
                            row_begin + max_x,
                            column_begin + max_y,
                        ] += current_pool_d_values
        return self.d_inputs
