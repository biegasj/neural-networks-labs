import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

tensor = np.random.rand(5, 5)
print(tensor)

windows = sliding_window_view(tensor, (2, 2), axis=(-1, -2))
print(windows)

print(tensor.shape, windows.shape)

kernel = np.random.rand(2, 2)
hadamard_product = kernel * windows
print(hadamard_product)

# Cross correlation, feature_map
cross_correlation = hadamard_product.sum((-1, -2))
print(cross_correlation)

# Sigmoid
cc_act = 1.0 / (1.0 + np.exp(cross_correlation))
print(cc_act)


# Create an example array with the original shape
original_shape = (337, 1, 26, 26, 3, 3)
original_array = np.random.rand(*original_shape)

# Calculate the strides for the new shape
target_shape = (337, 26, 26, 5, 3, 3)
target_strides = (
    original_array.strides[0],
    original_array.strides[2],
    original_array.strides[3],
    original_array.strides[1],
    original_array.strides[4],
    original_array.strides[5],
)

# Create a view with the new shape and strides
view_array = np.lib.stride_tricks.as_strided(
    original_array, shape=target_shape, strides=target_strides
)

# Check the shapes
print("Original shape:", original_array.shape)
print("View shape:", view_array.shape)

print("original")
print(original_array[0])

print("view")
print(view_array[0])


original_array = np.random.random((337, 1, 5, 5, 2, 2))
squeezed_array = np.squeeze(original_array, axis=1)
expanded_array = np.expand_dims(squeezed_array, axis=3)
repeated_array = np.repeat(expanded_array, repeats=5, axis=3)
print(repeated_array)


# Reshape using np.repeat
reshaped_array = np.repeat(expanded_array, repeats=5, axis=3)

# Squeeze the singleton dimension

# Print the shape of the reshaped array
print(reshaped_array.shape)
