import matplotlib.pyplot as plt
import numpy as np

from assignment_3.layers import Layer
from assignment_3.networks import NeuralNetwork

FASHION_MNIST_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def plot_samples(X: np.ndarray, y: np.ndarray) -> None:
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))

    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(FASHION_MNIST_LABELS[y[i]])

    plt.tight_layout()
    plt.show()


def plot_original_and_reconstructed(
    X_original: np.ndarray, X_reconstructed: np.ndarray, y: np.ndarray
) -> None:
    num_row, num_col = 2, 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))

    for i in range(num_col):
        ax = axes[0, i % num_col]
        ax.imshow(X_original[i].reshape(28, 28), cmap="gray")
        ax.set_title(FASHION_MNIST_LABELS[y[i]])

    for i in range(num_col):
        ax = axes[1, i % num_col]
        ax.imshow(X_reconstructed[i].reshape(28, 28), cmap="gray")
        ax.set_title(FASHION_MNIST_LABELS[y[i]])

    plt.tight_layout()
    plt.show()


def plot_autoencoder_layers(layers: list[Layer], label: str) -> None:
    num_col = len(layers)
    fig, axes = plt.subplots(1, num_col, figsize=(1.5 * num_col, 2))

    for i in range(num_col):
        ax = axes[i % num_col]
        ax.imshow(layers[i], cmap="gray")
        ax.set_title(label)

    plt.tight_layout()
    plt.show()


def plot_predictions_randomly(
    neural_network: NeuralNetwork, X: np.ndarray, y: np.ndarray, shape: int = 28
) -> None:
    num_row = 3
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))

    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        random_index = np.random.randint(0, X.shape[0])
        ax.imshow(X[random_index].reshape(shape, shape), cmap="gray")
        prediction = np.argmax(neural_network.forward(X[random_index]), axis=1)
        correct_label = y[random_index]
        ax.set_title("pred={}, true={}".format(prediction[0], correct_label))
    plt.tight_layout()
    plt.show()
