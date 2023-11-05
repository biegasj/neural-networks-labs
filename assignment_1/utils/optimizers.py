import numpy as np

from assignment_1.models import NeuralNetwork
from assignment_1.utils.loss_functions import LossFunction
from assignment_1.utils.metrics import classification_accuracy, regression_accuracy


class MiniBatchOptimizer:
    def __init__(
        self,
        neural_network: NeuralNetwork,
        *,
        loss_function: LossFunction,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        decay: float | None = None,
        classification: bool = False,
    ):
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_of_updates = 0
        self.classification = classification

        self.all_loss_values = []
        self.all_accuracy_values = []

    def optimize(self, X: np.ndarray, y: np.ndarray) -> None:
        total_batches = X.shape[0] // self.batch_size
        precision = np.std(y) if not self.classification else None

        print(f"Started optimization of {self.neural_network.__class__.__name__}, classification={self.classification}")
        for epoch_index in range(self.epochs):
            for batch_index in range(total_batches):
                X_batch, y_batch = self._get_batch(batch_index, X, y)
                y_pred_batch = self.neural_network.forward(X_batch)

                if batch_index % 10 == 0:
                    loss = self.loss_function.calculate(y_pred_batch, y_batch)
                    self.all_loss_values.append(loss)

                if batch_index % 10 == 0:
                    if self.classification:
                        acc = classification_accuracy(y_pred_batch, y_batch)
                    else:
                        acc = regression_accuracy(y_pred_batch, y_batch, precision)
                    self.all_accuracy_values.append(acc)

                d_loss = self.loss_function.backward(y_pred_batch, y_batch)
                self.neural_network.backward(d_loss)
                self._update_params()
            print(f"Epoch {epoch_index + 1}  --  accuracy {acc:.7f} - loss {loss:.7f}")

    def _get_batch(
        self, batch_index: int, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        begin_index = batch_index * self.batch_size
        end_index = begin_index + self.batch_size
        return X[begin_index:end_index], y[begin_index:end_index]

    def _update_params(self) -> None:
        for layer in self.neural_network.all_layers:
            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases
        if self.decay:
            self.learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.num_of_updates)
            )
            self.num_of_updates += 1
