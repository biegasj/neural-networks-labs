import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MnistDataset:
    def __init__(self, path: str, test_size: float = 0.2):
        self.data = pd.read_csv(path)
        self.X, self.y = self._split_features_and_targets(target_column="label")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )

    def _split_features_and_targets(
        self, target_column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self.data.loc[:, self.data.columns != target_column].values / 255.0
        y = self.data.loc[:, target_column].values.reshape(-1)
        return X, y


class AffnistDataset:
    def __init__(
        self,
        training_data_path: str,
        test_data_path: str,
        load_n_batches: int = 1,
    ):
        self.load_n_batches = load_n_batches
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self._load_features_and_targets(
            training_data_path, test_data_path, load_n_batches
        )

    @staticmethod
    def _load_features_and_targets(
        training_data_path: str, test_data_path: str, load_n_batches: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = [], [], [], []
        for batch_index in range(load_n_batches):
            X_train.append(
                np.load(
                    os.path.join(training_data_path, f"images_{batch_index + 1}.npy")
                )
                / 255.0
            )
            X_test.append(
                np.load(os.path.join(test_data_path, f"images_{batch_index + 1}.npy"))
                / 255.0
            )
            y_train.append(
                np.load(
                    os.path.join(training_data_path, f"labels_{batch_index + 1}.npy")
                )
            )
            y_test.append(
                np.load(os.path.join(test_data_path, f"labels_{batch_index + 1}.npy"))
            )
        X_train, X_test, y_train, y_test = (
            np.concatenate(X_train),
            np.concatenate(X_test),
            np.concatenate(y_train),
            np.concatenate(y_test),
        )
        return X_train, X_test, y_train, y_test


class ForestFiresDataset:
    def __init__(self, path: str, test_size: float = 0.2):
        self.data = pd.read_csv(path)
        self.test_size = test_size
        self.X, self.y = self._split_features_and_targets(target_column="area")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )

    def _split_features_and_targets(
        self, target_column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self.data.loc[:, self.data.columns != target_column].values
        y = self.data.loc[:, target_column].values.reshape(-1, 1)
        return X, y


if __name__ == "__main__":
    affnist_dataset = AffnistDataset(
        "data/processed/affnist/training_batches", "data/processed/affnist/test_batches"
    )
    ff_dataset = ForestFiresDataset("data/processed/forest+fires/forestfires.csv")
