import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing

MONTH_MAPPING = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
DAY_MAPPING = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}


class BasePreprocessor(ABC):
    input_dir: str | Path
    output_dir: str | Path

    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def process(self) -> None:
        pass


class AffnistPreprocessor(BasePreprocessor):
    def process(self) -> None:
        file_contents = loadmat(self.input_path)
        dataset = file_contents["affNISTdata"]
        images = dataset["image"][0][0].transpose().astype(np.float64) / 255.0
        labels = dataset["label_int"][0][0][0].astype(np.int64)

        np.save(self._add_prefix_to_filename(self.output_path, "images"), images)
        np.save(self._add_prefix_to_filename(self.output_path, "labels"), labels)

    @staticmethod
    def _add_prefix_to_filename(path: str, prefix: str) -> str:
        dirname, filename = os.path.split(path)
        return os.path.join(dirname, f"{prefix}_{filename}")


class ForestFiresDatasetPreprocessor(BasePreprocessor):
    def process(self) -> None:
        df = pd.read_csv(self.input_path)

        scaler = preprocessing.MinMaxScaler()
        df["area"] = np.log1p(df["area"])
        df["month"] = df["month"].map(MONTH_MAPPING)
        df["day"] = df["day"].map(DAY_MAPPING)

        df.to_csv(self.output_path, index=False)


if __name__ == "__main__":
    forest_fires_preprocessor = ForestFiresDatasetPreprocessor(
        input_path="data/raw/forest+fires/forestfires.csv",
        output_path="data/processed/forest+fires/forestfires.csv",
    )
    forest_fires_preprocessor.process()
    #
    # for batch_index in range(32):
    #     affnist_preprocessor = AffnistPreprocessor(
    #         input_path=f"data/raw/affnist/training_batches/{batch_index + 1}.mat",
    #         output_path=f"data/processed/affnist/training_batches/{batch_index + 1}.npy",
    #     )
    #     affnist_preprocessor.process()
