from abc import ABC, abstractmethod

import numpy as np


class WeightInitializer(ABC):
    @staticmethod
    @abstractmethod
    def initialize(input_dim: int, output_dim: int) -> np.ndarray:
        pass


class RandomWeightInitializer(WeightInitializer):
    @staticmethod
    def initialize(input_dim: int, output_dim: int) -> np.ndarray:
        return np.random.randn(input_dim, output_dim) * 0.01


class XavierWeightInitializer(WeightInitializer):
    @staticmethod
    def initialize(input_dim: int, output_dim: int) -> np.ndarray:
        return np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)


class HeWeightInitializer(WeightInitializer):
    @staticmethod
    def initialize(input_dim: int, output_dim: int) -> np.ndarray:
        return np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
