import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants and configuration
@dataclass
class DataLoaderConfig:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: Tuple[int, int]
    num_epochs: int

class DataLoaderMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3

class DataLoaderException(Exception):
    pass

class DataNotLoadedError(DataLoaderException):
    pass

class DataLoadingError(DataLoaderException):
    pass

class DataLoader(ABC):
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.image_size = config.image_size
        self.num_epochs = config.num_epochs
        self.mode = DataLoaderMode.TRAIN

    @abstractmethod
    def load_data(self) -> List[np.ndarray]:
        pass

    @abstractmethod
    def create_dataset(self) -> Dataset:
        pass

    def create_data_loader(self) -> DataLoader:
        dataset = self.create_dataset()
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def load_and_create_data_loader(self) -> DataLoader:
        try:
            data = self.load_data()
            dataset = self.create_dataset(data)
            return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        except DataNotLoadedError as e:
            logging.error(f"Data not loaded: {e}")
            raise
        except DataLoadingError as e:
            logging.error(f"Error loading data: {e}")
            raise

class ImageDataset(Dataset):
    def __init__(self, data: List[np.ndarray], image_size: Tuple[int, int]):
        self.data = data
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data[index]
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        return torch.from_numpy(image).float()

class ImageDataLoader(DataLoader):
    def load_data(self) -> List[np.ndarray]:
        data = []
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    data.append(image)
        if not data:
            raise DataNotLoadedError("No data found in the specified directory")
        return data

    def create_dataset(self, data: List[np.ndarray]) -> Dataset:
        return ImageDataset(data, self.image_size)

class FlowDataset(Dataset):
    def __init__(self, data: List[np.ndarray], image_size: Tuple[int, int]):
        self.data = data
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data[index]
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        return torch.from_numpy(image).float()

class FlowDataLoader(DataLoader):
    def load_data(self) -> List[np.ndarray]:
        data = []
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    data.append(image)
        if not data:
            raise DataNotLoadedError("No data found in the specified directory")
        return data

    def create_dataset(self, data: List[np.ndarray]) -> Dataset:
        return FlowDataset(data, self.image_size)

class VelocityThresholdDataLoader(DataLoader):
    def load_data(self) -> List[np.ndarray]:
        data = []
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    data.append(image)
        if not data:
            raise DataNotLoadedError("No data found in the specified directory")
        return data

    def create_dataset(self, data: List[np.ndarray]) -> Dataset:
        return ImageDataset(data, self.image_size)

class FlowTheoryDataLoader(DataLoader):
    def load_data(self) -> List[np.ndarray]:
        data = []
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    data.append(image)
        if not data:
            raise DataNotLoadedError("No data found in the specified directory")
        return data

    def create_dataset(self, data: List[np.ndarray]) -> Dataset:
        return FlowDataset(data, self.image_size)

def create_data_loader(config: DataLoaderConfig) -> DataLoader:
    if config.mode == DataLoaderMode.TRAIN:
        return ImageDataLoader(config)
    elif config.mode == DataLoaderMode.VALIDATION:
        return FlowDataLoader(config)
    elif config.mode == DataLoaderMode.TEST:
        return VelocityThresholdDataLoader(config)
    else:
        raise ValueError("Invalid data loader mode")

def main():
    config = DataLoaderConfig(
        data_dir="/path/to/data",
        batch_size=32,
        num_workers=4,
        image_size=(224, 224),
        num_epochs=10
    )
    data_loader = create_data_loader(config)
    data_loader.load_and_create_data_loader()

if __name__ == "__main__":
    main()