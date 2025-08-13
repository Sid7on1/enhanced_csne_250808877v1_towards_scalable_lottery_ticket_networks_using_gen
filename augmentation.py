import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import cv2
import random
import math
from typing import List, Tuple, Dict
from augmentation.config import Config
from augmentation.exceptions import AugmentationError
from augmentation.utils import get_logger, get_transform

logger = get_logger(__name__)

class DataAugmentation:
    def __init__(self, config: Config):
        self.config = config
        self.transform = get_transform(config)

    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        try:
            image = self.transform(image)
            return image
        except Exception as e:
            raise AugmentationError(f"Failed to apply augmentation: {str(e)}")

    def random_flip(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        return image

    def random_rotation(self, image: np.ndarray) -> np.ndarray:
        angle = random.uniform(-10, 10)
        image = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        image = cv2.warpAffine(image, image, (image.shape[1], image.shape[0]))
        return image

    def random_scaling(self, image: np.ndarray) -> np.ndarray:
        scale = random.uniform(0.9, 1.1)
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        return image

    def random_color_jitter(self, image: np.ndarray) -> np.ndarray:
        brightness = random.uniform(0.5, 1.5)
        contrast = random.uniform(0.5, 1.5)
        saturation = random.uniform(0.5, 1.5)
        hue = random.uniform(-0.1, 0.1)

        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[..., 1] = cv2.convertScaleAbs(image[..., 1], alpha=saturation, beta=0)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[..., 0] = cv2.convertScaleAbs(image[..., 0], alpha=1, beta=hue * 180)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def random_blur(self, image: np.ndarray) -> np.ndarray:
        blur_type = random.choice(['gaussian', 'median', 'bilateral'])
        if blur_type == 'gaussian':
            kernel_size = random.randint(3, 11)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'median':
            kernel_size = random.randint(3, 11)
            image = cv2.medianBlur(image, kernel_size)
        elif blur_type == 'bilateral':
            diameter = random.randint(3, 11)
            image = cv2.bilateralFilter(image, diameter, 10, 10)
        return image

    def random_noise(self, image: np.ndarray) -> np.ndarray:
        noise_type = random.choice(['gaussian', 'salt', 'pepper'])
        if noise_type == 'gaussian':
            mean = random.uniform(-10, 10)
            std = random.uniform(0.1, 1.0)
            image = image + np.random.normal(mean, std, image.shape)
        elif noise_type == 'salt':
            salt_prob = random.uniform(0.01, 0.1)
            image = np.where(np.random.rand(*image.shape) < salt_prob, 255, image)
        elif noise_type == 'pepper':
            pepper_prob = random.uniform(0.01, 0.1)
            image = np.where(np.random.rand(*image.shape) < pepper_prob, 0, image)
        return image

class AugmentationError(Exception):
    pass

class Config:
    def __init__(self):
        self.augmentation = {
            'flip': True,
            'rotation': True,
            'scaling': True,
            'color_jitter': True,
            'blur': True,
            'noise': True
        }

def get_transform(config: Config) -> transforms.Compose:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if config.augmentation['flip']:
        transform.transforms.append(transforms.RandomHorizontalFlip())
    if config.augmentation['rotation']:
        transform.transforms.append(transforms.RandomRotation(10))
    if config.augmentation['scaling']:
        transform.transforms.append(transforms.RandomScale(0.9, 1.1))
    if config.augmentation['color_jitter']:
        transform.transforms.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
    if config.augmentation['blur']:
        transform.transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5))
    if config.augmentation['noise']:
        transform.transforms.append(transforms.RandomApply([transforms.ToTensor(), transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)], p=0.5))
    return transform

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    config = Config()
    augmentation = DataAugmentation(config)
    image = np.random.rand(256, 256, 3) * 255
    image = augmentation.apply_augmentation(image)
    logger.info(image.shape)

if __name__ == '__main__':
    main()