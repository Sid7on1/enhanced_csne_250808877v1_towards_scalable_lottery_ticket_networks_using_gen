import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'model': {
        'name': 'default_model',
        'architecture': 'resnet50',
        'pretrained': True
    },
    'training': {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    },
    'data': {
        'path': '/path/to/data',
        'split': 0.8
    }
}

# Define an Enum for logging levels
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

# Define a dataclass for configuration
@dataclass
class Config:
    model: Dict[str, str]
    training: Dict[str, float]
    data: Dict[str, str]

# Define a context manager for loading configuration
@contextmanager
def load_config(config_file: str = CONFIG_FILE):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        yield config
    except FileNotFoundError:
        logger.error(f'Configuration file not found: {config_file}')
        yield DEFAULT_CONFIG
    except yaml.YAMLError as e:
        logger.error(f'Error parsing configuration file: {e}')
        yield DEFAULT_CONFIG

# Define a function to validate configuration
def validate_config(config: Config) -> bool:
    # Validate model configuration
    if not config.model['name']:
        logger.error('Model name is required')
        return False
    if not config.model['architecture']:
        logger.error('Model architecture is required')
        return False
    if not config.model['pretrained']:
        logger.error('Pretrained flag is required')
        return False

    # Validate training configuration
    if not config.training['batch_size']:
        logger.error('Batch size is required')
        return False
    if not config.training['epochs']:
        logger.error('Number of epochs is required')
        return False
    if not config.training['learning_rate']:
        logger.error('Learning rate is required')
        return False

    # Validate data configuration
    if not config.data['path']:
        logger.error('Data path is required')
        return False
    if not config.data['split']:
        logger.error('Data split is required')
        return False

    return True

# Define a function to save configuration
def save_config(config: Config, config_file: str = CONFIG_FILE):
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Define a function to get configuration
def get_config() -> Config:
    with load_config() as config:
        if not validate_config(config):
            logger.error('Invalid configuration')
            return DEFAULT_CONFIG
        return config

# Define a function to update configuration
def update_config(config: Config, config_file: str = CONFIG_FILE):
    save_config(config, config_file)

# Define a function to print configuration
def print_config(config: Config):
    logger.info('Model configuration:')
    logger.info(f'Name: {config.model["name"]}')
    logger.info(f'Architecture: {config.model["architecture"]}')
    logger.info(f'Pretrained: {config.model["pretrained"]}')
    logger.info('Training configuration:')
    logger.info(f'Batch size: {config.training["batch_size"]}')
    logger.info(f'Epochs: {config.training["epochs"]}')
    logger.info(f'Learning rate: {config.training["learning_rate"]}')
    logger.info('Data configuration:')
    logger.info(f'Path: {config.data["path"]}')
    logger.info(f'Split: {config.data["split"]}')

# Define a function to get configuration as a dictionary
def get_config_dict() -> Dict[str, str]:
    config = get_config()
    return {
        'model': config.model,
        'training': config.training,
        'data': config.data
    }

# Define a function to update configuration from a dictionary
def update_config_dict(config_dict: Dict[str, str], config_file: str = CONFIG_FILE):
    config = Config(
        model=config_dict['model'],
        training=config_dict['training'],
        data=config_dict['data']
    )
    update_config(config, config_file)

# Define a function to print configuration as a dictionary
def print_config_dict(config_dict: Dict[str, str]):
    logger.info('Model configuration:')
    logger.info(f'Name: {config_dict["model"]["name"]}')
    logger.info(f'Architecture: {config_dict["model"]["architecture"]}')
    logger.info(f'Pretrained: {config_dict["model"]["pretrained"]}')
    logger.info('Training configuration:')
    logger.info(f'Batch size: {config_dict["training"]["batch_size"]}')
    logger.info(f'Epochs: {config_dict["training"]["epochs"]}')
    logger.info(f'Learning rate: {config_dict["training"]["learning_rate"]}')
    logger.info('Data configuration:')
    logger.info(f'Path: {config_dict["data"]["path"]}')
    logger.info(f'Split: {config_dict["data"]["split"]}')