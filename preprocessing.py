import logging
import os
import random
import tempfile
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "image_dir": "images",
    "output_dir": "preprocessed_images",
    "temp_dir": tempfile.gettempdir(),
    "valid_extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
    "logging_level": logging.INFO,
    "debug_mode": False,
    "performance_metrics": True,
}


class ImagePreprocessor:
    """
    Image Preprocessor class for performing various preprocessing operations on images.

    ...

    Attributes
    ----------
    image_dir : str
        Path to the directory containing input images
    output_dir : str
        Path to the directory where preprocessed images will be saved
    temp_dir : str
        Path to the temporary directory for storing intermediate files
    valid_extensions : list of str
        List of valid image file extensions
    logging_level : int
        Logging level for the logger
    debug_mode : bool
        Flag for enabling debug mode, which provides more detailed logging
    performance_metrics : bool
        Flag for enabling performance metrics tracking

    Methods
    -------
    preprocess_images(self, image_paths: Union[str, List[str]] = None)
        Preprocess a single image or a list of images
    crop_images(self, image: np.array, crop_size: Tuple[int, int] = (224, 224))
        Crop an image to a specified size
    resize_images(self, image: np.array, resize_size: Tuple[int, int] = (224, 224))
        Resize an image to a specified size
    augment_images(self, image: np.array)
        Perform data augmentation on an image
    save_preprocessed_images(self, preprocessed_images: List[np.array], output_dir: str)
        Save preprocessed images to a specified directory
    ...

    """

    def __init__(
        self,
        image_dir: str = CONFIG["image_dir"],
        output_dir: str = CONFIG["output_dir"],
        temp_dir: str = CONFIG["temp_dir"],
        valid_extensions: List[str] = CONFIG["valid_extensions"],
        logging_level: int = CONFIG["logging_level"],
        debug_mode: bool = CONFIG["debug_mode"],
        performance_metrics: bool = CONFIG["performance_metrics"],
    ):
        """
        Initialize the ImagePreprocessor class with the given configurations.

        Parameters
        ----------
        image_dir : str, optional
            Path to the directory containing input images, by default CONFIG['image_dir']
        output_dir : str, optional
            Path to the directory where preprocessed images will be saved, by default CONFIG['output_dir']
        temp_dir : str, optional
            Path to the temporary directory for storing intermediate files, by default CONFIG['temp_dir']
        valid_extensions : list of str, optional
            List of valid image file extensions, by default CONFIG['valid_extensions']
        logging_level : int, optional
            Logging level for the logger, by default CONFIG['logging_level']
        debug_mode : bool, optional
            Flag for enabling debug mode, providing more detailed logging, by default CONFIG['debug_mode']
        performance_metrics : bool, optional
            Flag for enabling performance metrics tracking, by default CONFIG['performance_metrics']

        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.valid_extensions = valid_extensions
        self.logging_level = logging_level
        self.debug_mode = debug_mode
        self.performance_metrics = performance_metrics

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set logging level
        logger.setLevel(self.logging_level)

        if self.debug_mode:
            logger.debug("Debug mode enabled. Detailed logging will be provided.")

    def _validate_image(self, image_path: str) -> bool:
        """
        Validate if a given file is an image with a supported extension.

        Parameters
        ----------
        image_path : str
            Path to the image file

        Returns
        -------
        bool
            True if the file is a valid image, False otherwise

        """
        filename, file_extension = os.path.splitext(image_path)
        return file_extension.lower() in self.valid_extensions

    def _load_image(self, image_path: str) -> Optional[np.array]:
        """
        Load an image from the given file path.

        Parameters
        ----------
        image_path : str
            Path to the image file

        Returns
        -------
        np.array or None
            Loaded image as a numpy array, or None if the image is invalid

        """
        if not self._validate_image(image_path):
            logger.error(f"Invalid image format: {image_path}")
            return None

        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image: {image_path}, {e}")
            return None

    def _save_image(self, image: np.array, output_path: str) -> bool:
        """
        Save an image to the specified output path.

        Parameters
        ----------
        image : np.array
            Image to be saved
        output_path : str
            Path to save the image

        Returns
        -------
        bool
            True if the image is saved successfully, False otherwise

        """
        try:
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            logger.error(f"Error saving image: {output_path}, {e}")
            return False

    def _crop_image(self, image: np.array, crop_size: Tuple[int, int] = (224, 224)) -> np.array:
        """
        Crop an image to a specified size.

        Parameters
        ----------
        image : np.array
            Input image to be cropped
        crop_size : tuple of int, optional
            Size to crop the image to, by default (224, 224)

        Returns
        -------
        np.array
            Cropped image

        """
        height, width, _ = image.shape
        crop_height, crop_width = crop_size

        # Handle invalid crop size
        if crop_height <= 0 or crop_width <= 0 or crop_height > height or crop_width > width:
            logger.error(f"Invalid crop size: {crop_size}")
            return image

        # Randomly select crop position
        x_start = random.randint(0, width - crop_width)
        y_start = random.randint(0, height - crop_height)

        # Crop the image
        return image[y_start : y_start + crop_height, x_start : x_start + crop_width, :]

    def _resize_image(self, image: np.array, resize_size: Tuple[int, int] = (224, 224)) -> np.array:
        """
        Resize an image to a specified size.

        Parameters
        ----------
        image : np.array
            Input image to be resized
        resize_size : tuple of int, optional
            Size to resize the image to, by default (224, 224)

        Returns
        -------
        np.array
            Resized image

        """
        height, width, _ = image.shape
        resize_height, resize_width = resize_size

        # Handle invalid resize size
        if resize_height <= 0 or resize_width <= 0:
            logger.error(f"Invalid resize size: {resize_size}")
            return image

        # Resize the image
        return cv2.resize(image, (resize_width, resize_height))

    def _augment_image(self, image: np.array) -> np.array:
        """
        Perform data augmentation on an image by applying random transformations.

        Parameters
        ----------
        image : np.array
            Input image to be augmented

        Returns
        -------
        np.array
            Augmented image

        """
        # Randomly flip the image horizontally
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Randomly rotate the image by a small angle
        angle = random.uniform(-10, 10)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return image

    def preprocess_images(
        self, image_paths: Union[str, List[str]] = None
    ) -> Tuple[List[np.array], List[str]]:
        """
        Preprocess a single image or a list of images by performing cropping, resizing, and augmentation.

        Parameters
        ----------
        image_paths : str or list of str, optional
            Path to a single image file or a list of image file paths to preprocess, by default None
            If None, all images in the 'image_dir' will be preprocessed

        Returns
        -------
        tuple of list of np.array and list of str
            List of preprocessed images and their corresponding original file paths

        """
        if image_paths is None:
            image_paths = [
                os.path.join(self.image_dir, filename) for filename in os.listdir(self.image_dir)
            ]
        elif isinstance(image_paths, str):
            image_paths = [image_paths]

        preprocessed_images = []
        original_paths = []

        for image_path in image_paths:
            image = self._load_image(image_path)
            if image is None:
                continue

            # Crop the image
            image = self._crop_image(image)

            # Resize the image
            image = self._resize_image(image)

            # Augment the image
            image = self._augment_image(image)

            preprocessed_images.append(image)
            original_paths.append(image_path)

        return preprocessed_images, original_paths

    def save_preprocessed_images(
        self, preprocessed_images: List[np.array], original_paths: List[str]
    ) -> None:
        """
        Save preprocessed images to the specified output directory.

        Parameters
        ----------
        preprocessed_images : list of np.array
            List of preprocessed images to be saved
        original_paths : list of str
            List of original file paths corresponding to the preprocessed images

        Returns
        -------
        None

        """
        for image, original_path in zip(preprocessed_images, original_paths):
            filename = os.path.basename(original_path)
            output_path = os.path.join(self.output_dir, filename)
            self._save_image(image, output_path)

    def _calculate_performance_metrics(
        self, image_paths: List[str], start_time: float
    ) -> pd.DataFrame:
        """
        Calculate and log performance metrics for the preprocessing.

        Parameters
        ----------
        image_paths : list of str
            List of image file paths that were preprocessed
        start_time : float
            Start time of the preprocessing

        Returns
        -------
        pd.DataFrame
            Dataframe containing performance metrics

        """
        end_time = time.time()
        total_time = end_time - start_time
        images_per_second = len(image_paths) / total_time if total_time > 0 else 0

        metrics = {
            "total_images": len(image_paths),
            "total_time": total_time,
            "images_per_second": images_per_second,
        }

        df = pd.DataFrame(metrics, index=[0])
        return df

    def run_preprocessing(self) -> None:
        """
        Main function to perform image preprocessing.

        It loads images from the 'image_dir', preprocesses them, saves the preprocessed images,
        and logs performance metrics if enabled.

        Returns
        -------
        None

        """
        logger.info("Starting image preprocessing...")

        start_time = time.time()

        # Preprocess images
        preprocessed_images, original_paths = self.preprocess_images()

        # Save preprocessed images
        self.save_preprocessed_images(preprocessed_images, original_paths)

        # Log performance metrics if enabled
        if self.performance_metrics:
            metrics_df = self._calculate_performance_metrics(original_paths, start_time)
            logger.info(
                "Image preprocessing performance metrics:\n{}".format(metrics_df)
            )

        logger.info("Image preprocessing completed.")


# Example usage
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    preprocessor.run_preprocessing()