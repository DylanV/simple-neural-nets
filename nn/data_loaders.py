"""Utilities for loading datasets"""
import gzip
import os
import struct
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np


class MNISTLoader:

    def __init__(self, data_cache_dir: str = os.path.join(os.getcwd(), 'data', 'mnist')):
        """Utility for loading the MNIST dataset from Yann LeCunn's website

        Parameters
        ----------
        data_cache_dir : str
            Path where MNIST gzipped files will be cached
        """
        self.data_cache_dir = data_cache_dir
        # Training set
        self.train_data = None
        self.train_labels = None
        # Test set
        self.test_data = None
        self.test_labels = None

        # url data
        self.base_url = 'http://yann.lecun.com/exdb/mnist/'
        self.train_images_filename = 'train-images-idx3-ubyte.gz'
        self.train_labels_filename = 'train-labels-idx1-ubyte.gz'
        self.test_images_filename = 't10k-images-idx3-ubyte.gz'
        self.test_labels_filename = 't10k-labels-idx1-ubyte.gz'

    def _download_dataset(self):
        """Download the MNIST dataset if it is not already cached"""
        try:
            os.makedirs(self.data_cache_dir)
        except FileExistsError:
            pass

        # The files that need to be downloaded
        filenames = [self.train_images_filename, self.train_labels_filename,
                     self.test_images_filename, self.test_labels_filename]

        # Files that already exist in the cache directory
        cached_files = os.listdir(self.data_cache_dir)

        for fname in filenames:
            if fname not in cached_files:
                down_dir = os.path.join(self.data_cache_dir, fname)
                url = self.base_url + fname
                urlretrieve(url, down_dir)

    def _load_idx_file(self, file_path: str) -> np.ndarray:
        """Load an idx file at the given file_path into an ndarray"""
        with gzip.open(file_path, 'rb') as f:
            # The first 4 bytes give us the data type and number of dimensions
            # First two bytes are zero
            _, data_type, dims = struct.unpack('>HBB', f.read(4))
            # Now that we know the number of dimensions we can get the shape of each
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            # With the shape load the raw data and use numpy to get it in a usable form
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

    def get_training_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Load the MNISt training set

        Returns
        -------
        (train_data, train_labels) : tuple of ndarray
            The training images and labels

        """
        self._download_dataset()  # Make sure the files are downloaded

        data = self._load_idx_file(os.path.join(self.data_cache_dir, self.train_images_filename))
        labels = self._load_idx_file(os.path.join(self.data_cache_dir, self.train_labels_filename))

        return data, labels

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Load the MNISt test set

        Returns
        -------
        (test_data, test_labels) : tuple of ndarray
            The test images and labels

        """
        self._download_dataset()  # Make sure the files are downloaded

        data = self._load_idx_file(os.path.join(self.data_cache_dir, self.test_images_filename))
        labels = self._load_idx_file(os.path.join(self.data_cache_dir, self.test_labels_filename))

        return data, labels

    def get_train_and_test_set(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the train and test datasets

        Returns
        -------
        (train_data, train_labels, test_data, test_labels) : tuple of ndarray
            The full MNIST dataset

        """
        train_data, train_labels = self.get_training_set()
        test_data, test_labels = self.get_test_set()

        return train_data, train_labels, test_data, test_labels
