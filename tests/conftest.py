"""
Shared pytest fixtures for the CIFAR-100 test suite
"""

import os
import pytest
from src.data import build_datasets, DEFAULT_SEED


@pytest.fixture(scope="session",name="datasets")
def fixture_datasets():
    """
    Fixture to provide the 45000/5000/10000 train/val/test split of
    CIFAR-100 dataset.

    Returns:
        tuple: (train_set, val_set, test_set)
    """
    data_dir = os.environ.get("DATA_DIR", "data")
    return build_datasets(data_dir, DEFAULT_SEED)


@pytest.fixture(scope="session")
def train_set(datasets):
    """
    Fixture to provide the training split of CIFAR-100.

    Returns:
        TransformedSubset: The 45000-image training split, with the
        augmented training transform. Raw images are available through
        ``.dataset.data`` indexed by ``.indices``.
    """
    return datasets[0]


@pytest.fixture(scope="session")
def val_set(datasets):
    """
    Fixture to provide the validation split of CIFAR-100.

    Returns:
        TransformedSubset: The 5000-image validation split, with the
        deterministic evaluation transform. Raw images are available
        through ``.dataset.data`` indexed by ``.indices``.
    """
    return datasets[1]


@pytest.fixture(scope="session")
def test_set(datasets):
    """
    Fixture to provide the test split of CIFAR-100.

    Returns:
        torchvision.datasets.CIFAR100: The official 10000-image test set,
        with the deterministic evaluation transform. Raw images are
        available directly through ``.data``.
    """
    return datasets[2]
