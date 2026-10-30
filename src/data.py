"""
Data loading and preprocessing for CIFAR-100.

Builds a stratified 45,000 / 5,000 train/validation split from the official
CIFAR-100 training set, keeps the official 10,000-image test set unchanged,
computes normalization statistics from the training split only, and defines
separate training (augmented) and evaluation (deterministic) transform
pipelines.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class TransformedSubset(Dataset):
    """
    Wraps a dataset subset and applies a specific transform over the items
    without mutating the original dataset.

    This lets the training and validation splits share one untransformed base
    dataset while each applies its own transform (augmented for training,
    deterministic for validation).

    Args:
        dataset (Dataset): Base dataset returning (PIL image, label) pairs.
            Should be constructed without a transform.
        indices (array-like of int): Indices into ``dataset`` that make up
            this subset.
        transform (callable, optional): Transform applied to each image when
            it is fetched. If None, images are returned unchanged.
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        """
        Fetch one item from the subset.

        Args:
            idx (int): Position within this subset (0 to len(self) - 1).

        Returns:
            tuple: (image, label), where image has this subset's transform
            applied and label is an integer class index (0-99).
        """
        # Fetch raw/untransformed data from base dataset
        # (Ensure base dataset doesn't have an active conflicting transform)
        img, label = self.dataset[self.indices[idx]]

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """
        Return the number of items in the subset.

        Returns:
            int: Number of indices in this subset.
        """
        return len(self.indices)


def compute_stats(trainval_set, train_indices):
    """
    Compute per-channel mean and standard deviation from the training split.

    Statistics are computed over all pixels of all training images together
    (reducing over the image, height, and width axes), using only the training
    indices so no validation or test data leaks into normalization.

    Args:
        trainval_set (torchvision.datasets.CIFAR100): Official CIFAR-100
            training set. Its ``.data`` attribute is a uint8 array of shape
            (N, 32, 32, 3).
        train_indices (array-like of int): Indices of the training split.

    Returns:
        tuple: (mean, std), each a numpy array of shape (3,) giving per-channel
        (RGB) values on the [0, 1] scale, matching the output of
        ``ToDtype(torch.float32, scale=True)``.
    """
    train_data = trainval_set.data[train_indices]
    average = train_data.mean(axis=(0,1,2))
    std = train_data.std(axis=(0,1,2))
    return average/255, std/255


def transforms(train, t_mean,t_std):
    """
    Build the image transform pipeline for training or evaluation.

    The training pipeline applies random augmentation (padded random crop,
    horizontal flip, RandAugment, and random erasing) around tensor
    conversion and normalization. The evaluation pipeline applies only
    tensor conversion and normalization, so it is deterministic.

    Args:
        train (bool): If True, return the augmented training pipeline;
            otherwise return the deterministic evaluation pipeline.
        t_mean (sequence of float): Per-channel mean for normalization,
            on the [0, 1] scale.
        t_std (sequence of float): Per-channel standard deviation for
            normalization, on the [0, 1] scale.

    Returns:
        v2.Compose: The transform pipeline. Takes a PIL image and returns a
        normalized float32 tensor of shape (3, 32, 32).
    """
    if train:
        transforms = v2.Compose([
            v2.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=t_mean, std=t_std),
            v2.RandomErasing(p=0.25)
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=t_mean, std=t_std)
        ])
    return transforms


def build_datasets(data_dir, seed):
    """
    Load CIFAR-100 and build the training, validation, and test datasets.

    Splits the official 50,000-image training set into 45,000 training and
    5,000 validation images using a stratified split (450 / 50 per class)
    with a fixed random seed. The official 10,000-image test set is used
    unchanged. Normalization statistics are computed from the training split
    only. The training split uses the augmented pipeline; validation and
    test use the deterministic evaluation pipeline.

    Args:
        data_dir (str): Directory where CIFAR-100 is stored or downloaded to.
        seed (int): Random seed for the stratified train/validation split.

    Returns:
        tuple: (train_set, val_set, test_set), where train_set and val_set
        are TransformedSubset instances and test_set is a
        torchvision.datasets.CIFAR100 instance.
    """
    trainval_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True)
    targets = np.array(trainval_set.targets)
    train_indices, val_indices = train_test_split(
        np.arange(len(targets)),
        test_size= 5000,
        shuffle=True,
        random_state=seed,
        stratify=targets
    )
    t_mean,t_std = compute_stats(trainval_set, train_indices)
    train_set = TransformedSubset(trainval_set, train_indices,
                                  transform=transforms(True, t_mean, t_std))
    val_set= TransformedSubset(trainval_set, val_indices,
                               transform=transforms(False, t_mean, t_std))
    test_set = torchvision.datasets.CIFAR100(root=data_dir,
                                             train=False,download=True,
                                             transform=transforms(False, t_mean, t_std)
    )
    return train_set, val_set, test_set
