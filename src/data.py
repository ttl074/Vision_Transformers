"""
Data loading and preprocessing for CIFAR-100.

Builds a stratified 45,000 / 5,000 train/validation split from the official
CIFAR-100 training set, with exact-duplicate images kept out of validation,
keeps the official 10,000-image test set unchanged, computes normalization
statistics from the training split only, and defines separate training
(augmented) and evaluation (deterministic) transform pipelines, plus the
batch-level MixUp/CutMix augmentation applied during training.
"""

import hashlib
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 2026


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
    average = train_data.mean(axis=(0, 1, 2))
    std = train_data.std(axis=(0, 1, 2))
    return average / 255, std / 255


def build_transforms(train, t_mean, t_std):
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
        transforms = v2.Compose(
            [
                v2.RandomCrop(size=32, padding=4, padding_mode="reflect"),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandAugment(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=t_mean, std=t_std),
                v2.RandomErasing(p=0.25),
            ]
        )
    else:
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=t_mean, std=t_std),
            ]
        )
    return transforms


def build_datasets(data_dir, seed):
    """
    Load CIFAR-100 and build the training, validation, and test datasets.

    Splits the official 50,000-image training set into 45,000 training and
    5,000 validation images using a stratified split (450 / 50 per class)
    with a fixed random seed. Exact-duplicate images that the split places in
    both train and validation are then moved into train, with same-class
    replacements, so split sizes and per-class counts are unchanged. The
    official 10,000-image test set is used unchanged. Normalization
    statistics are computed from the final training split only. The training
    split uses the augmented pipeline; validation and test use the
    deterministic evaluation pipeline.

    Args:
        data_dir (str): Directory where CIFAR-100 is stored or downloaded to.
        seed (int): Random seed for the stratified train/validation split and
            for choosing duplicate replacements.

    Returns:
        tuple: (train_set, val_set, test_set), where train_set and val_set
        are TransformedSubset instances and test_set is a
        torchvision.datasets.CIFAR100 instance.
    """
    trainval_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True
    )
    targets = np.array(trainval_set.targets)
    train_indices, val_indices = train_test_split(
        np.arange(len(targets)),
        test_size=5000,
        shuffle=True,
        random_state=seed,
        stratify=targets,
    )
    groups = image_hash_groups(trainval_set.data)
    dup_groups = [v for v in groups.values() if len(v) > 1]
    train_indices, val_indices = _resolve_split_dups(
        train_indices, val_indices, targets, dup_groups, seed
    )
    t_mean, t_std = compute_stats(trainval_set, train_indices)
    train_set = TransformedSubset(
        trainval_set, train_indices, transform=build_transforms(True, t_mean, t_std)
    )
    val_set = TransformedSubset(
        trainval_set, val_indices, transform=build_transforms(False, t_mean, t_std)
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=build_transforms(False, t_mean, t_std),
    )
    return train_set, val_set, test_set


def image_hash_groups(img_arr):
    """
    Group image indices by the MD5 hash of each image's raw bytes.

    Images with identical pixel values share a hash, so any group with more
    than one index is a set of exact duplicates. Near-duplicates (shifted,
    cropped, or recolored versions) hash differently and are not grouped.

    Args:
        img_arr (np.ndarray): Images of shape (N, 32, 32, 3), dtype uint8.

    Returns:
        dict[str, list[int]]: Maps each distinct image's hex digest to the
        list of indices where it appears. Indices are positions in
        ``img_arr``: if a subset is passed (e.g. ``data[train_indices]``),
        they are positions within that subset, not original dataset indices.
    """
    groups = defaultdict(list)
    for idx, img in enumerate(img_arr):
        groups[hashlib.md5(img.tobytes()).hexdigest()].append(idx)
    return dict(groups)


def _swap_dups(train_idx_set, val_idx_set, pool, dup, replacement):
    """
    Move one duplicate from validation to train and one replacement back.

    Modifies all three sets in place: ``dup`` moves from validation to
    train, ``replacement`` moves from train to validation, and
    ``replacement`` is removed from ``pool`` so it cannot be chosen again.

    Args:
        train_idx_set (set of int): Training-split indices.
        val_idx_set (set of int): Validation-split indices.
        pool (set of int): Training indices eligible as replacements.
        dup (int): Validation index of a duplicated image to move to train.
        replacement (int): Training index of a same-class image to move to
            validation.
    """
    val_idx_set.remove(dup)
    val_idx_set.add(replacement)
    train_idx_set.remove(replacement)
    train_idx_set.add(dup)
    pool.remove(replacement)


def _find_split_groups(dup_groups, train_idx_set):
    """
    Return the duplicate groups split across train and validation.

    A group counts as split when exactly one of its indices is in train
    (and the rest are in validation). CIFAR-100's duplicate groups are all
    pairs, so this means one copy in each split.

    Args:
        dup_groups (list of list of int): Groups of indices whose images are
            exact duplicates.
        train_idx_set (set of int): Training-split indices.

    Returns:
        list of list of int: The groups with copies in both splits.
    """
    splits = []
    for group in dup_groups:
        s = sum(idx in train_idx_set for idx in group)
        if s == 1:
            splits.append(group)
    return splits


def _resolve_split_dups(train_indices, val_indices, targets, dup_groups, seed):
    """
    Move exact-duplicate images split across train and validation into train.

    A stratified split ignores image content, so both copies of a duplicated
    image can land in different splits, putting the same image in train and
    validation. For each such group, every copy in validation is moved to
    train and replaced by a training image of the same class, so split sizes
    and per-class counts are unchanged. Replacements are drawn only from
    training images that belong to no duplicate group, so a swap cannot
    create a new split pair. The choice is deterministic given ``seed``.

    Args:
        train_indices (array-like of int): Training-split indices.
        val_indices (array-like of int): Validation-split indices.
        targets (np.ndarray): Labels for the full training set, indexed by
            dataset index.
        dup_groups (list of list of int): Groups of indices whose images are
            exact duplicates, each with more than one index.
        seed (int): Seed for choosing replacements.

    Returns:
        tuple: (train_indices, val_indices) as new sorted numpy arrays. The
        inputs are not modified.
    """
    rng = np.random.default_rng(seed)
    train_idx_set = set(train_indices)
    val_idx_set = set(val_indices)
    splits = _find_split_groups(dup_groups, train_idx_set)
    flattened_dups = {idx for group in dup_groups for idx in group}
    pool = {idx for idx in train_idx_set if idx not in flattened_dups}
    for group in splits:
        for dup in group:
            if dup in val_idx_set:
                label = targets[dup]
                replacement = rng.choice(
                    sorted(idx for idx in pool if targets[idx] == label)
                )
                _swap_dups(train_idx_set, val_idx_set, pool, dup, replacement)
    return np.array(sorted(train_idx_set)), np.array(sorted(val_idx_set))


def build_mix(mixup_alpha, cutmix_alpha, num_classes=100):
    """
    Build the batch-level MixUp/CutMix augmentation for training.

    Each call applies one of the two, chosen at random with equal
    probability. MixUp blends two images pixel by pixel; CutMix pastes a
    rectangle from one image into another. Labels are mixed in the same
    proportion (by blend weight or by patch area), producing soft targets.
    The mixing ratio is drawn from Beta(alpha, alpha). Applied in the
    training loop after batching, since it needs pairs of images; never
    used for validation or test.

    Args:
        mixup_alpha (float): Beta-distribution parameter for MixUp.
        cutmix_alpha (float): Beta-distribution parameter for CutMix.
        num_classes (int): Number of classes, for the soft label vectors.

    Returns:
        v2.RandomChoice: Callable taking (images, labels), with images of
        shape (B, C, H, W) and integer labels of shape (B,), and returning
        the mixed images and soft labels of shape (B, num_classes).
    """
    return v2.RandomChoice([
        v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes),
        v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
    ])
