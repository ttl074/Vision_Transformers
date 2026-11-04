"""
Tests for the CIFAR-100 dataset splits.

Verifies that the train and validation splits are disjoint and together
partition the official training set, and that exact-duplicate images are
confined to known cases: none shared between train and validation, none
between validation and test, and exactly the 10 train/test duplicates
present in the official CIFAR-100 files.
"""

import numpy as np
from src.data import image_hash_groups

# Exact duplicates shared between the official CIFAR-100 training and test
# files, found by hashing raw image bytes (see the duplicate investigation
# notebook). They come from the source data, not from the split.
KNOWN_TRAIN_TEST_DUPS = 10


def test_dataset_disjoint(train_set, val_set):
    """
    Check that the training and validation splits share no indices.
    """
    train_indices = train_set.indices
    val_indices = val_set.indices
    idx_int = np.intersect1d(train_indices, val_indices)
    index_message = f"Training and Validation are not disjoint: {idx_int[:5]}"
    assert len(idx_int) == 0, index_message


def test_dataset_partition(train_set, val_set, test_set):
    """
    Check split sizes and that train and validation partition the training set.

    Train and validation must contain 45,000 and 5,000 indices, the test set
    must contain 10,000 images, and the union of the train and validation
    indices must be exactly 0-49,999. Together, the sizes and the union also
    imply that no index is repeated within or across the two splits.
    """
    train_indices = train_set.indices
    val_indices = val_set.indices
    train_message = f"Expected 45000 training indices, got {len(train_indices)}"
    val_message = f"Expected 5000 validation indices, got {len(val_indices)}"
    test_message = f"Expected 10000 test images, got {len(test_set)}"
    assert len(train_indices) == 45000, train_message
    assert len(val_indices) == 5000, val_message
    assert len(test_set) == 10000, test_message
    trainval_union = np.union1d(train_indices, val_indices)
    union_message = "Union of train and val split does not cover 0-49999"
    assert np.array_equal(trainval_union, np.arange(50000)), union_message


def test_no_val_overlap(train_set, val_set):
    """
    Check that no exact duplicate image appears in both train and validation.

    The official training file contains exact duplicates, and a stratified
    split can place the two copies of one image in different splits, which
    index-level checks cannot detect. ``resolve_split_dups`` moves such copies
    into train; this test confirms by comparing MD5 hashes of raw image bytes
    that no image content is shared between the two splits. It detects exact
    pixel-for-pixel duplicates only, not near-duplicates (e.g. slightly
    shifted or recolored versions of the same image).
    """
    train_indices = train_set.indices
    val_indices = val_set.indices
    train_imgs = train_set.dataset.data[train_indices]
    val_imgs = val_set.dataset.data[val_indices]
    train_hashes = image_hash_groups(train_imgs).keys()
    val_hashes = image_hash_groups(val_imgs).keys()
    trainval_int = train_hashes & val_hashes
    trainval_message = f"{len(trainval_int)} Train and Validation images overlap"
    assert not trainval_int, trainval_message


def test_no_test_overlap(train_set, val_set, test_set):
    """
    Check exact-duplicate overlap between the training data and the test set.

    The official CIFAR-100 training and test files share
    ``KNOWN_TRAIN_TEST_DUPS`` exact-duplicate images. The test set must stay
    unchanged and the training split must keep 45,000 images, so these remain
    in train; the test fails if the count changes in either direction, which
    would mean the split itself introduced or removed overlap. Validation
    must share no images with the test set. Compares MD5 hashes of raw image
    bytes, so only exact duplicates are detected, not near-duplicates.
    """
    train_indices = train_set.indices
    val_indices = val_set.indices
    train_imgs = train_set.dataset.data[train_indices]
    train_hashes = image_hash_groups(train_imgs).keys()
    val_imgs = val_set.dataset.data[val_indices]
    val_hashes = image_hash_groups(val_imgs).keys()
    test_hashes = image_hash_groups(test_set.data).keys()
    traintest_int = train_hashes & test_hashes
    valtest_int = val_hashes & test_hashes
    trtest_message = (
        f"Train & Test overlap: {len(traintest_int)}, expected {KNOWN_TRAIN_TEST_DUPS}"
    )
    assert len(traintest_int) == KNOWN_TRAIN_TEST_DUPS, trtest_message
    assert len(valtest_int) == 0, (
        f"Validation and Test images overlap: {len(valtest_int)}"
    )
