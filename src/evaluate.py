"""
Evaluate a trained ViT or hybrid ViT on the CIFAR-100 test set.

Loads a best.pt checkpoint written by src.train, rebuilds the model from the
run settings stored in it, and evaluates the official 10,000-image test set
once, using the same normalization as training. Computes top-1 accuracy,
macro precision, recall, and F1, weighted F1, per-class precision, recall,
and F1, the confusion matrix, and one-vs-rest macro ROC-AUC, plus the error
analysis: the five best and worst classes by F1 and the most frequent
confusions. Writes the predictions (.npz), a summary (.json), and per-class
metrics (.csv) to --log_dir, and the confusion matrices and a grid of the
most confident misclassifications to --fig_dir. Output files are prefixed
with the run name, taken from the checkpoint's directory.

Run from the repository root:
    python -m src.evaluate --ckpt checkpoints/<run name>/best.pt
"""

import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from src.build import build_model
from src.data import build_datasets, DEFAULT_SEED


def load_model(ckpt_path, device):
    """
    Rebuild a trained model from a best.pt checkpoint, ready for inference.

    The tokenizer and depth come from the run settings saved in the
    checkpoint, so the architecture matches the one that was trained.

    Args:
        ckpt_path (str): Path to best.pt.
        device (torch.device): Device to load the model onto.

    Returns:
        tuple: (model, epoch, val_loss):
            - model (ViT): The trained model, in eval mode.
            - epoch (int): Epoch at which the checkpoint was saved.
            - val_loss (float): Validation loss at that epoch.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["run_info"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["epoch"], ckpt["val_loss"]


def predict(model, loader, device):
    """
    Run the model over a dataset and collect its predictions.

    Runs in full precision without autocast, and computes the softmax in
    float32, so each row of probabilities sums to 1 closely enough for
    sklearn's multiclass ROC-AUC check.

    Args:
        model (nn.Module): Model in eval mode.
        loader (DataLoader): Evaluation batches, unshuffled, so position i of
            each output corresponds to image i of the dataset.
        device (torch.device): Device on which to run the model.

    Returns:
        tuple: (labels, preds, probs) as NumPy arrays of shapes (N,), (N,),
        and (N, num_classes).
    """
    with torch.no_grad():
        all_probs, all_labels = [], []
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = logits.float().softmax(dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = probs.argmax(axis=1)
    return labels, preds, probs


def compute_metrics(labels, preds, probs):
    """
    Compute the classification metrics for a set of predictions.

    Macro averages are unweighted means over classes; the weighted F1 weights
    each class by its number of true images. Classes that are never
    predicted get a precision of 0.

    Args:
        labels (np.ndarray): True class of each image, shape (N,).
        preds (np.ndarray): Predicted class of each image, shape (N,).
        probs (np.ndarray): Predicted class probabilities, shape
            (N, num_classes).

    Returns:
        dict: With keys:
            - "scalars" (dict): Top-1 accuracy, macro precision, recall, and
              F1, weighted F1, and one-vs-rest macro ROC-AUC.
            - "per_class" (dict): Precision, recall, F1, and support arrays
              of length num_classes, indexed by class.
            - "confusion" (np.ndarray): Confusion matrix of counts, rows true
              class, columns predicted class.
    """
    precision, recall, fscore, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0)
    auc_score = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return {
        "scalars": {
            "top_one_acc": (preds == labels).mean(),
            "macro_precision": precision.mean(),
            "macro_recall": recall.mean(),
            "macro_f1": fscore.mean(),
            "weighted_f1": np.average(fscore, weights=support),
            "roc_auc_ovr_macro": auc_score
        },
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": fscore,
            "support": support
        },
        "confusion": confusion_matrix(labels, preds)
    }


def top_bottom_classes(fscore, class_names, k=5):
    """
    Find the classes with the highest and lowest F1 scores.

    Args:
        fscore (np.ndarray): Per-class F1, indexed by class.
        class_names (list[str]): Class names, indexed by class.
        k (int): Number of classes to return at each end.

    Returns:
        tuple: (top, bottom), each a list of k (class name, F1) pairs: top
        from best to worse, bottom from worst to better.
    """
    order = np.argsort(fscore)
    top = order[-k:][::-1]
    bottom = order[:k]
    top_names = [(class_names[i], fscore[i]) for i in top]
    bot_names = [(class_names[i], fscore[i]) for i in bottom]
    return top_names, bot_names


def top_confusion(cm, class_names, k=10):
    """
    Find the most frequent misclassifications in a confusion matrix.

    Correct predictions (the diagonal) are excluded, so only errors are
    ranked. Each (true, predicted) direction is counted separately.

    Args:
        cm (np.ndarray): Confusion matrix of counts, rows true class,
            columns predicted class. Not modified.
        class_names (list[str]): Class names, indexed by class.
        k (int): Number of confusions to return.

    Returns:
        list[tuple]: k (true class name, predicted class name, count)
        tuples, most frequent first.
    """
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat = np.argsort(cm_off, axis=None)
    most_confused = flat[-k:][::-1]
    rows, cols = np.divmod(most_confused, cm_off.shape[1])
    return [(class_names[r], class_names[c], int(cm_off[r, c])) for r, c in zip(rows, cols)]


def most_confident_errors(labels, preds, probs, k=12):
    """
    Find the misclassified images the model was most confident about.

    Confidence is the probability the model gave its (wrong) prediction.

    Args:
        labels (np.ndarray): True class of each image, shape (N,).
        preds (np.ndarray): Predicted class of each image, shape (N,).
        probs (np.ndarray): Predicted class probabilities, shape
            (N, num_classes).
        k (int): Number of images to return.

    Returns:
        np.ndarray: Dataset indices of the k most confident errors, most
        confident first.
    """
    mistakes = np.flatnonzero(labels != preds)
    conf = probs.max(axis=1)[mistakes]
    ordered = np.argsort(conf)[::-1][:k]
    return mistakes[ordered]


def save_results(prefix, summary, predictions, per_class, class_names):
    """
    Write the predictions, summary, and per-class metrics to disk.

    Creates <prefix>_test_preds.npz (arrays labels, preds, and probs),
    <prefix>_metrics.json (the summary), and <prefix>_per_class.csv (one row
    per class). The .npz holds the single test evaluation, so figures and
    tables can be remade without rerunning the model.

    Args:
        prefix (str): Path prefix for the output files, including the
            directory and run name.
        summary (dict): JSON-serializable summary of the evaluation.
        predictions (tuple): (labels, preds, probs) from predict.
        per_class (dict): Per-class metric arrays from compute_metrics.
        class_names (list[str]): Class names, used as the CSV row labels.
    """
    labels, preds, probs = predictions
    np.savez(f"{prefix}_test_preds.npz", labels=labels, preds=preds, probs=probs)
    with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(per_class, index=class_names).to_csv(f"{prefix}_per_class.csv")


def plot_confusion_matrix(cm, path, title, zero_diagonal=False):
    """
    Save a heatmap of the row-normalized confusion matrix.

    Each row is divided by its total, so cell (i, j) is the fraction of
    class i's images predicted as class j and the diagonal is per-class
    recall. The full matrix uses a fixed 0-1 color scale, so plots of
    different models are comparable. With zero_diagonal, correct predictions
    are removed and the color scale is fitted to the largest error, so
    frequent confusions stand out.

    Args:
        cm (np.ndarray): Confusion matrix of counts, rows true class,
            columns predicted class. Not modified.
        path (str): Output image path.
        title (str): Plot title.
        zero_diagonal (bool): Plot errors only.
    """
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    if zero_diagonal:
        np.fill_diagonal(cm_norm, 0)
    vmax = cm_norm.max() if zero_diagonal else 1.0
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=vmax)
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    ticks = range(0, cm.shape[0], 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(indices, images, predictions, class_names, path):
    """
    Save a grid of misclassified images with their true and predicted classes.

    Each panel shows the raw image, titled with the true class and the
    probability the model gave it, and the predicted class and its
    probability. The grid has four columns; unused panels are left blank.

    Args:
        indices (np.ndarray): Dataset indices of the images to show.
        images (np.ndarray): Raw images, shape (N, H, W, 3), dtype uint8.
        predictions (tuple): (labels, preds, probs) from predict.
        class_names (list[str]): Class names, indexed by class.
        path (str): Output image path.
    """
    labels, preds, probs = predictions
    ncols = 4
    nrows = math.ceil(len(indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(images[idx], interpolation="nearest")
        ax.set_title(f"true: {class_names[labels[idx]]} ({probs[idx, labels[idx]]:.2f})\n"
                     f"pred: {class_names[preds[idx]]} ({probs[idx, preds[idx]]:.2f})",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    """
    Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Checkpoint path and data and output directories.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on the CIFAR-100 test set')
    parser.add_argument('--ckpt', required=True, help='Path to best.pt')
    parser.add_argument('--data_dir', default='data', help='Directory where CIFAR-100 is stored')
    parser.add_argument('--log_dir', default='logs', help='Directory for metrics and predictions')
    parser.add_argument('--fig_dir', default='figures', help='Directory for figures')
    return parser.parse_args()


def main(args):
    """
    Evaluate a checkpoint on the test set and save its metrics and figures.

    The test set is built with build_datasets and the fixed split seed, so it
    is normalized with the training split's statistics. Output files are
    prefixed with the checkpoint's directory name (the run name).

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch, val_loss = load_model(args.ckpt, device)
    test_set = build_datasets(args.data_dir, DEFAULT_SEED)[2]
    predictions = predict(model, DataLoader(test_set, batch_size=256, shuffle=False), device)
    metrics = compute_metrics(*predictions)
    name = os.path.basename(os.path.dirname(args.ckpt))
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    summary = {
        **metrics["scalars"],
        "best_epoch": epoch,
        "best_val_loss": val_loss,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "top_bottom_f1": dict(zip(("top", "bottom"), top_bottom_classes(
            metrics["per_class"]["f1"], test_set.classes))),
        "top_confusions": top_confusion(metrics["confusion"], test_set.classes)
    }
    save_results(os.path.join(args.log_dir, name), summary,
                 predictions, metrics["per_class"], test_set.classes)
    fig_prefix = os.path.join(args.fig_dir, name)
    plot_confusion_matrix(metrics["confusion"], f"{fig_prefix}_confusion.png",
                          f"{name}: confusion matrix (row-normalized)")
    plot_confusion_matrix(metrics["confusion"], f"{fig_prefix}_confusion_errors.png",
                          f"{name}: errors only (diagonal removed)", zero_diagonal=True)
    plot_misclassified(most_confident_errors(*predictions), test_set.data, predictions,
                       test_set.classes, f"{fig_prefix}_misclassified.png")


if __name__ == "__main__":
    main(parse_args())
