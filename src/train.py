"""
Train a ViT or hybrid ViT on CIFAR-100 from scratch.

Builds the stratified train/validation split and the test set, constructs the
model selected by --tokenizer ("patch" for the plain ViT, "resnet" for the
hybrid), and trains it with AdamW (weight decay on weights only), linear
warmup then cosine decay stepped per batch, MixUp/CutMix, label smoothing,
stochastic depth, gradient clipping, and optional mixed precision. Each epoch
logs train and validation metrics to Weights & Biases, saves the weights with
lowest validation loss to best.pt and a full resume checkpoint to last.pt, and
stops early after --early_stop_patience epochs without validation-loss improvement.
An interrupted run continues from last.pt with --resume. The test set is
evaluated separately, from best.pt.

Run from the repository root:
    python -m src.train [options]
"""

import argparse
import csv
from dataclasses import dataclass, asdict
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch import nn
import wandb
from tqdm import tqdm

from src.data import build_datasets, DEFAULT_SEED, build_mix
from src.models import ViT, ViTConfig


WANDB_PROJECT = "Image_Classification_Transformers"
LOG_DIR = "logs"


def parse_args():
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Data, model, optimization, logging, and
        checkpoint settings. Run with --help for each argument's description.
    """
    parser = argparse.ArgumentParser(description='Image Classification with Transformers')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory where CIFAR-100 is stored')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Image batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Number of epochs with no val-loss improvement before early stop')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Peak learning rate reached after warmup for model training')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of epochs of linear learning-rate warmup')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Amount of label smoothing to prevent overconfidence')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of parallel subprocesses for data loading')
    parser.add_argument('--run_seed', type=int, default=DEFAULT_SEED,
                        help='Seed for random and torch for reproducible output')
    parser.add_argument('--tokenizer', choices=['patch', 'resnet'], default='patch',
                        help='Whether to use patches (ViT) or ResNet tokenizer (Hybrid)')
    parser.add_argument('--depth', type=int, default=7,
                        help='Number of transformer encoder blocks')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Stochastic depth drop rate for transformer blocks')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                        help='MixUp Beta-distribution alpha')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix Beta-distribution alpha')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help='Use automatic mixed precision for training (bf16 on GPUs that '
                        'support it natively, fp16 with gradient scaling otherwise)')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Log the run to Weights & Biases (disable with --no-wandb)')
    parser.add_argument('--run_name', default=None, help='Name of the run to log in W&B')
    parser.add_argument('--resume', default=None, help='Path to last.pt for resuming training')
    parser.add_argument('--ckpt_dir', default="checkpoints",
                        help='Directory to store last.pt and best.pt')

    return parser.parse_args()


def set_seed(run_seed):
    """
    Seed Python's, NumPy's, and PyTorch's random number generators.

    Covers weight initialization, DataLoader shuffling, and data
    augmentation. The train/validation split uses its own fixed seed, so
    changing ``run_seed`` changes only training randomness. GPU runs with the
    same seed can still differ slightly, since some CUDA operations are not
    deterministic.

    Args:
        run_seed (int): Seed for this training run.
    """
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)


def init_wandb(args, run_id=None):
    """
    Start a Weights & Biases run, or a disabled no-op run with --no-wandb.

    Records all arguments as the run config and plots every metric against
    the logged "epoch" value. If ``run_id`` is given, the existing run is
    resumed, and initialization fails if it does not exist.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        run_id (str, optional): ID of an existing W&B run to continue when
            resuming; None starts a new run.
    """
    wandb.init(project=WANDB_PROJECT, name=args.run_name, config=vars(args),
               mode="online" if args.wandb else "disabled", id=run_id,
               resume="must" if run_id else None)
    wandb.define_metric("*", step_metric="epoch")


def setup_amp(args, device):
    """
    Choose the mixed-precision settings for training from --amp and the GPU's capabilities.

    AMP is off when --amp is not given or when training on CPU. On GPUs with native bf16
    support (compute capability 8.0 or higher, e.g. A100), uses bf16, which has the same
    exponent range as fp32, so gradients do not underflow and no loss scaling is needed.
    On older GPUs (e.g. T4), uses fp16 with an enabled GradScaler to keep small gradients
    from underflowing to zero.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (uses --amp).
        device (torch.device): Device on which to train.

    Returns:
        tuple: (amp_dtype, scaler):
            - amp_dtype (torch.dtype or None): torch.bfloat16 or torch.float16 for autocast,
              or None when AMP is off.
            - scaler (torch.amp.GradScaler): Enabled only for fp16; when disabled, its
              scale/step/update calls pass through unchanged, so the training loop needs
              no AMP-specific branches.
    """
    if not args.amp or device.type != "cuda":
        return None, torch.amp.GradScaler("cuda", enabled=False)
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    return torch.float16, torch.amp.GradScaler("cuda")


def restore_args(args, ckpt):
    """
    Rebuild the run's arguments from a checkpoint for resuming training.

    Precedence, lowest to highest: the current command's arguments (so flags added
    after the checkpoint was saved still get their defaults), then the saved run's
    arguments from ckpt["run_info"] (so model, data, and training settings match the
    original run), then the session-specific values resume, num_workers, and wandb from
    the current command (which depend on the machine and session, not the experiment).
    The bookkeeping key wandb_id is dropped. ckpt is not modified.

    Args:
        args (argparse.Namespace): Arguments parsed from the current command.
        ckpt (dict): Checkpoint loaded from last.pt, written by save_checkpoint.

    Returns:
        argparse.Namespace: Arguments for the resumed run.
    """
    args_dict = dict(vars(args))
    args_dict.update(ckpt["run_info"])
    args_dict.pop("wandb_id", None)
    for attr in ("resume", "num_workers", "wandb"):
        args_dict[attr] = getattr(args, attr)
    return argparse.Namespace(**args_dict)


@dataclass
class TrainComponents:
    """
    Objects used to optimize the model and augment training batches.

    Attributes:
        criterion (callable): Training loss function (mean reduction,
            with label smoothing).
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scaler (torch.amp.GradScaler): Loss scaler for fp16 AMP; disabled
            (pass-through) for bf16 or no AMP.
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning-rate scheduler stepped
            after every optimizer step (linear warmup, then cosine decay).
        mix (callable): MixUp/CutMix batch augmentation, applied to training
            batches only; returns mixed images and soft labels.
    """
    criterion: object
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    scheduler: object
    mix: object


@dataclass
class TrainState:
    """
    Training progress, saved in last.pt so a resumed run continues exactly.

    Defaults are a fresh run's starting values.

    Attributes:
        epoch (int): Last completed epoch (0 before training starts).
        best_val_loss (float): Lowest validation loss so far.
        best_epoch (int): Epoch with the lowest validation loss (0 before
            training starts).
        epochs_no_improvement (int): Consecutive epochs since the validation
            loss last improved, for early stopping.
    """
    epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_no_improvement: int = 0


@dataclass
class TrainSettings:
    """
    Fixed settings for a training run.

    Attributes:
        device (torch.device): Device on which to train and evaluate.
        run_dir (str): Directory for this run's best.pt and last.pt.
        epochs (int): Maximum number of training epochs.
        early_stop_patience (int): Epochs without validation-loss improvement
            before stopping.
        run_info (dict): Run arguments plus the W&B run ID, saved in both
            checkpoints.
        amp_dtype (torch.dtype or None): Autocast dtype, or None when AMP is off.
        eval_criterion (nn.Module): Loss for evaluation: plain cross-entropy,
            without label smoothing, so validation loss is comparable across runs.
    """
    device: torch.device
    run_dir: str
    epochs: int
    early_stop_patience: int
    run_info: dict
    amp_dtype: torch.dtype
    eval_criterion: nn.Module


def build_loaders(train_set, val_set, test_set, batch_size, num_workers):
    """
    Wrap the three datasets in DataLoaders.

    The training loader shuffles and drops the last partial batch; the
    validation and test loaders keep every image in a fixed order, so
    evaluation is identical each time.

    Args:
        train_set (Dataset): Training split.
        val_set (Dataset): Validation split.
        test_set (Dataset): Test set.
        batch_size (int): Images per batch.
        num_workers (int): DataLoader worker processes.

    Returns:
        tuple: (train_loader, val_loader, test_loader).
    """
    shared = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0
    }
    train_loader = DataLoader(train_set, **shared, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, **shared, shuffle=False)
    test_loader = DataLoader(test_set, **shared, shuffle=False)
    return train_loader, val_loader, test_loader


def build_optimizer(model, args):
    """
    Build AdamW with weight decay applied to weights only.

    Parameters with fewer than two dimensions (biases and normalization
    weights) and the position embedding are excluded from weight decay, as in
    standard ViT training recipes.

    Args:
        model (nn.Module): The model to optimize.
        args (argparse.Namespace): Uses learning_rate and weight_decay.

    Returns:
        torch.optim.AdamW: Optimizer with a decay group and a no-decay group.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.ndim < 2 or name == "pos_embed":
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
    return optimizer


def build_scheduler(optimizer, args, steps_per_epoch):
    """
    Build a linear-warmup, then cosine-decay learning-rate schedule.

    The learning rate rises linearly from 0.1% of its peak to the peak over
    the warmup epochs, then decays to zero along a half-cosine over the
    remaining epochs. All lengths are in optimizer steps, so the scheduler is
    stepped once per batch.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.
        args (argparse.Namespace): Uses epochs and warmup_epochs.
        steps_per_epoch (int): Training batches per epoch.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: The combined schedule.
    """
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    chain = SequentialLR(optimizer, schedulers=[warmup, cosine],
                         milestones=[warmup_steps])
    return chain


def build_components(model, args, device, steps_per_epoch):
    """
    Build the objects used to optimize the model and augment training batches.

    Creates the mixed-precision settings, the AdamW optimizer (with weight
    decay on weights only), the warmup-then-cosine learning-rate schedule,
    the label-smoothed training loss, and the MixUp/CutMix batch
    augmentation.

    Args:
        model (nn.Module): The model to optimize.
        args (argparse.Namespace): Parsed command-line arguments (uses amp,
            learning_rate, weight_decay, epochs, warmup_epochs,
            label_smoothing, mixup_alpha, and cutmix_alpha).
        device (torch.device): Device on which to train.
        steps_per_epoch (int): Training batches per epoch, for the schedule.

    Returns:
        tuple: (components, amp_dtype):
            - components (TrainComponents): Loss, optimizer, scaler,
              scheduler, and MixUp/CutMix augmentation.
            - amp_dtype (torch.dtype or None): Autocast dtype, or None when
              AMP is off.
    """
    amp_dtype, scaler = setup_amp(args, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mix = build_mix(args.mixup_alpha, args.cutmix_alpha)
    components = TrainComponents(criterion=criterion, optimizer=optimizer,
                                 scaler=scaler, scheduler=scheduler, mix=mix)
    return components, amp_dtype


def train_one_epoch(model, dataloader, components, device, amp_dtype):
    """
    Train the model for one pass over the training set.

    For each batch: MixUp or CutMix, forward pass and loss against the soft
    labels under autocast, backward pass through the gradient scaler,
    gradient-norm clipping at 1.0, an optimizer step, and a scheduler step.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training batches.
        components (TrainComponents): Loss, optimizer, scaler, scheduler,
            and MixUp/CutMix augmentation.
        device (torch.device): Device on which to train.
        amp_dtype (torch.dtype or None): Autocast dtype, or None when AMP is off.

    Returns:
        tuple: (mean loss, accuracy) over the epoch. Accuracy compares
        predictions on the mixed images with their original labels, so it is
        an approximate measure that runs below validation accuracy.
    """
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mixed_images, soft_labels = components.mix(images, labels)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(mixed_images)
            loss = components.criterion(logits, soft_labels)
        components.optimizer.zero_grad()
        components.scaler.scale(loss).backward()
        components.scaler.unscale_(components.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        components.scaler.step(components.optimizer)
        components.scaler.update()
        components.scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


def _atomic_save(obj, path):
    """
    Save an object to ``path`` so the file is never left half-written.

    Writes to a temporary file, then replaces ``path`` with it in one step,
    so an interruption during saving leaves the previous file intact.

    Args:
        obj: Object to save with torch.save.
        path (str): Destination file path.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def append_log(path, row):
    """
    Append one epoch's metrics as a row to a CSV log file.

    Writes the header (the row's keys) only when the file does not exist yet,
    so a resumed run keeps adding to the same file.

    Args:
        path (str): CSV file to append to.
        row (dict): Metric names mapped to values for one epoch.
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(model, loaders, components, state, settings):
    """
    Run the training loop from the epoch after ``state.epoch``.

    Each epoch trains, evaluates on the validation set, logs metrics to W&B
    and to logs/<run name>_training_log.csv, saves best.pt when validation
    loss improves (recording the best epoch in ``state``), and saves last.pt.
    Stops early after ``settings.early_stop_patience`` consecutive epochs
    without improvement.

    Args:
        model (nn.Module): The model to train.
        loaders (tuple): (train_loader, val_loader).
        components (TrainComponents): Loss, optimizer, scaler, scheduler,
            and MixUp/CutMix augmentation.
        state (TrainState): Progress to start from; updated in place.
        settings (TrainSettings): Fixed settings for the run.

    Returns:
        TrainState: Progress when training ended.
    """
    train_loader, val_loader = loaders
    for epoch in range(state.epoch + 1, settings.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, components,
                                                settings.device, settings.amp_dtype)
        val_loss, val_acc = evaluate(model, val_loader, settings.eval_criterion,
                                     device=settings.device, amp_dtype=settings.amp_dtype)
        metrics = {"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc,
                   "val/loss": val_loss, "val/acc": val_acc,
                   "lr": components.optimizer.param_groups[0]["lr"],
                   "epoch_time_s": time.perf_counter() - start
        }
        wandb.log(metrics)
        print(f"epoch {epoch}: train {train_acc:.3f} / val {val_acc:.3f}, val loss {val_loss:.4f}")
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.best_epoch = epoch
            state.epochs_no_improvement = 0
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "run_info": settings.run_info
            }
            _atomic_save(checkpoint, os.path.join(settings.run_dir, "best.pt"))
        else:
            state.epochs_no_improvement += 1
        state.epoch = epoch
        save_checkpoint(os.path.join(settings.run_dir, "last.pt"), model, components,
                        state, settings.run_info)
        append_log(os.path.join(LOG_DIR, f"{os.path.basename(settings.run_dir)}"
                                        "_training_log.csv"), metrics)
        if state.epochs_no_improvement >= settings.early_stop_patience:
            print(f"Early stopping after {state.epochs_no_improvement} epochs with no improvement")
            break
    return state


def evaluate(model, dataloader, criterion, device, amp_dtype):
    """
    Compute the model's mean loss and accuracy on a dataset, without training.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Evaluation batches.
        criterion (nn.Module): Loss function.
        device (torch.device): Device on which to evaluate.
        amp_dtype (torch.dtype or None): Autocast dtype, or None when AMP is off.

    Returns:
        tuple: (mean loss, accuracy).
    """
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def save_checkpoint(path, model, components, state, run_info):
    """
    Save everything needed to resume training exactly.

    Stores the model, optimizer, scheduler, and scaler states, the training
    progress, and the run information, written atomically.

    Args:
        path (str): Destination file, normally last.pt in the run directory.
        model (nn.Module): The model being trained.
        components (TrainComponents): Optimizer, scheduler, and scaler to save.
        state (TrainState): Current training progress.
        run_info (dict): Run arguments plus the W&B run ID.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": components.optimizer.state_dict(),
        "scheduler": components.scheduler.state_dict(),
        "scaler": components.scaler.state_dict(),
        "train_state": asdict(state),
        "run_info": run_info
    }
    _atomic_save(checkpoint, path)


def load_checkpoint(ckpt, model, components):
    """
    Restore model and optimization state from a checkpoint saved by save_checkpoint.

    The model, optimizer, scheduler, and scaler must already be built with the
    original run's settings; their states are loaded in place.

    Args:
        ckpt (dict): Checkpoint loaded from last.pt.
        model (nn.Module): Model to load weights into.
        components (TrainComponents): Optimizer, scheduler, and scaler to restore.

    Returns:
        TrainState: The saved training progress.
    """
    model.load_state_dict(ckpt["model"])
    components.optimizer.load_state_dict(ckpt["optimizer"])
    components.scheduler.load_state_dict(ckpt["scheduler"])
    components.scaler.load_state_dict(ckpt["scaler"])
    return TrainState(**ckpt["train_state"])


def main(args):
    """
    Train a model from the command-line arguments, resuming if --resume is given.

    When resuming, the original run's arguments are restored from the
    checkpoint before seeding, so the model, data, and schedule match the
    interrupted run, and the original W&B run is continued. Checkpoints go to
    ckpt_dir/<run name or W&B run ID>.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = None
    run_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        args = restore_args(args, ckpt)
        run_id = ckpt["run_info"]["wandb_id"]
    set_seed(args.run_seed)
    datasets = build_datasets(args.data_dir, DEFAULT_SEED)
    dataloaders = build_loaders(datasets[0], datasets[1], datasets[2],
                                args.batch_size, args.num_workers)
    model = ViT(ViTConfig(tokenizer=args.tokenizer, depth=args.depth,
                          drop_path_rate=args.drop_path_rate)).to(device)
    init_wandb(args, run_id)
    components, amp_dtype = build_components(model, args, device, len(dataloaders[0]))
    state = load_checkpoint(ckpt, model, components) if ckpt else TrainState()
    run_dir = os.path.join(args.ckpt_dir, args.run_name or wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    run_info = {**vars(args), "wandb_id": wandb.run.id}
    eval_criterion = nn.CrossEntropyLoss()
    settings = TrainSettings(device=device, run_dir=run_dir, epochs=args.epochs,
                             early_stop_patience=args.early_stop_patience,
                             run_info=run_info, amp_dtype=amp_dtype,
                             eval_criterion=eval_criterion)
    state = train(model, (dataloaders[0], dataloaders[1]), components, state, settings)
    wandb.run.summary["best_val_loss"] = state.best_val_loss
    wandb.run.summary["stopped_epoch"] = state.epoch
    wandb.run.summary["best_epoch"] = state.best_epoch
    print(f"Finished at epoch {state.epoch}, best val loss {state.best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main(parse_args())
