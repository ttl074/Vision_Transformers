# Vision Transformers on CIFAR-100

Three transformer image classifiers trained from scratch on CIFAR-100 at its native 32 × 32 resolution, with the same data split, augmentation, optimizer, schedule, and training budget:

| Model | Description | Trainable parameters | Test top-1 | Macro F1 |
|---|---|---|---|---|
| Hybrid (primary) | ResNet-56 tokenizer + 7 transformer blocks | 4,604,084 | 76.58% | 0.7643 |
| Plain ViT (baseline) | 4 × 4 patch embedding + 9 transformer blocks | 4,799,076 | 71.52% | 0.7135 |
| Swin | Scaled-down Swin Transformer, 4 × 4 shifted windows | 4,247,064 | 70.74% | 0.7050 |

The full write-up is in `experiments.pdf`.

## Repository layout

```
src/            model, data, training, evaluation, and plotting code
  models.py     plain ViT and ResNet-Transformer hybrid
  swin.py       Swin Transformer
  build.py      builds a model from its run settings
  data.py       train/validation split, augmentation, MixUp/CutMix
  train.py      training loop
  evaluate.py   test-set evaluation and error analysis
  plot_curves.py training-curve and learning-rate figures
scripts/        train.sh and evaluate.sh
tests/          pytest tests for the data split and models
logs/           training logs, test metrics, and test predictions
figures/        all figures in the report
configs/        run configurations
stats.ipynb     duplicate check, parameter counts, McNemar tests
```

## Environment setup

The experiments were run on Google Colab with Python 3.13 and PyTorch 2.11.0 (CUDA 13.0 build). Run every command from the repository root.

On Colab, mount Google Drive and move into the repository:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/<path to>/Vision_Transformers
```

Locally, create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Dependency installation

All dependencies are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If your machine needs a specific CUDA build of PyTorch, install `torch==2.11.0` and `torchvision==0.26.0` for your CUDA version from pytorch.org first.

Training logs to Weights & Biases by default. Either set your API key (`wandb login`, or the `WANDB_API_KEY` environment variable) or add `--no-wandb` to any training command. The CSV training logs in `logs/` are written either way.

## Dataset download

CIFAR-100 is downloaded automatically by torchvision the first time training, evaluation, or the tests run. It is saved to `data/` (about 169 MB). To use a different folder, set `DATA_DIR`:

```bash
DATA_DIR=/content/data bash scripts/train.sh hybrid
```

If the download is slow, download `cifar-100-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar.html yourself and put it in `data/`. torchvision checks the file's MD5 checksum, skips the download, and extracts it.

The 50,000 training images are split into 45,000 training and 5,000 validation images with a stratified split (seed 2026). The split is built the same way every run, so all models use identical data.

## Training

`scripts/train.sh` sets the architecture, depth, run name, and seed for each model. All other settings are the defaults in `src/train.py` (batch size 128, AdamW with learning rate 5e-4 and weight decay 0.05, 10 warmup epochs then cosine decay, up to 200 epochs with early stopping after 30 epochs without validation-loss improvement, label smoothing 0.1, drop path 0.1, MixUp/CutMix, and mixed precision).

Train the primary model (ResNet-Transformer hybrid):

```bash
bash scripts/train.sh hybrid
```

Train the plain ViT baseline:

```bash
bash scripts/train.sh vit
```

Train the Swin Transformer:

```bash
bash scripts/train.sh swin
```

`bash scripts/train.sh both` trains the hybrid and then the plain ViT. Any arguments after the model name are passed to `src.train`, for example a short test run:

```bash
bash scripts/train.sh hybrid --epochs 2 --warmup_epochs 1 --run_name hybrid_smoke --no-wandb
```

Checkpoints are saved to `checkpoints/<run name>/` (set `CKPT_DIR` to change this): `best.pt` has the weights with the lowest validation loss, and `last.pt` is a full checkpoint for resuming. To resume an interrupted run:

```bash
bash scripts/train.sh hybrid --resume checkpoints/hybrid_seed2026/last.pt
```

Checkpoints are not included in this submission because of their size, so evaluation requires training the models first.

## Evaluating a saved checkpoint

Evaluate one model's `best.pt` on the 10,000-image test set:

```bash
bash scripts/evaluate.sh hybrid    # or vit, swin
bash scripts/evaluate.sh all       # all three
```

or evaluate any checkpoint directly:

```bash
python -m src.evaluate --ckpt checkpoints/hybrid_seed2026/best.pt
```

The model architecture is rebuilt from the settings stored in the checkpoint, so no model arguments are needed.

## Reproducing the figures and metrics

1. Train the three models (see Training).
2. Evaluate them: `bash scripts/evaluate.sh all`. This writes the test metrics, per-class metrics, predictions, confusion matrices, and misclassified-image grids.
3. Plot the training curves and learning-rate schedule: `python -m src.plot_curves`. This reads the CSV training logs in `logs/`.
4. Run `stats.ipynb` for the duplicate-image check, the parameter counts, and the McNemar tests between models. It reads the predictions saved in step 2.

Steps 3 and 4 only need the files already in `logs/`, so they can be rerun without retraining.

Run the tests and linter with:

```bash
pytest
pylint src tests
```

## Where logs and outputs are saved

| Output | Location |
|---|---|
| Checkpoints (`best.pt`, `last.pt`) | `checkpoints/<run name>/` |
| Per-epoch training log | `logs/<run name>_training_log.csv` |
| Test metrics summary | `logs/<run name>_metrics.json` |
| Per-class precision, recall, F1 | `logs/<run name>_per_class.csv` |
| Test labels, predictions, probabilities | `logs/<run name>_test_preds.npz` |
| Confusion matrices | `figures/<run name>_confusion.png`, `figures/<run name>_confusion_errors.png` |
| Most confident misclassifications | `figures/<run name>_misclassified.png` |
| Training curves, learning-rate schedule | `figures/training_curves.png`, `figures/lr_schedule.png` |
| Online training curves | Weights & Biases project `Image_Classification_Transformers` |

The run names are `hybrid_seed2026`, `vit_seed2026`, and `swin_seed2026`.

## Hardware used

All three models were trained and evaluated on one NVIDIA L4 GPU on Google Colab, with bf16 mixed precision. On GPUs without bf16 support, `src/train.py` uses fp16 with gradient scaling instead.

## Approximate training time

On one L4, from the epoch times in the training logs (training plus validation each epoch):

| Model | Epochs | Time per epoch | Total |
|---|---|---|---|
| Hybrid | 195 (early stopping) | 23.6 s | 1.3 h |
| Plain ViT | 200 | 14.4 s | 0.8 h |
| Swin | 200 | 17.0 s | 0.9 h |

## Known limitations

- Each model was trained with one random seed, so run-to-run variation is unknown. This matters most for the plain ViT and the Swin, which differ by only 0.78 points.
- The hybrid's ResNet-56 adds depth as well as convolutional inductive bias, so the experiments cannot separate the two effects.
- All models share one set of hyperparameters, which may not be the best for each architecture, especially the heavily scaled-down Swin. The Swin's last stage is a single 4 × 4 window, so it uses global attention.
- Accuracy on clean (unaugmented) training images was not logged, so fit to the training data is judged from the loss and accuracy curves.
- All models were still improving slightly when the learning rate reached zero, so a longer schedule could change the gaps between them.
- Compute is measured with time per epoch and attention memory, not FLOPs, and the epoch times include data loading.
- The CIFAR-100 training set has 14 pairs of exact duplicate images, 9 of them with conflicting labels. All pairs are kept in the training split. Ten test images are exact duplicates of training images and are kept so the test set stays standard. Near-duplicates are not checked.
- GPU results are not bit-for-bit reproducible, because some CUDA operations are nondeterministic.
- Results are for CIFAR-100 at 32 × 32 and may not hold for larger images or datasets.
