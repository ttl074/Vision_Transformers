#!/usr/bin/env bash
# Train the plain ViT baseline, the ResNet-Transformer hybrid, and/or the Swin
# Transformer on CIFAR-100.
#
# All models share every training setting (data split, augmentation,
# optimizer, schedule, duration, batch size, and seed), as the assignment's
# experimental controls require. Only the architecture differs; sizes are
# matched to within 10% of the hybrid's parameter count. Run from the
# repository root:
#
#   bash scripts/train.sh vit       # plain ViT baseline (depth 9)
#   bash scripts/train.sh hybrid    # ResNet-56 + Transformer hybrid (depth 7)
#   bash scripts/train.sh swin      # Swin Transformer (SwinConfig defaults)
#   bash scripts/train.sh both      # hybrid, then ViT
#
# Arguments after the model name are passed to src.train and override the
# defaults below, e.g. a short smoke run with its own name:
#   bash scripts/train.sh swin --epochs 2 --warmup_epochs 1 \
#       --run_name swin_smoke --no-wandb
#
# Paths default to folders inside the repository. Override them with
# environment variables, e.g. on Colab:
#   DATA_DIR=/content/data CKPT_DIR=/content/drive/MyDrive/vit_ckpts \
#       bash scripts/train.sh both
set -euo pipefail

usage() {
    echo "Usage: bash scripts/train.sh {vit|hybrid|swin|both} [src.train arguments]" >&2
    exit 1
}

[[ $# -ge 1 ]] || usage
MODEL="$1"
shift

DATA_DIR="${DATA_DIR:-data}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
SEED="${SEED:-2026}"  # matches DEFAULT_SEED in src/data.py

COMMON_ARGS=(--data_dir "$DATA_DIR" --ckpt_dir "$CKPT_DIR" --run_seed "$SEED")

train_vit() {
    python -m src.train --arch vit --tokenizer patch --depth 9 \
        --run_name "vit_seed${SEED}" "${COMMON_ARGS[@]}" "$@"
}

train_hybrid() {
    python -m src.train --arch vit --tokenizer resnet --depth 7 \
        --run_name "hybrid_seed${SEED}" "${COMMON_ARGS[@]}" "$@"
}

train_swin() {
    python -m src.train --arch swin \
        --run_name "swin_seed${SEED}" "${COMMON_ARGS[@]}" "$@"
}

case "$MODEL" in
    vit) train_vit "$@" ;;
    hybrid) train_hybrid "$@" ;;
    swin) train_swin "$@" ;;
    both) train_hybrid "$@"; train_vit "$@" ;;
    *) usage ;;
esac