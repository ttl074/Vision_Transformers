#!/usr/bin/env bash
# Train the plain ViT baseline and/or the ResNet-Transformer hybrid on CIFAR-100.
#
# Both models share every setting except the tokenizer and the encoder depth
# (chosen so their parameter counts are within 5% of each other), as the
# assignment's experimental controls require. Run from the repository root:
#
#   bash scripts/train.sh vit       # plain ViT baseline (depth 9)
#   bash scripts/train.sh hybrid    # ResNet-56 + Transformer hybrid (depth 7)
#   bash scripts/train.sh both      # hybrid, then ViT
#
# Arguments after the model name are passed to src.train and override the
# defaults below, e.g. a pilot run with its own name:
#   bash scripts/train.sh hybrid --epochs 40 --warmup_epochs 5 \
#       --learning_rate 1e-3 --run_name hybrid_pilot_lr1e-3
#
# Paths default to folders inside the repository. Override them with
# environment variables, e.g. on Colab:
#   DATA_DIR=/content/data CKPT_DIR=/content/drive/MyDrive/vit_ckpts \
#       bash scripts/train.sh both
set -euo pipefail

usage() {
    echo "Usage: bash scripts/train.sh {vit|hybrid|both} [src.train arguments]" >&2
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
    python -m src.train --tokenizer patch --depth 9 \
        --run_name "vit_seed${SEED}" "${COMMON_ARGS[@]}" "$@"
}

train_hybrid() {
    python -m src.train --tokenizer resnet --depth 7 \
        --run_name "hybrid_seed${SEED}" "${COMMON_ARGS[@]}" "$@"
}

case "$MODEL" in
    vit) train_vit "$@" ;;
    hybrid) train_hybrid "$@" ;;
    both) train_hybrid "$@"; train_vit "$@" ;;
    *) usage ;;
esac
