#!/usr/bin/env bash
# Evaluate trained models on the CIFAR-100 test set.
#
# Each model's best.pt (lowest validation loss) is evaluated once on the
# test set by src.evaluate, which writes metrics and predictions to logs/ and
# figures to figures/, prefixed with the run name. Run names match those
# used by scripts/train.sh. Run from the repository root:
#
#   bash scripts/evaluate.sh vit       # plain ViT baseline
#   bash scripts/evaluate.sh hybrid    # ResNet-Transformer hybrid
#   bash scripts/evaluate.sh swin      # Swin Transformer
#   bash scripts/evaluate.sh all       # all three, one after another
#
# Arguments after the model name are passed to src.evaluate, e.g.:
#   bash scripts/evaluate.sh all --fig_dir report_figures
#
# Paths default to folders inside the repository. Override them with
# environment variables, matching those used for training, e.g. on Colab:
#   CKPT_DIR=/content/drive/MyDrive/vit_ckpts bash scripts/evaluate.sh all
set -euo pipefail

usage() {
    echo "Usage: bash scripts/evaluate.sh {vit|hybrid|swin|all} [src.evaluate arguments]" >&2
    exit 1
}

[[ $# -ge 1 ]] || usage
MODEL="$1"
shift

DATA_DIR="${DATA_DIR:-data}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
SEED="${SEED:-2026}"  # matches the run names from scripts/train.sh

evaluate_run() {
    local ckpt="${CKPT_DIR}/$1_seed${SEED}/best.pt"
    shift
    if [[ ! -f "$ckpt" ]]; then
        echo "No checkpoint at $ckpt" >&2
        exit 1
    fi
    python -m src.evaluate --ckpt "$ckpt" --data_dir "$DATA_DIR" "$@"
}

case "$MODEL" in
    vit|hybrid|swin) evaluate_run "$MODEL" "$@" ;;
    all) for run in hybrid vit swin; do evaluate_run "$run" "$@"; done ;;
    *) usage ;;
esac