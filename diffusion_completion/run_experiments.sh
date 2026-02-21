#!/usr/bin/env bash
# ==============================================================
# run_experiments.sh  —  Full LiDAR completion experiment pipeline
#
# Phase 1: Training (12 runs: Conv-zero × 3, Conv-noise × 3, Diffusion × 3, Diffusion-v2 × 3)
# Phase 2: Evaluation with --save_per_sample (12 runs)
# Phase 3: Statistical tests (paired t-test, Wilcoxon, Cohen's d)
#
# Usage:
#   bash run_experiments.sh [options]
#
# Options:
#   --skip-training   Skip Phase 1 (use existing checkpoints)
#   --skip-eval       Skip Phase 2 (use existing eval results)
#   --device DEVICE   Compute device (default: cuda)
#   --epochs N        Training epochs (default: 100)
#
# Notes:
#   - baseline/, diffusion/, and diffusion_v2/ use local imports, so each
#     python command runs inside a subshell that cd's into the respective dir.
#   - Absolute paths are passed via PROJECT_ROOT to avoid path issues.
# ==============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Default parameters ─────────────────────────────────────────────────────
DEVICE="cuda"
EPOCHS=100
SEEDS=(42 43 44)
SKIP_TRAINING=0
SKIP_EVAL=0

TRAIN_DIRS=(
    "${PROJECT_ROOT}/output/train"
    "${PROJECT_ROOT}/output_validation/train"
)
VAL_DIRS=(
    "${PROJECT_ROOT}/output/val"
    "${PROJECT_ROOT}/output_validation/val"
)

# ── Argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-training) SKIP_TRAINING=1; shift ;;
        --skip-eval)     SKIP_EVAL=1;     shift ;;
        --device)        DEVICE="$2";     shift 2 ;;
        --epochs)        EPOCHS="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " LiDAR Completion Experiment Pipeline"
echo "============================================================"
echo " PROJECT_ROOT  : ${PROJECT_ROOT}"
echo " DEVICE        : ${DEVICE}"
echo " EPOCHS        : ${EPOCHS}"
echo " SEEDS         : ${SEEDS[*]}"
echo " SKIP_TRAINING : ${SKIP_TRAINING}"
echo " SKIP_EVAL     : ${SKIP_EVAL}"
echo " Models        : Conv-zero, Conv-noise, Diffusion v1, Diffusion v2"
echo "============================================================"

# ── Phase 1: Training ─────────────────────────────────────────────────────
if [[ ${SKIP_TRAINING} -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo " Phase 1: Training (12 runs)"
    echo "============================================================"

    for SEED in "${SEEDS[@]}"; do

        # Conv(zero)
        EXP_NAME="conv_zero_seed${SEED}"
        echo ""
        echo ">>> Training ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/baseline"
            python train.py \
                --train_dirs "${TRAIN_DIRS[@]}" \
                --val_dirs "${VAL_DIRS[@]}" \
                --experiment_name "${EXP_NAME}" \
                --output_dir "${PROJECT_ROOT}/baseline/experiments" \
                --fill_mode zero \
                --epochs "${EPOCHS}" \
                --batch_size 64 \
                --lr 1e-3 \
                --seed "${SEED}" \
                --device "${DEVICE}"
        )

        # Conv(noise)
        EXP_NAME="conv_noise_seed${SEED}"
        echo ""
        echo ">>> Training ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/baseline"
            python train.py \
                --train_dirs "${TRAIN_DIRS[@]}" \
                --val_dirs "${VAL_DIRS[@]}" \
                --experiment_name "${EXP_NAME}" \
                --output_dir "${PROJECT_ROOT}/baseline/experiments" \
                --fill_mode noise \
                --epochs "${EPOCHS}" \
                --batch_size 64 \
                --lr 1e-3 \
                --seed "${SEED}" \
                --device "${DEVICE}"
        )

        # Diffusion(noise)
        EXP_NAME="diffusion_seed${SEED}"
        echo ""
        echo ">>> Training ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/diffusion"
            python train.py \
                --train_dirs "${TRAIN_DIRS[@]}" \
                --val_dirs "${VAL_DIRS[@]}" \
                --experiment_name "${EXP_NAME}" \
                --output_dir "${PROJECT_ROOT}/diffusion/experiments" \
                --epochs "${EPOCHS}" \
                --batch_size 64 \
                --lr 2e-4 \
                --T 100 \
                --seed "${SEED}" \
                --device "${DEVICE}"
        )

        # Diffusion v2 (Dual Encoder + FiLM)
        EXP_NAME="diffusion_v2_seed${SEED}"
        echo ""
        echo ">>> Training ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/diffusion_v2"
            python train.py \
                --train_dirs "${TRAIN_DIRS[@]}" \
                --val_dirs "${VAL_DIRS[@]}" \
                --experiment_name "${EXP_NAME}" \
                --output_dir "${PROJECT_ROOT}/diffusion_v2/experiments" \
                --epochs "${EPOCHS}" \
                --batch_size 64 \
                --lr 2e-4 \
                --T 100 \
                --base_channels 64 \
                --num_res_blocks 2 \
                --seed "${SEED}" \
                --device "${DEVICE}"
        )

    done
fi  # end Phase 1

# ── Phase 2: Evaluation ────────────────────────────────────────────────────
if [[ ${SKIP_EVAL} -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo " Phase 2: Evaluation (12 runs)"
    echo "============================================================"

    for SEED in "${SEEDS[@]}"; do

        # Conv(zero) evaluation
        EXP_NAME="conv_zero_seed${SEED}"
        CKPT="${PROJECT_ROOT}/baseline/experiments/${EXP_NAME}/checkpoints/best_model.pth"
        OUT_DIR="${PROJECT_ROOT}/results/eval/${EXP_NAME}"
        echo ""
        echo ">>> Evaluating ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/baseline"
            python evaluate.py \
                --checkpoint "${CKPT}" \
                --data_dirs "${VAL_DIRS[@]}" \
                --output_dir "${OUT_DIR}" \
                --fill_mode zero \
                --batch_size 64 \
                --device "${DEVICE}" \
                --save_per_sample
        )

        # Conv(noise) evaluation
        EXP_NAME="conv_noise_seed${SEED}"
        CKPT="${PROJECT_ROOT}/baseline/experiments/${EXP_NAME}/checkpoints/best_model.pth"
        OUT_DIR="${PROJECT_ROOT}/results/eval/${EXP_NAME}"
        echo ""
        echo ">>> Evaluating ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/baseline"
            python evaluate.py \
                --checkpoint "${CKPT}" \
                --data_dirs "${VAL_DIRS[@]}" \
                --output_dir "${OUT_DIR}" \
                --fill_mode noise \
                --batch_size 64 \
                --device "${DEVICE}" \
                --save_per_sample
        )

        # Diffusion evaluation
        EXP_NAME="diffusion_seed${SEED}"
        CKPT="${PROJECT_ROOT}/diffusion/experiments/${EXP_NAME}/checkpoints/best_model.pth"
        OUT_DIR="${PROJECT_ROOT}/results/eval/${EXP_NAME}"
        echo ""
        echo ">>> Evaluating ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/diffusion"
            python sample.py \
                --checkpoint "${CKPT}" \
                --data_dirs "${VAL_DIRS[@]}" \
                --output_dir "${OUT_DIR}" \
                --T 100 \
                --batch_size 32 \
                --device "${DEVICE}" \
                --save_per_sample
        )

        # Diffusion v2 evaluation
        EXP_NAME="diffusion_v2_seed${SEED}"
        CKPT="${PROJECT_ROOT}/diffusion_v2/experiments/${EXP_NAME}/checkpoints/best_model.pth"
        OUT_DIR="${PROJECT_ROOT}/results/eval/${EXP_NAME}"
        echo ""
        echo ">>> Evaluating ${EXP_NAME} ..."
        (
            cd "${PROJECT_ROOT}/diffusion_v2"
            python sample.py \
                --checkpoint "${CKPT}" \
                --data_dirs "${VAL_DIRS[@]}" \
                --output_dir "${OUT_DIR}" \
                --T 100 \
                --base_channels 64 \
                --num_res_blocks 2 \
                --batch_size 32 \
                --device "${DEVICE}" \
                --save_per_sample
        )

    done
fi  # end Phase 2

# ── Phase 3: Statistical Tests ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 3: Statistical Tests"
echo "============================================================"

ZERO_DIRS=()
NOISE_DIRS=()
DIFF_DIRS=()
DIFF_V2_DIRS=()
for SEED in "${SEEDS[@]}"; do
    ZERO_DIRS+=("${PROJECT_ROOT}/results/eval/conv_zero_seed${SEED}")
    NOISE_DIRS+=("${PROJECT_ROOT}/results/eval/conv_noise_seed${SEED}")
    DIFF_DIRS+=("${PROJECT_ROOT}/results/eval/diffusion_seed${SEED}")
    DIFF_V2_DIRS+=("${PROJECT_ROOT}/results/eval/diffusion_v2_seed${SEED}")
done

python "${PROJECT_ROOT}/statistical_test.py" \
    --zero_fill_dirs "${ZERO_DIRS[@]}" \
    --noise_fill_dirs "${NOISE_DIRS[@]}" \
    --diffusion_dirs "${DIFF_DIRS[@]}" \
    --diffusion_v2_dirs "${DIFF_V2_DIRS[@]}" \
    --output_dir "${PROJECT_ROOT}/results/statistical_tests" \
    --seeds "${SEEDS[@]}"

echo ""
echo "============================================================"
echo " All phases completed!"
echo " Results: ${PROJECT_ROOT}/results/statistical_tests/statistical_results.json"
echo " Models compared: Conv-zero, Conv-noise, Diffusion v1, Diffusion v2"
echo "============================================================"
