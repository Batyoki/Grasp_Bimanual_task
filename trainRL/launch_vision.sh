#!/bin/bash
#SBATCH --job-name=vision_rl_train
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/vision_train_%j.log
#SBATCH --error=logs/vision_train_%j.err

set -e

TOTAL_TIMESTEPS="${1:-1000000}"
NUM_ENVS="${2:-4}"

TOTAL_TIMESTEPS="$(echo "$TOTAL_TIMESTEPS" | xargs)"
NUM_ENVS="$(echo "$NUM_ENVS" | xargs)"

export BASE_DIR="${HOME}/yash"
export TRAINRL_DIR="${BASE_DIR}/trainRL"
JOB_ID=${SLURM_JOB_ID:-local_$(date +%s)}
export RUN_DIR="${TRAINRL_DIR}/logs/vision_run_${JOB_ID}"
mkdir -p "${RUN_DIR}"

export PYTHONUNBUFFERED=1
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export PATH="${BASE_DIR}/miniforge3/envs/isaac_fresh/bin:${BASE_DIR}/miniforge3/bin:${PATH}"

cd "${TRAINRL_DIR}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  CUSTOM VISION RL - SINGLE GPU TRAINING                        ║"
echo "║  Job ID: ${JOB_ID}                                             ║"
echo "║  GPU: 0 (H100)                                                 ║"
echo "║  Environments: ${NUM_ENVS}                                            ║"
echo "║  Timesteps: ${TOTAL_TIMESTEPS}                                   ║"
echo "║  Output Dir: ${RUN_DIR}                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Copy scripts to the log folder for reproducibility
cp train_vision.py "${RUN_DIR}/"
cp vision_network.py "${RUN_DIR}/"

export GPU_ID=0
export NUM_ENVS=${NUM_ENVS}
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}
export RUN_DIR=${RUN_DIR}

python train_vision.py 2>&1 | tee "${RUN_DIR}/training.log"

TRAIN_EXIT_CODE=$?

echo ""
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "✅ Vision Training completed successfully!"
    echo "Results saved to: ${RUN_DIR}/"
else
    echo "❌ Training failed (exit code: $TRAIN_EXIT_CODE)"
fi
exit ${TRAIN_EXIT_CODE}