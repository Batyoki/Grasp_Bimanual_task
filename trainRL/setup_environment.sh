#!/bin/bash
################################################################################
# BIMANUAL RL - ENVIRONMENT SETUP & DEPENDENCY FIX
# ════════════════════════════════════════════════════════════════════════════
# Fixes dependency issues and prepares environment for training
# Following master_bimanual.sh guidelines
#
# Run on cluster:  sbatch setup_environment.sh
# Run locally:     bash setup_environment.sh
#
################################################################################

#SBATCH --job-name=bimanual_setup
#SBATCH --partition=gpu-a100
#SBATCH --time=00:30:00
#SBATCH --output=setup_%j.log
#SBATCH --error=setup_%j.err

set -e

################################################################################
# PART 1: HPC SURVIVAL SHIELDS & ENVIRONMENT SETUP
################################################################################

export BASE_DIR="${HOME}/yash"
export TRAINRL_DIR="${BASE_DIR}/trainRL"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  BIMANUAL RL - ENVIRONMENT SETUP & DEPENDENCY FIX             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 1A. REAL-TIME LOGGING
export PYTHONUNBUFFERED=1

# 1B. EULA BYPASS
export ACCEPT_EULA=Y
export ISAACSIM_ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# 1C. CACHE UNLOCK
unset PYTHONNOUSERSITE

# 1D. ACTIVATE ENVIRONMENT
echo "[INFO] Activating conda environment..."
source "${BASE_DIR}/miniforge3/bin/activate"
conda activate isaac_fresh

echo "[INFO] ✓ Environment activated"
echo ""

################################################################################
# PART 2: FIX NUMPY COMPATIBILITY ISSUE
################################################################################

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Fixing NumPy 2.x Compatibility Issues                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "[INFO] Current NumPy version:"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"

echo ""
echo "[INFO] Downgrading NumPy to <2 (required for stable-baselines3)..."
pip install --quiet "numpy<2" --upgrade

echo "[INFO] Updating dependent packages..."
pip install --quiet "numexpr>=2.8" --upgrade
pip install --quiet "bottleneck>=1.3" --upgrade
pip install --quiet "pandas>=2.0" --upgrade

echo ""
echo "[INFO] ✓ NumPy dependencies fixed"
echo ""

################################################################################
# PART 3: INSTALL MISSING OPENCV
################################################################################

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Installing Missing OpenCV                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "[INFO] Installing opencv-python..."
pip install --quiet opencv-python

echo ""
echo "[INFO] ✓ OpenCV installed"
echo ""

################################################################################
# PART 4: VERIFY ALL CRITICAL DEPENDENCIES
################################################################################

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Verifying All Critical Dependencies                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

verify_package() {
    local module=$1
    local display_name=$2
    if python -c "import ${module}" 2>/dev/null; then
        echo "  ✓ ${display_name}"
        return 0
    else
        echo "  ✗ ${display_name} - FAILED"
        return 1
    fi
}

FAIL_COUNT=0

echo "Core dependencies:"
verify_package "torch" "PyTorch" || ((FAIL_COUNT++))
verify_package "gymnasium" "Gymnasium" || ((FAIL_COUNT++))
verify_package "stable_baselines3" "Stable-Baselines3" || ((FAIL_COUNT++))
verify_package "numpy" "NumPy" || ((FAIL_COUNT++))

echo ""
echo "Data & Visualization:"
verify_package "pandas" "Pandas" || ((FAIL_COUNT++))
verify_package "imageio" "imageio" || ((FAIL_COUNT++))
verify_package "cv2" "OpenCV" || ((FAIL_COUNT++))
verify_package "tensorboard" "TensorBoard" || ((FAIL_COUNT++))
verify_package "yaml" "PyYAML" || ((FAIL_COUNT++))

echo ""
echo "IsaacLab (will test during training):"
python -c "from isaaclab.app import AppLauncher; print('  ✓ IsaacLab AppLauncher')" 2>/dev/null || echo "  ⚠ IsaacLab (expected - loads only with headless mode)"

echo ""

if [[ ${FAIL_COUNT} -eq 0 ]]; then
    echo "═" * 60
    echo "✓ ALL CRITICAL DEPENDENCIES VERIFIED"
    echo "═" * 60
    echo ""
    echo "Setup complete! You can now:"
    echo "  1. Run integration tests:    cd ${TRAINRL_DIR} && python integration_test.py"
    echo "  2. Submit training:          sbatch launch_parallel_training.sh"
    echo ""
    exit 0
else
    echo "═" * 60
    echo "✗ ${FAIL_COUNT} DEPENDENCY ISSUES FOUND"
    echo "═" * 60
    echo ""
    echo "Try fixing with:"
    echo "  pip install --upgrade --force-reinstall -r requirements.txt"
    echo ""
    exit 1
fi
