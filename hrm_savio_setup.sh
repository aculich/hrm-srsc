#!/bin/bash
#===============================================================================
# HRM (Hierarchical Reasoning Model) Setup and Job Submission Script for Savio
# UC Berkeley Research Computing
#
# This script sets up the environment and submits SLURM jobs for training
# the Hierarchical Reasoning Model on the Savio HPC cluster.
#
# Usage:
#   ./hrm_savio_setup.sh [setup|submit|both] [experiment_type]
#
# experiment_type options:
#   - sudoku_demo    : Single GPU Sudoku demo (~10 hours on RTX 4070 equivalent)
#   - arc1           : ARC-1 dataset (8 GPUs, ~24 hours)
#   - arc2           : ARC-2 dataset (8 GPUs, ~24 hours)
#   - sudoku_1k      : Sudoku Extreme 1k (8 GPUs, ~10 minutes)
#   - maze_1k        : Maze 30x30 Hard 1k (8 GPUs, ~1 hour)
#   - sudoku_full    : Full Sudoku Hard (8 GPUs, ~2 hours)
#===============================================================================

set -e  # Exit on any error

# ============================================================================
# Configuration Variables - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Your Savio account information
SAVIO_ACCOUNT="${SAVIO_ACCOUNT:-fc_your_account}"  # Replace with your account
SAVIO_EMAIL="${SAVIO_EMAIL:-your_email@berkeley.edu}"  # Replace with your email

# Project paths
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/HRM_project}"
SCRATCH_DIR="${SCRATCH_DIR:-$SCRATCH/hrm_workdir}"
ENV_PATH="${ENV_PATH:-$SCRATCH/hrm_env}"

# Default experiment type
EXPERIMENT_TYPE="${2:-sudoku_demo}"

# ============================================================================
# Available Modules on Savio (Based on current module listings)
# ============================================================================

# CUDA versions available: cuda/11.8.0, cuda/12.2.1, cuda/12.6.0
CUDA_MODULE="cuda/12.6.0"  # Exact version specified in HRM requirements

# Python versions available: python/3.10.12-gcc-11.4.0, python/3.11.6-gcc-11.4.0
PYTHON_MODULE="python/3.11.6-gcc-11.4.0"  # Compatible with HRM requirements

# Compiler versions available: gcc/10.5.0, gcc/11.4.0, gcc/13.2.0
GCC_MODULE="gcc/11.4.0"

# Additional useful modules
NINJA_MODULE="ninja/1.11.1"
CMAKE_MODULE="cmake/3.27.7"

# ============================================================================
# GPU Configurations for Different Experiments
# ============================================================================

setup_gpu_config() {
    local exp_type="$1"
    
    case "$exp_type" in
        "sudoku_demo")
            GPU_PARTITION="savio4_gpu"
            GPU_TYPE="A5000"
            NUM_GPUS=1
            CPU_PER_TASK=4
            MEMORY="32GB"
            TIME_LIMIT="12:00:00"
            NODES=1
            ;;
        "arc1"|"arc2"|"sudoku_1k"|"maze_1k"|"sudoku_full")
            GPU_PARTITION="savio4_gpu"
            GPU_TYPE="A5000"
            NUM_GPUS=8
            CPU_PER_TASK=32
            MEMORY="256GB"
            TIME_LIMIT="24:00:00"
            NODES=1
            ;;
        *)
            echo "Unknown experiment type: $exp_type"
            echo "Available types: sudoku_demo, arc1, arc2, sudoku_1k, maze_1k, sudoku_full"
            exit 1
            ;;
    esac
    
    # Quality of Service - adjust based on your access
    if [[ "$NUM_GPUS" -eq 8 ]]; then
        QOS="savio_normal"  # May need to be adjusted based on your allocation
    else
        QOS="savio_normal"
    fi
}

# ============================================================================
# Environment Setup Function
# ============================================================================

setup_environment() {
    echo "Setting up HRM environment on Savio..."
    
    # Create project directory
    mkdir -p "$PROJECT_ROOT"
    mkdir -p "$SCRATCH_DIR"
    
    # Check if we're on a login node
    if [[ -z "$SLURM_JOB_ID" ]]; then
        echo "Running on login node - setting up environment..."
    else
        echo "Running in SLURM job - loading modules..."
    fi
    
    # Purge any existing modules
    module purge
    
    # Load required modules
    echo "Loading modules..."
    module load "$GCC_MODULE"
    module load "$PYTHON_MODULE"
    
    # Try to load CUDA 12.6, fall back to 12.2 if not available
    if module load "$CUDA_MODULE" 2>/dev/null; then
        echo "Loaded CUDA $CUDA_MODULE"
    elif module load "cuda/12.2.1" 2>/dev/null; then
        echo "Loaded CUDA 12.2.1 (fallback)"
        CUDA_MODULE="cuda/12.2.1"
    else
        echo "WARNING: No compatible CUDA module found"
        echo "Available CUDA modules:"
        module avail cuda
        exit 1
    fi
    
    # Load additional build tools
    if module avail "$NINJA_MODULE" &>/dev/null; then
        module load "$NINJA_MODULE"
    fi
    
    if module avail "$CMAKE_MODULE" &>/dev/null; then
        module load "$CMAKE_MODULE"
    fi
    
    echo "Loaded modules:"
    module list
    
    # Set up Python virtual environment
    if [[ ! -d "$ENV_PATH" ]]; then
        echo "Creating Python virtual environment at $ENV_PATH..."
        python -m venv "$ENV_PATH"
    fi
    
    # Activate virtual environment
    source "$ENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install basic dependencies
    echo "Installing Python dependencies..."
    pip install packaging ninja
    
    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA support..."
    if [[ "$CUDA_MODULE" == *"12.6"* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    elif [[ "$CUDA_MODULE" == *"12.2"* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # Clone HRM repository if it doesn't exist
    if [[ ! -d "$PROJECT_ROOT/HRM" ]]; then
        echo "Cloning HRM repository..."
        cd "$PROJECT_ROOT"
        # Note: You'll need to provide the actual repository URL
        # git clone <HRM_REPOSITORY_URL> HRM
        echo "Please clone the HRM repository to $PROJECT_ROOT/HRM"
        echo "If you have the code locally, copy it to $PROJECT_ROOT/HRM"
    fi
    
    # Install HRM requirements
    if [[ -f "$PROJECT_ROOT/HRM/requirements.txt" ]]; then
        echo "Installing HRM requirements..."
        cd "$PROJECT_ROOT/HRM"
        pip install -r requirements.txt
    fi
    
    # Install FlashAttention based on GPU type
    echo "Installing FlashAttention..."
    if [[ "$GPU_TYPE" == "L40" ]]; then
        echo "Installing FlashAttention 3 for Hopper architecture..."
        # For L40 GPUs (Hopper architecture)
        if [[ ! -d "$SCRATCH_DIR/flash-attention" ]]; then
            cd "$SCRATCH_DIR"
            git clone https://github.com/Dao-AILab/flash-attention.git
        fi
        cd "$SCRATCH_DIR/flash-attention/hopper"
        python setup.py install
    else
        echo "Installing FlashAttention 2 for Ampere/earlier architecture..."
        # For A5000, A40, V100, GTX2080TI, TITAN GPUs
        pip install flash-attn
    fi
    
    # Initialize git submodules for datasets
    if [[ -d "$PROJECT_ROOT/HRM" ]]; then
        cd "$PROJECT_ROOT/HRM"
        if [[ -f ".gitmodules" ]]; then
            echo "Initializing git submodules..."
            git submodule update --init --recursive
        fi
    fi
    
    # Set up W&B (optional)
    echo "Setting up Weights & Biases..."
    pip install wandb
    echo "Run 'wandb login' to authenticate W&B (optional)"
    
    echo "Environment setup complete!"
    echo "Virtual environment: $ENV_PATH"
    echo "Project directory: $PROJECT_ROOT"
    echo "Scratch directory: $SCRATCH_DIR"
}

# ============================================================================
# Dataset Preparation Function
# ============================================================================

prepare_datasets() {
    local exp_type="$1"
    
    echo "Preparing datasets for experiment: $exp_type"
    
    cd "$PROJECT_ROOT/HRM"
    source "$ENV_PATH/bin/activate"
    
    case "$exp_type" in
        "sudoku_demo"|"sudoku_1k")
            echo "Building Sudoku dataset..."
            if [[ "$exp_type" == "sudoku_demo" ]]; then
                python dataset/build_sudoku_dataset.py \
                    --output-dir "$SCRATCH_DIR/data/sudoku-extreme-1k-aug-1000" \
                    --subsample-size 1000 \
                    --num-aug 1000
            else
                python dataset/build_sudoku_dataset.py \
                    --output-dir "$SCRATCH_DIR/data/sudoku-extreme-1k-aug-1000" \
                    --subsample-size 1000 \
                    --num-aug 1000
            fi
            ;;
        "arc1")
            echo "Building ARC-1 dataset..."
            python dataset/build_arc_dataset.py \
                --output-dir "$SCRATCH_DIR/data/arc-1-aug-1000"
            ;;
        "arc2")
            echo "Building ARC-2 dataset..."
            python dataset/build_arc_dataset.py \
                --dataset-dirs dataset/raw-data/ARC-AGI-2/data \
                --output-dir "$SCRATCH_DIR/data/arc-2-aug-1000"
            ;;
        "maze_1k")
            echo "Building Maze dataset..."
            python dataset/build_maze_dataset.py \
                --output-dir "$SCRATCH_DIR/data/maze-30x30-hard-1k"
            ;;
        "sudoku_full")
            echo "Building full Sudoku dataset..."
            python dataset/build_sudoku_dataset.py \
                --output-dir "$SCRATCH_DIR/data/sudoku-hard-full"
            ;;
    esac
}

# ============================================================================
# SLURM Job Script Generation
# ============================================================================

generate_slurm_script() {
    local exp_type="$1"
    local script_path="$SCRATCH_DIR/hrm_${exp_type}.slurm"
    
    setup_gpu_config "$exp_type"
    
    cat > "$script_path" << EOF
#!/bin/bash
#SBATCH --job-name=HRM_${exp_type}
#SBATCH --account=${SAVIO_ACCOUNT}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPU_PER_TASK}
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SCRATCH_DIR}/slurm-%j-${exp_type}.out
#SBATCH --error=${SCRATCH_DIR}/slurm-%j-${exp_type}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SAVIO_EMAIL}

# ============================================================================
# SLURM Job Script for HRM ${exp_type} experiment
# Generated on: $(date)
# ============================================================================

echo "Job started at: \$(date)"
echo "Running on node: \$SLURMD_NODENAME"
echo "Job ID: \$SLURM_JOB_ID"
echo "GPU allocation: \$CUDA_VISIBLE_DEVICES"

# Setup environment
module purge
module load ${GCC_MODULE}
module load ${PYTHON_MODULE}
module load ${CUDA_MODULE}

# Load additional modules if available
if module avail ${NINJA_MODULE} &>/dev/null; then
    module load ${NINJA_MODULE}
fi

if module avail ${CMAKE_MODULE} &>/dev/null; then
    module load ${CMAKE_MODULE}
fi

echo "Loaded modules:"
module list

# Activate virtual environment
source ${ENV_PATH}/bin/activate

# Set environment variables
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1  # For debugging CUDA errors

# Navigate to project directory
cd ${PROJECT_ROOT}/HRM

# Verify GPU availability
echo "GPU information:"
nvidia-smi

# Set data path based on experiment type
EOF

    # Add experiment-specific configurations
    case "$exp_type" in
        "sudoku_demo")
            cat >> "$script_path" << EOF

# Sudoku Demo - Single GPU
echo "Starting Sudoku demo training..."

# Prepare dataset if not exists
if [[ ! -d "${SCRATCH_DIR}/data/sudoku-extreme-1k-aug-1000" ]]; then
    echo "Preparing Sudoku dataset..."
    python dataset/build_sudoku_dataset.py \\
        --output-dir "${SCRATCH_DIR}/data/sudoku-extreme-1k-aug-1000" \\
        --subsample-size 1000 \\
        --num-aug 1000
fi

# Run training
python pretrain.py \\
    data_path="${SCRATCH_DIR}/data/sudoku-extreme-1k-aug-1000" \\
    epochs=20000 \\
    eval_interval=2000 \\
    global_batch_size=384 \\
    lr=7e-5 \\
    puzzle_emb_lr=7e-5 \\
    weight_decay=1.0 \\
    puzzle_emb_weight_decay=1.0

EOF
            ;;
        "arc1")
            cat >> "$script_path" << EOF

# ARC-1 - 8 GPU distributed training
echo "Starting ARC-1 training..."

# Prepare dataset if not exists
if [[ ! -d "${SCRATCH_DIR}/data/arc-1-aug-1000" ]]; then
    echo "Preparing ARC-1 dataset..."
    python dataset/build_arc_dataset.py \\
        --output-dir "${SCRATCH_DIR}/data/arc-1-aug-1000"
fi

# Run distributed training
torchrun --nproc-per-node ${NUM_GPUS} pretrain.py \\
    data_path="${SCRATCH_DIR}/data/arc-1-aug-1000"

EOF
            ;;
        "arc2")
            cat >> "$script_path" << EOF

# ARC-2 - 8 GPU distributed training
echo "Starting ARC-2 training..."

# Prepare dataset if not exists
if [[ ! -d "${SCRATCH_DIR}/data/arc-2-aug-1000" ]]; then
    echo "Preparing ARC-2 dataset..."
    python dataset/build_arc_dataset.py \\
        --dataset-dirs dataset/raw-data/ARC-AGI-2/data \\
        --output-dir "${SCRATCH_DIR}/data/arc-2-aug-1000"
fi

# Run distributed training
torchrun --nproc-per-node ${NUM_GPUS} pretrain.py \\
    data_path="${SCRATCH_DIR}/data/arc-2-aug-1000"

EOF
            ;;
        "sudoku_1k")
            cat >> "$script_path" << EOF

# Sudoku Extreme 1k - 8 GPU distributed training
echo "Starting Sudoku 1k training..."

# Prepare dataset if not exists
if [[ ! -d "${SCRATCH_DIR}/data/sudoku-extreme-1k-aug-1000" ]]; then
    echo "Preparing Sudoku dataset..."
    python dataset/build_sudoku_dataset.py \\
        --output-dir "${SCRATCH_DIR}/data/sudoku-extreme-1k-aug-1000" \\
        --subsample-size 1000 \\
        --num-aug 1000
fi

# Run distributed training
torchrun --nproc-per-node ${NUM_GPUS} pretrain.py \\
    data_path="${SCRATCH_DIR}/data/sudoku-extreme-1k-aug-1000" \\
    epochs=20000 \\
    eval_interval=2000 \\
    lr=1e-4 \\
    puzzle_emb_lr=1e-4 \\
    weight_decay=1.0 \\
    puzzle_emb_weight_decay=1.0

EOF
            ;;
        "maze_1k")
            cat >> "$script_path" << EOF

# Maze 30x30 Hard 1k - 8 GPU distributed training
echo "Starting Maze 1k training..."

# Prepare dataset if not exists
if [[ ! -d "${SCRATCH_DIR}/data/maze-30x30-hard-1k" ]]; then
    echo "Preparing Maze dataset..."
    python dataset/build_maze_dataset.py \\
        --output-dir "${SCRATCH_DIR}/data/maze-30x30-hard-1k"
fi

# Run distributed training
torchrun --nproc-per-node ${NUM_GPUS} pretrain.py \\
    data_path="${SCRATCH_DIR}/data/maze-30x30-hard-1k" \\
    epochs=20000 \\
    eval_interval=2000 \\
    lr=1e-4 \\
    puzzle_emb_lr=1e-4 \\
    weight_decay=1.0 \\
    puzzle_emb_weight_decay=1.0

EOF
            ;;
        "sudoku_full")
            cat >> "$script_path" << EOF

# Full Sudoku Hard - 8 GPU distributed training
echo "Starting Full Sudoku training..."

# Prepare dataset if not exists
if [[ ! -d "${SCRATCH_DIR}/data/sudoku-hard-full" ]]; then
    echo "Preparing full Sudoku dataset..."
    python dataset/build_sudoku_dataset.py \\
        --output-dir "${SCRATCH_DIR}/data/sudoku-hard-full"
fi

# Run distributed training
torchrun --nproc-per-node ${NUM_GPUS} pretrain.py \\
    data_path="${SCRATCH_DIR}/data/sudoku-hard-full" \\
    epochs=100 \\
    eval_interval=10 \\
    lr_min_ratio=0.1 \\
    global_batch_size=2304 \\
    lr=3e-4 \\
    puzzle_emb_lr=3e-4 \\
    weight_decay=0.1 \\
    puzzle_emb_weight_decay=0.1 \\
    arch.loss.loss_type=softmax_cross_entropy \\
    arch.L_cycles=8 \\
    arch.halt_max_steps=8 \\
    arch.pos_encodings=learned

EOF
            ;;
    esac
    
    # Add common ending
    cat >> "$script_path" << EOF

echo "Training completed at: \$(date)"

# Optional: Run evaluation if checkpoint exists
# CHECKPOINT_PATH=\$(find . -name "*.pt" -o -name "*.pth" | head -n 1)
# if [[ -n "\$CHECKPOINT_PATH" ]]; then
#     echo "Running evaluation on checkpoint: \$CHECKPOINT_PATH"
#     torchrun --nproc-per-node ${NUM_GPUS} evaluate.py checkpoint="\$CHECKPOINT_PATH"
# fi

echo "Job completed at: \$(date)"
EOF

    echo "Generated SLURM script: $script_path"
    return 0
}

# ============================================================================
# Job Submission Function
# ============================================================================

submit_job() {
    local exp_type="$1"
    local script_path="$SCRATCH_DIR/hrm_${exp_type}.slurm"
    
    if [[ ! -f "$script_path" ]]; then
        echo "SLURM script not found: $script_path"
        echo "Run setup first: $0 setup $exp_type"
        exit 1
    fi
    
    echo "Submitting SLURM job for experiment: $exp_type"
    echo "Script: $script_path"
    
    # Check account and QoS
    echo "Checking account and QoS access..."
    if command -v sacctmgr &> /dev/null; then
        echo "Available QoS for account $SAVIO_ACCOUNT:"
        sacctmgr -p show qos where account="$SAVIO_ACCOUNT" format=name%20
    fi
    
    # Submit the job
    if sbatch "$script_path"; then
        echo "Job submitted successfully!"
        echo "Monitor with: squeue -u \$USER"
        echo "Cancel with: scancel <JOBID>"
        echo "Logs will be in: $SCRATCH_DIR/"
    else
        echo "Failed to submit job"
        exit 1
    fi
}

# ============================================================================
# Information and Help Functions
# ============================================================================

show_status() {
    echo "=== HRM Savio Status ==="
    echo "Project root: $PROJECT_ROOT"
    echo "Scratch directory: $SCRATCH_DIR"
    echo "Virtual environment: $ENV_PATH"
    echo "Savio account: $SAVIO_ACCOUNT"
    echo "Savio email: $SAVIO_EMAIL"
    echo ""
    
    if [[ -d "$ENV_PATH" ]]; then
        echo "✓ Virtual environment exists"
    else
        echo "✗ Virtual environment not found"
    fi
    
    if [[ -d "$PROJECT_ROOT/HRM" ]]; then
        echo "✓ HRM project directory exists"
    else
        echo "✗ HRM project directory not found"
    fi
    
    echo ""
    echo "=== Current SLURM Jobs ==="
    if command -v squeue &> /dev/null; then
        squeue -u "$USER" -o "%.10i %.20j %.10T %.10M %.6D %.20S"
    else
        echo "squeue command not available"
    fi
}

show_help() {
    cat << EOF
HRM Savio Setup and Job Submission Script

Usage: $0 [COMMAND] [EXPERIMENT_TYPE]

COMMANDS:
    setup       - Set up environment and prepare datasets
    submit      - Submit SLURM job for training
    both        - Run setup then submit job
    status      - Show current status
    help        - Show this help message

EXPERIMENT_TYPES:
    sudoku_demo - Single GPU Sudoku demo (~10 hours)
    arc1        - ARC-1 dataset (8 GPUs, ~24 hours)
    arc2        - ARC-2 dataset (8 GPUs, ~24 hours)
    sudoku_1k   - Sudoku Extreme 1k (8 GPUs, ~10 minutes)
    maze_1k     - Maze 30x30 Hard 1k (8 GPUs, ~1 hour)
    sudoku_full - Full Sudoku Hard (8 GPUs, ~2 hours)

ENVIRONMENT VARIABLES:
    SAVIO_ACCOUNT  - Your Savio account (required)
    SAVIO_EMAIL    - Your email for notifications (required)
    PROJECT_ROOT   - Project directory (default: \$HOME/HRM_project)
    SCRATCH_DIR    - Scratch working directory (default: \$SCRATCH/hrm_workdir)
    ENV_PATH       - Python virtual environment path (default: \$SCRATCH/hrm_env)

EXAMPLES:
    # Set up environment for Sudoku demo
    $0 setup sudoku_demo
    
    # Submit ARC-1 training job
    $0 submit arc1
    
    # Set up and submit in one command
    $0 both sudoku_1k
    
    # Check status
    $0 status

PREREQUISITES:
    1. Set SAVIO_ACCOUNT and SAVIO_EMAIL environment variables
    2. Have HRM source code available
    3. Ensure you have access to GPU partitions on Savio

NOTES:
    - This script uses the latest CUDA 12.6 module when available
    - FlashAttention is automatically selected based on GPU architecture
    - All data and checkpoints are stored in SCRATCH for fast I/O
    - Virtual environment is created in SCRATCH to avoid home directory limits

For more information, see the Savio documentation:
https://docs-research-it.berkeley.edu/services/high-performance-computing/
EOF
}

# ============================================================================
# Module Verification Function
# ============================================================================

verify_modules() {
    echo "=== Verifying Savio Modules ==="
    
    echo "Checking CUDA modules..."
    if module avail cuda 2>&1 | grep -q "12.6.0"; then
        echo "✓ CUDA 12.6.0 available"
    elif module avail cuda 2>&1 | grep -q "12.2"; then
        echo "⚠ CUDA 12.6.0 not found, will use 12.2.1"
    else
        echo "✗ No compatible CUDA module found"
    fi
    
    echo "Checking Python modules..."
    if module avail python 2>&1 | grep -q "3.11.6"; then
        echo "✓ Python 3.11.6 available"
    elif module avail python 2>&1 | grep -q "3.10"; then
        echo "⚠ Python 3.11.6 not found, will use 3.10"
    else
        echo "✗ No compatible Python module found"
    fi
    
    echo "Checking PyTorch ML modules..."
    if module avail ml/pytorch 2>&1 | grep -q "pytorch"; then
        echo "ℹ Pre-built PyTorch modules available, but we'll install via pip for CUDA compatibility"
    fi
    
    echo ""
    echo "Run 'module spider <module_name>' for detailed information about specific modules"
}

# ============================================================================
# Main Function
# ============================================================================

main() {
    local command="${1:-help}"
    
    # Check required environment variables
    if [[ -z "$SAVIO_ACCOUNT" && "$command" != "help" && "$command" != "status" ]]; then
        echo "Error: SAVIO_ACCOUNT environment variable not set"
        echo "Set it with: export SAVIO_ACCOUNT=your_account_name"
        exit 1
    fi
    
    if [[ -z "$SAVIO_EMAIL" && "$command" != "help" && "$command" != "status" ]]; then
        echo "Error: SAVIO_EMAIL environment variable not set"
        echo "Set it with: export SAVIO_EMAIL=your_email@berkeley.edu"
        exit 1
    fi
    
    case "$command" in
        "setup")
            verify_modules
            setup_environment
            generate_slurm_script "$EXPERIMENT_TYPE"
            echo "Setup complete! Next step: $0 submit $EXPERIMENT_TYPE"
            ;;
        "submit")
            generate_slurm_script "$EXPERIMENT_TYPE"
            submit_job "$EXPERIMENT_TYPE"
            ;;
        "both")
            verify_modules
            setup_environment
            generate_slurm_script "$EXPERIMENT_TYPE"
            submit_job "$EXPERIMENT_TYPE"
            ;;
        "status")
            show_status
            ;;
        "verify")
            verify_modules
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# ============================================================================
# Script Execution
# ============================================================================

# Update TODO progress
echo "HRM Savio setup script ready!"

main "$@"