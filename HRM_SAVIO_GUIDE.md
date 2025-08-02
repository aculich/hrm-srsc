# Running HRM on UC Berkeley Savio HPC Cluster

This guide provides complete instructions for setting up and running the Hierarchical Reasoning Model (HRM) on the Savio high-performance computing cluster at UC Berkeley.

## Quick Start

1. **Set environment variables**:
   ```bash
   export SAVIO_ACCOUNT="fc_your_account"  # Replace with your actual account
   export SAVIO_EMAIL="your_email@berkeley.edu"  # Replace with your email
   ```

2. **Run setup and training**:
   ```bash
   # For a quick Sudoku demo (single GPU)
   ./hrm_savio_setup.sh both sudoku_demo
   
   # For full ARC-1 training (8 GPUs)
   ./hrm_savio_setup.sh both arc1
   ```

## Prerequisites

### Savio Account Access
- Access to Savio cluster with GPU allocation
- Account with access to `savio4_gpu` partition (recommended)
- Sufficient service units for GPU hours

### Source Code
- HRM source code (place in `$HOME/HRM_project/HRM/` or set `PROJECT_ROOT`)
- Git repository with submodules initialized

## Detailed Setup Instructions

### 1. Account Configuration

First, determine your Savio account details:

```bash
# Check your account associations
sacctmgr show user $USER format=account%20,qos%50

# Check available QoS for your account
sacctmgr -p show qos where account="your_account" format=name%20
```

Set the environment variables:
```bash
export SAVIO_ACCOUNT="fc_your_faculty"  # Your actual account
export SAVIO_EMAIL="your_email@berkeley.edu"
```

Add these to your `~/.bashrc` or `~/.bash_profile` to make them permanent.

### 2. GPU Partition Access

Check available GPU resources:
```bash
# Check GPU partition status
sinfo -p savio4_gpu,savio3_gpu -o "%P %A %D %N %G"

# Check GPU types available
scontrol show partition savio4_gpu
```

### 3. Source Code Setup

Prepare the HRM source code:

```bash
# Option 1: If you have the code locally
cp -r /path/to/HRM $HOME/HRM_project/

# Option 2: Clone from repository (update URL as needed)
mkdir -p $HOME/HRM_project
cd $HOME/HRM_project
git clone <HRM_REPOSITORY_URL> HRM
cd HRM
git submodule update --init --recursive
```

## Available Experiments

The setup script supports these pre-configured experiments:

| Experiment | GPUs | Runtime | Memory | Description |
|------------|------|---------|---------|-------------|
| `sudoku_demo` | 1 | ~12h | 32GB | Single GPU Sudoku demonstration |
| `arc1` | 8 | ~24h | 256GB | ARC-1 dataset training |
| `arc2` | 8 | ~24h | 256GB | ARC-2 dataset training |
| `sudoku_1k` | 8 | ~10min | 256GB | Sudoku Extreme 1000 examples |
| `maze_1k` | 8 | ~1h | 256GB | Maze 30x30 Hard 1000 examples |
| `sudoku_full` | 8 | ~2h | 256GB | Full Sudoku Hard dataset |

## Usage Examples

### Environment Setup Only
```bash
# Set up environment without submitting job
./hrm_savio_setup.sh setup sudoku_demo
```

### Job Submission Only
```bash
# Submit job (assumes environment is already set up)
./hrm_savio_setup.sh submit arc1
```

### Complete Workflow
```bash
# Setup environment and submit job in one command
./hrm_savio_setup.sh both maze_1k
```

### Status Checking
```bash
# Check current status
./hrm_savio_setup.sh status

# Check running jobs
squeue -u $USER

# Check job details
scontrol show job <JOBID>
```

## System Configuration

### Module Selection

The script automatically selects the best available modules:

- **CUDA**: Prefers 12.6.0, falls back to 12.2.1
- **Python**: Uses 3.11.6, falls back to 3.10.12
- **GCC**: Uses 11.4.0 for compatibility
- **Additional**: Loads ninja, cmake when available

### GPU Selection Strategy

| GPU Type | Architecture | Memory | CPU Ratio | Use Case |
|----------|-------------|---------|-----------|----------|
| **A5000** | Ampere | 24GB | 4:1 | Recommended for most experiments |
| **L40** | Hopper | 46GB | 8:1 | Large models, uses FlashAttention 3 |
| **A40** | Ampere | 48GB | 8:1 | High memory requirements |

### Storage Strategy

- **Home Directory** (`$HOME`): Scripts and small files only
- **Scratch Storage** (`$SCRATCH`): Data, models, virtual environment
- **Project Directory**: Source code and configurations

## Monitoring and Debugging

### Job Monitoring
```bash
# Watch job queue
watch -n 30 'squeue -u $USER'

# Monitor GPU usage during job
ssh <node_name>
nvidia-smi -l 5
```

### Log Files
```bash
# View job output
tail -f $SCRATCH/hrm_workdir/slurm-<JOBID>-<experiment>.out

# View job errors
tail -f $SCRATCH/hrm_workdir/slurm-<JOBID>-<experiment>.err
```

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in job script
global_batch_size=192  # Instead of 384
```

#### 2. Module Loading Failures
```bash
# Check available modules
module avail cuda
module avail python

# Use script's verify function
./hrm_savio_setup.sh verify
```

#### 3. FlashAttention Installation Issues
```bash
# Check GPU architecture compatibility
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Manual installation
source $SCRATCH/hrm_env/bin/activate
pip install flash-attn --no-build-isolation
```

#### 4. Dataset Preparation Failures
```bash
# Manual dataset building
cd $HOME/HRM_project/HRM
source $SCRATCH/hrm_env/bin/activate
python dataset/build_sudoku_dataset.py --help
```

## Performance Optimization

### For Single GPU Jobs (sudoku_demo)
- Use `savio4_gpu` with A5000 GPUs
- Set `OMP_NUM_THREADS=8`
- Consider CPU-intensive preprocessing

### For Multi-GPU Jobs (8 GPU)
- Request full node to avoid interference
- Use distributed training with `torchrun`
- Monitor GPU utilization across all devices

### Memory Management
- Store datasets on scratch filesystem (`$SCRATCH`)
- Use appropriate batch sizes for GPU memory
- Monitor memory usage with `nvidia-smi`

## Cost Management

### Service Unit Usage
- **A5000 GPU**: 4.67 SU per core-hour
- **A40 GPU**: 3.67 SU per core-hour  
- **L40 GPU**: 4.67 SU per core-hour

### Optimization Strategies
1. Use the `sudoku_demo` experiment first to test setup
2. Monitor jobs and cancel if not progressing
3. Use checkpointing to resume interrupted jobs
4. Consider lower-priority queues if available

## Advanced Configuration

### Custom Data Paths
```bash
export PROJECT_ROOT="/global/home/users/$USER/custom_hrm"
export SCRATCH_DIR="/global/scratch/users/$USER/hrm_data"
export ENV_PATH="/global/scratch/users/$USER/hrm_python_env"
```

### Manual SLURM Script Customization

The generated SLURM scripts are saved in `$SCRATCH_DIR/hrm_<experiment>.slurm`. You can modify these directly before submission:

```bash
# Edit generated script
nano $SCRATCH_DIR/hrm_sudoku_demo.slurm

# Submit modified script
sbatch $SCRATCH_DIR/hrm_sudoku_demo.slurm
```

### Custom Experiment Parameters

Modify the experiment parameters by editing the generated SLURM script or creating custom training commands:

```bash
# Example: Custom Sudoku training
torchrun --nproc-per-node 8 pretrain.py \
    data_path="/path/to/custom/data" \
    epochs=30000 \
    lr=5e-5 \
    global_batch_size=512
```

## Integration with Weights & Biases

### Setup W&B
```bash
# Login to W&B (run once)
source $SCRATCH/hrm_env/bin/activate
wandb login

# Your W&B API key will be saved automatically
```

### Monitor Training
- View real-time metrics at [wandb.ai](https://wandb.ai)
- Track GPU utilization and training progress
- Compare different experiment runs

## Troubleshooting

### Script Issues
```bash
# Get help
./hrm_savio_setup.sh help

# Check script status
./hrm_savio_setup.sh status

# Verify module availability
./hrm_savio_setup.sh verify
```

### SLURM Issues
```bash
# Check job status
scontrol show job <JOBID>

# Check partition availability
sinfo -p savio4_gpu

# Check account limits
sacctmgr show user $USER format=account%20,qos%50
```

### Environment Issues
```bash
# Recreate virtual environment
rm -rf $SCRATCH/hrm_env
./hrm_savio_setup.sh setup <experiment>

# Check Python packages
source $SCRATCH/hrm_env/bin/activate
pip list | grep torch
```

## Support Resources

- **Savio Documentation**: [docs-research-it.berkeley.edu](https://docs-research-it.berkeley.edu/services/high-performance-computing/)
- **Savio Consulting**: [research-it@berkeley.edu](mailto:research-it@berkeley.edu)
- **Office Hours**: Check Research IT website for current schedule
- **Slack Channel**: BRC Users Slack workspace

## References

1. [Savio User Guide](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/)
2. [Savio GPU Documentation](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/#gpu-job-script)
3. [HRM Paper](https://arxiv.org/abs/2506.21734) - Hierarchical Reasoning Model
4. [Berkeley Research Computing](https://research-it.berkeley.edu/services/high-performance-computing)

---

**Note**: This guide is based on the Savio system configuration as of January 2025. Module availability and system specifications may change. Always check current documentation and run `./hrm_savio_setup.sh verify` to confirm module availability.