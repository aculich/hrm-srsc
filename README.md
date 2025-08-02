# Hierarchical Reasoning Model (HRM) - Installation & Setup Guide

This repository contains the complete setup for running the [Hierarchical Reasoning Model (HRM)](https://github.com/sapientinc/HRM) from Sapient Inc., a novel recurrent architecture for complex reasoning tasks.

> **Paper**: [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) - Wang et al., 2025

## 🎯 Quick Start Options

Choose your preferred environment:

- **[🖥️ Local Installation](#local-installation)** - For NVIDIA GPU systems
- **[☁️ Savio HPC Cluster](#savio-hpc-cluster)** - UC Berkeley cluster (recommended)
- **[🌐 Savio Open OnDemand](#savio-open-ondemand-quick-start)** - Web-based interface (easiest)

## 🖥️ Local Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA 12.6 (preferred) or 12.2+
- Python 3.11+
- Git with LFS support

### System Requirements

- **GPU Memory**: 24GB+ recommended (A5000/RTX 4090 or better)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space

### Installation Steps

1. **Clone the HRM repository**:

   ```bash
   git clone https://github.com/sapientinc/HRM.git
   cd HRM
   git submodule update --init --recursive
   ```

2. **Set up Python environment**:

   ```bash
   python -m venv hrm_env
   source hrm_env/bin/activate  # On Windows: hrm_env\Scripts\activate
   pip install --upgrade pip setuptools wheel
   ```

3. **Install CUDA dependencies**:

   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   
   # Install build dependencies
   pip install packaging ninja wheel setuptools setuptools-scm
   ```

4. **Install HRM requirements**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Install FlashAttention**:

   ```bash
   # For most modern GPUs (Ampere/Ada Lovelace)
   pip install flash-attn
   
   # For Hopper GPUs (H100, L40) - if needed
   # git clone https://github.com/Dao-AILab/flash-attention.git
   # cd flash-attention/hopper
   # python setup.py install
   ```

6. **Verify installation**:

   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import flash_attn; print('FlashAttention installed successfully')"
   ```

### Quick Test Run

Run a minimal Sudoku demo (single GPU):

```bash
# Prepare small dataset
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-test \
    --subsample-size 100 \
    --num-aug 10

# Train for a few iterations (test run)
OMP_NUM_THREADS=8 python pretrain.py \
    data_path=data/sudoku-test \
    epochs=100 \
    eval_interval=50 \
    global_batch_size=32 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

## ☁️ Savio HPC Cluster

For UC Berkeley users with Savio access, use our automated setup script.

### Prerequisites

- Savio account with GPU allocation
- Access to `savio4_gpu` partition

### Quick Setup

1. **Configure credentials**:

   ```bash
   export SAVIO_ACCOUNT="fc_your_account"  # Replace with your account
   export SAVIO_EMAIL="your_email@berkeley.edu"
   ```

2. **Clone this repository** (contains setup scripts):

   ```bash
   git clone <this_repository_url>
   cd <repository_name>
   ```

3. **Run automated setup and training**:

   ```bash
   # Quick Sudoku demo (single GPU, ~2 hours)
   ./hrm_savio_setup.sh both sudoku_demo
   
   # Full ARC-1 training (8 GPUs, ~24 hours)
   ./hrm_savio_setup.sh both arc1
   ```

For detailed Savio instructions, see [`HRM_SAVIO_GUIDE.md`](HRM_SAVIO_GUIDE.md).

## 🌐 Savio Open OnDemand Quick Start

**The easiest way to get started** - no command line required!

### Step 1: Access Open OnDemand

1. Go to [https://ood.brc.berkeley.edu](https://ood.brc.berkeley.edu)
2. Log in with your Berkeley credentials
3. Navigate to **Interactive Apps** → **Jupyter**

### Step 2: Launch Jupyter Session

Configure your session:

- **Account**: Your Savio account (e.g., `fc_faculty`)
- **Partition**: `savio4_gpu`
- **QoS**: `savio_normal`
- **Number of nodes**: 1
- **Number of cores**: 8
- **GPUs**: `gpu:A5000:1` (for single GPU test)
- **Memory**: 32 GB
- **Wall time**: 4 hours (for initial test)

Click **Launch** and wait for the session to start.

### Step 3: Set Up Environment

Once Jupyter launches, open a **Terminal** and run:

```bash
# Navigate to your scratch directory
cd $SCRATCH

# Load required modules
module purge
module load python/3.11.6-gcc-11.4.0
module load cuda/12.6.0

# Create virtual environment
python -m venv hrm_env
source hrm_env/bin/activate

# Install PyTorch and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn packaging ninja einops tqdm coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub adam-atan2

# Clone HRM repository
git clone https://github.com/sapientinc/HRM.git
cd HRM
git submodule update --init --recursive
```

### Step 4: Quick Test in Jupyter

Create a new **Python 3** notebook and run:

```python
# Cell 1: Verify GPU access
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Cell 2: Test FlashAttention
try:
    import flash_attn
    print("FlashAttention imported successfully!")
except ImportError as e:
    print(f"FlashAttention import failed: {e}")
    print("Installing FlashAttention...")
    !pip install flash-attn
    import flash_attn
    print("FlashAttention installed and imported!")

# Cell 3: Prepare minimal dataset
import os
os.chdir('/global/scratch/users/$USER/HRM')  # Replace $USER with your username

# Build tiny dataset for testing
!python dataset/build_sudoku_dataset.py --output-dir data/sudoku-minimal --subsample-size 10 --num-aug 5

print("Minimal dataset created!")
```

### Step 5: Run Minimal Training

```python
# Cell 4: Start minimal training
import subprocess
import os

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '8'

# Run minimal training (just a few steps to test)
cmd = [
    'python', 'pretrain.py',
    'data_path=data/sudoku-minimal',
    'epochs=10',
    'eval_interval=5',
    'global_batch_size=8',
    'lr=7e-5',
    'puzzle_emb_lr=7e-5',
    'weight_decay=1.0',
    'puzzle_emb_weight_decay=1.0'
]

print("Starting HRM training test...")
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)
print(f"Return code: {result.returncode}")
```

### Step 6: Monitor Progress

If training starts successfully, you can monitor it with:

```python
# Cell 5: Check training progress
import time
import os

# Look for log files or checkpoints
if os.path.exists('logs'):
    !ls -la logs/
if os.path.exists('checkpoints'):
    !ls -la checkpoints/

# Check GPU utilization
!nvidia-smi
```

### Troubleshooting Open OnDemand

**Common Issues:**

1. **Session won't start**: Check if your account has GPU allocation

   ```bash
   sacctmgr show user $USER format=account%20,qos%50
   ```

2. **Out of memory**: Reduce batch size or use smaller dataset

   ```python
   # Use smaller parameters
   global_batch_size=4
   subsample_size=5
   ```

3. **CUDA errors**: Verify GPU allocation in your session

   ```bash
   echo $CUDA_VISIBLE_DEVICES
   nvidia-smi
   ```

4. **Module not found**: Reload modules in each new terminal

   ```bash
   module purge
   module load python/3.11.6-gcc-11.4.0 cuda/12.6.0
   source hrm_env/bin/activate
   ```

## 📊 Available Experiments

Once your test runs successfully, try these full experiments:

| Experiment | Dataset | GPUs | Runtime | Description |
|------------|---------|------|---------|-------------|
| **Sudoku Demo** | 1K samples | 1 | ~10h | Perfect for testing |
| **ARC-1** | 960 samples | 8 | ~24h | Reasoning benchmark |
| **ARC-2** | 1.1K samples | 8 | ~24h | Latest ARC dataset |
| **Maze Solving** | 1K samples | 8 | ~1h | Path planning |
| **Full Sudoku** | Full dataset | 8 | ~2h | Complete Sudoku training |

## 📈 Monitoring with Weights & Biases

Set up experiment tracking:

```bash
# Install and login to W&B
pip install wandb
wandb login  # Enter your API key

# Training metrics will be automatically logged
```

View your experiments at [wandb.ai](https://wandb.ai).

## 🔧 Advanced Configuration

### Custom Experiments

Create custom training configurations by modifying the arguments:

```bash
python pretrain.py \
    data_path=your/custom/dataset \
    epochs=50000 \
    eval_interval=1000 \
    global_batch_size=512 \
    lr=1e-4 \
    arch.L_cycles=16 \
    arch.halt_max_steps=16
```

### Multi-GPU Training

For distributed training on multiple GPUs:

```bash
# On Savio (8 GPUs)
torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-1-aug-1000

# Local multi-GPU
torchrun --nproc-per-node 4 pretrain.py data_path=data/sudoku-1k
```

### Evaluation

Evaluate trained models:

```bash
# Single GPU evaluation
python evaluate.py checkpoint=path/to/checkpoint.pt

# Multi-GPU evaluation
torchrun --nproc-per-node 8 evaluate.py checkpoint=path/to/checkpoint.pt

# Use the arc_eval.ipynb notebook for detailed ARC analysis
```

## 📚 Resources

- **📄 Paper**: [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- **💻 Original Repository**: [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)
- **🏛️ Savio Documentation**: [Berkeley Research Computing](https://docs-research-it.berkeley.edu/services/high-performance-computing/)
- **📧 Support**: [research-it@berkeley.edu](mailto:research-it@berkeley.edu)

## 🐛 Troubleshooting

### CUDA Issues

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

- Reduce `global_batch_size`
- Use gradient accumulation
- Enable mixed precision training

### Module Issues

```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check FlashAttention
pip install flash-attn --no-build-isolation
```

## 📄 License

This project uses the Apache 2.0 License. See the [original HRM repository](https://github.com/sapientinc/HRM) for details.

## 🙏 Citation

If you use this work, please cite:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

---

**⚡ Quick Start Summary:**

1. **Local**: Clone → Install → Test with minimal Sudoku
2. **Savio CLI**: Set credentials → Run `./hrm_savio_setup.sh both sudoku_demo`
3. **Savio Web**: Open OnDemand → Jupyter → Follow Step-by-step guide above

Choose the method that best fits your setup and experience level!