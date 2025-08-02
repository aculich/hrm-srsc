# Running on Savio HPC Cluster at UC Berkeley



## Research IT Chatbot Results

This project, the Hierarchical Reasoning Model (HRM), involves training deep learning models with PyTorch and CUDA, which is well-suited for Savio's GPU nodes. The quick start guide indicates two main configurations: a single GPU for a quick demo and an 8-GPU setup for full-scale experiments.

Here's a recommended Slurm configuration and setup guide for running HRM on Savio:

### GPU Selection

*   **For single GPU experiments (e.g., Sudoku Solver demo):**
    *   You can use any GPU type available on Savio. A good choice would be an **A40** on `savio3_gpu` or an **A5000** on `savio4_gpu`. The A40 has an 8:1 CPU:GPU ratio, which aligns well with the `OMP_NUM_THREADS=8` setting.
    *   Example Slurm `gres` specification: `--gres=gpu:A40:1`
*   **For 8-GPU full-scale experiments:**
    *   You will need a node with 8 GPUs. The **A5000** or **L40** GPUs on the `savio4_gpu` partition are ideal, as nodes in this partition typically have 8 GPUs. The **TITAN** GPUs on `savio3_gpu` also offer 8 GPUs per node.
    *   The L40 GPUs on `savio4_gpu` have an 8:1 CPU:GPU ratio, meaning an 8-GPU node would provide 64 CPU cores, which aligns perfectly with `OMP_NUM_THREADS=8` for each of the 8 `torchrun` processes (8 GPUs * 8 threads/GPU = 64 threads).
    *   Example Slurm `gres` specification: `--gres=gpu:L40:8`

### Slurm Job Script Template

Below is a template for your Slurm job script. Remember to replace placeholders like `<your_account>`, `<your_username>`, and choose the appropriate partition, QoS, and GPU type.

```bash
#!/bin/bash
#SBATCH --job-name=HRM_training
#SBATCH --account=<your_account>        # Your Savio account (e.g., co_faculty)
#SBATCH --partition=<savio_gpu_partition> # e.g., savio4_gpu for A5000/L40, savio3_gpu for A40/TITAN
#SBATCH --qos=<appropriate_qos>         # Check 'sacctmgr -p show qos' for your allowed QoS
#SBATCH --nodes=1                       # Request one node
#SBATCH --ntasks-per-node=1             # torchrun handles parallelism within the node
#SBATCH --cpus-per-task=<num_cpus>      # For 1 GPU: 8 (for A40) or 4 (for A5000)
                                        # For 8 GPUs: 64 (for L40) or 32 (for A5000/TITAN)
#SBATCH --gres=gpu:<gpu_type>:<num_gpus> # e.g., gpu:L40:8 or gpu:A40:1
#SBATCH --time=24:00:00                 # Max runtime (adjust based on experiment, e.g., 10:00:00 for demo, 24:00:00 for full-scale)
#SBATCH --output=slurm-%j.out           # Standard output file
#SBATCH --error=slurm-%j.err            # Standard error file
#SBATCH --mail-type=END,FAIL            # Email notifications for job end or failure
#SBATCH --mail-user=<your_email>        # Your email address

# --- Setup Environment ---
# Purge existing modules to ensure a clean environment
module purge

# Load necessary modules
# The paper specifies CUDA 12.6. Check 'module avail cuda' on Savio for available versions
# and choose the closest compatible one (e.g., cuda/12.2).
module load python/3.9
module load cuda/12.2 # Adjust CUDA version as available on Savio

# Create and activate a Python virtual environment (highly recommended)
# This only needs to be done once. If you've already created it, just activate.
# python -m venv $SCRATCH/hrm_env
# source $SCRATCH/hrm_env/bin/activate

# Install Python dependencies (run these commands only once after activating your venv)
# Navigate to your project directory first
# cd /global/home/users/<your_username>/HRM_project

# pip install -r requirements.txt
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# pip install packaging ninja wheel setuptools setuptools-scm

# Install FlashAttention based on your chosen GPU type:
# For Hopper GPUs (L40):
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# python setup.py install
# For Ampere or earlier GPUs (A5000, A40, V100, GTX2080TI, TITAN):
# pip install flash-attn

# Log in to Weights & Biases (if using)
# wandb login

# --- Run Experiment ---
# Set OMP_NUM_THREADS as specified in the paper
export OMP_NUM_THREADS=8

# Navigate to your project directory where the HRM code is located
cd /global/home/users/<your_username>/HRM_project # Adjust this path

# Example: Quick Demo - Sudoku Solver (single GPU)
# Uncomment and run if you are using a single GPU setup
# python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
# python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Example: Full-scale ARC-1 (8 GPUs)
# Uncomment and run if you are using an 8-GPU setup
# Ensure submodules are initialized: git submodule update --init --recursive
# python dataset/build_arc_dataset.py
torchrun --nproc-per-node 8 pretrain.py

# Choose and uncomment the specific full-scale experiment you want to run:
# ARC-2:
# torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000

# Sudoku Extreme (1k):
# torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Maze 30x30 Hard (1k):
# torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Full Sudoku-Hard:
# torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 lr_min_ratio=0.1 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned

# Evaluation (after training)
# torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

### Key Considerations:

1.  **Account and QoS:** Replace `<your_account>` and `<appropriate_qos>` with your actual Savio account and the Quality of Service you are allowed to use. You can check your allowed QoS with `sacctmgr -p show qos`.
2.  **CUDA Version:** The paper specifies CUDA 12.6. Savio may not have this exact version. Use `module avail cuda` to see available versions and select the closest compatible one (e.g., `cuda/12.2`).
3.  **Python Environment:** Always use a Python virtual environment (`venv` or `conda`) to manage your project's dependencies. Install all `pip` packages within this environment.
4.  **FlashAttention:** Ensure you install the correct FlashAttention version (`flash-attention/hopper` for L40 GPUs, `flash-attn` for A5000, A40, V100, TITAN, GTX2080TI).
5.  **`OMP_NUM_THREADS`:** The script sets `export OMP_NUM_THREADS=8` as specified in the paper. This helps optimize CPU-bound operations.
6.  **`torchrun`:** For multi-GPU training, the project uses `torchrun`. This command handles distributing the workload across the requested GPUs on a single node. You should request `--ntasks-per-node=1` as `torchrun` manages the parallel processes.
7.  **Data Paths:** Adjust `data_path` arguments to point to where you store your datasets on Savio (e.g., in your `$SCRATCH` directory).
8.  **Runtime:** Adjust `--time` based on the expected runtime of your specific experiment.
9.  **Project Directory:** Make sure to `cd` into your project directory where the `pretrain.py` and `dataset` scripts are located.

To submit your job, save the script (e.g., `hrm_job.sh`) and run:
`sbatch hrm_job.sh`

---

## Savio Documentation

### Savio Hardware Config

https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

| Partition                                 | Nodes | Node List                                                                                                                     | CPU Model                         | # Cores / Node | Memory / Node | Infiniband | Specialty                                    | Scheduler Allocation |
|-------------------------------------------|-------|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|----------------|---------------|------------|----------------------------------------------|----------------------|
| savio [Retired]                           | 132   | n0[000-095].savio1                                                                                                            | Intel Xeon E5-2670 v2             | 20             | 64 GB         | FDR        | -                                            | By Node              |
| savio_bigmem [Retired]                    | 4     | n0[096-099].savio1                                                                                                            | Intel Xeon E5-2670 v2             | 20             | 512 GB        | FDR        | BIGMEM                                       | By Node              |
| savio2                                    | 124   | n0[027-150].savio2                                                                                                            | Intel Xeon E5-2670 v3             | 24             | 64 GB         | FDR        | -                                            | By Node              |
| savio2                                    | 4     | n0[290-293].savio2                                                                                                            | Intel Xeon E5-2650 v4             | 24             | 64 GB         | FDR        | -                                            | By Node              |
| savio2                                    | 35    | n0[187-210].savio2 n0[230-240].savio2                                                                                         | Intel Xeon E5-2680 v4             | 28             | 64 GB         | FDR        | -                                            | By Node              |
| savio2_bigmem                             | 36    | n0[151-182].savio2                                                                                                            | Intel Xeon E5-2670 v3             | 24             | 128 GB        | FDR        | BIGMEM                                       | By Node              |
| savio2_bigmem                             | 8     | n0[282-289].savio2                                                                                                            | Intel Xeon E5-2650 v3             | 24             | 128 GB        | FDR        | BIGMEM                                       | By Node              |
| savio2_htc                                | 20    | n0[000-011].savio2 n0[215-222].savio2                                                                                         | Intel Xeon E5-2643 v3             | 12             | 128 GB        | FDR        | HTC                                          | By Core              |
| savio2_gpu [Retired (GPUs not available)] | 17    | n0[012-026].savio2 n0[223-224].savio2                                                                                         | Intel Xeon E5-2623 v3             | 8              | 64 GB         | FDR        | 4x Nvidia K80 (12 GB GPU memory per GPU)     | By Core              |
| savio2_1080ti                             | 8     | n0[227-229].savio2 n0[298-302]                                                                                                | Intel Xeon E5-2623 v3             | 8              | 64 GB         | FDR        | 4x Nvidia 1080ti (11 GB GPU memory per GPU)  | By Core              |
| savio2_knl                                | 28    | n0[254-281].savio2                                                                                                            | Intel Xeon Phi 7210               | 64             | 188 GB        | FDR        | Intel Phi                                    | By Node              |
| savio3                                    | 112   | n0[010-029].savio3 n0[042-125].savio3 n0[146-149].savio3                                                                      | Intel Xeon Skylake 6130 @ 2.1 GHz | 32             | 96 GB         | FDR        |                                              | By Node              |
| savio3                                    | 80    | n0[126-133].savio3 n0[139-142].savio3 n0[150-157] n0[170-173].savio3 n0[193-208].savio3 n0[222-257].savio3 n0[265-272].savio3 | Intel Xeon Skylake 6230 @ 2.1 GHz | 40             | 96 GB         | FDR        |                                              | By Node              |
| savio3_bigmem                             | 16    | n0[006-009].savio3 n0[030-041].savio3                                                                                         | Intel Xeon Skylake 6130 @ 2.1 GHz | 32             | 384 GB        | FDR        | BIGMEM                                       | By Node              |
| savio3_bigmem                             | 4     | n0[162-165].savio3                                                                                                            | Intel Xeon Gold 6230 @ 2.1 GHz    | 40             | 384 GB        | FDR        | BIGMEM                                       | By Node              |
| savio3_htc                                | 24    | n0[166-169].savio3 n0[177-192].savio3 n0[218-221].savio3                                                                      | Intel Xeon Skylake 6230 @ 2.1 GHz | 40             | 384 GB        | FDR        | HTC                                          | By Core              |
| savio3_xlmem                              | 2     | n0[000-001].savio3                                                                                                            | Intel Xeon Skylake 6130 @ 2.1 GHz | 32             | 1.5 TB        | FDR        | XL Memory                                    | By Node              |
| savio3_xlmem                              | 2     | n0[002-003].savio3                                                                                                            | Intel Xeon Skylake 6130 @ 2.1 GHz | 52             | 1.5 TB        | FDR        | XL Memory                                    | By Node              |
| savio3_gpu                                | 9     | n0[134-138].savio3 n0[158-161].savio3                                                                                         | Intel Xeon Skylake 6130 @ 2.1 GHz | 8              | 96 GB         | FDR        | 4x GTX 2080ti GPU (11 GB GPU memory per GPU) | By Core              |
| savio3_gpu                                | 6     | n0[143-145].savio3 n0[174-176].savio3                                                                                         | Intel Xeon Skylake 6130 @ 2.1 GHz | 32             | 384 GB        | FDR        | 8x TITAN RTX GPU (24 GB GPU memory per GPU)  | By Core              |
| savio3_gpu                                | 1     | n0004.savio3                                                                                                                  | Intel Xeon Silver 4208 @ 2.1 GHz  | 16             | 192 GB        | FDR        | 2x Tesla V100 GPU (32 GB GPU memory per GPU) | By Core              |
| savio3_gpu                                | 1     | n0005.savio3                                                                                                                  | Intel Xeon E5-2623 v3 @ 3.0 GHz   | 8              | 64 GB         | FDR        | 2x Tesla V100 GPU (32 GB GPU memory per GPU) | By Core              |
| savio3_gpu                                | 14    | n0[209-216].savio3 n0[263-264].savio3 n0[273-276].savio3                                                                      | AMD EPYC 7302P (or 7313P)         | 16             | 250 GB        | FDR        | 2x A40 GPU (48 GB GPU memory per GPU)        | By Core              |
| savio3_gpu                                | 2     | n0217.savio3 n0262.savio3                                                                                                     | AMD EPYC 7443P                    | 24             | 250 GB        | FDR        | 2x A40 GPU (48 GB GPU memory per GPU)        | By Core              |
| savio3_gpu                                | 4     | n0[277-281].savio3                                                                                                            | Intel Xeon Gold 6338              | 64             | 512 GB        | FDR        | 4x A40 GPU (48 GB GPU memory per GPU)        | By Core              |
| savio4_htc                                | 108   | n0[000-027].savio4 n0[116-119].savio4 n0[146-165].savio4br>n0[187-242].savio4                                                 | Intel Xeon Gold 6330 @ 2.0 GHz    | 56             | 512 GB        | FDR        |                                              | By Core              |
| savio4_htc                                | 104   | n0[028-059].savio4 n0[064-115].savio4 n0[166-185].savio4                                                                      | Intel Xeon Gold 6330 @ 2.0 GHz    | 56             | 256 GB        | FDR        |                                              | By Core              |
| savio4_gpu                                | 26    | n0[120-145].savio4                                                                                                            | Intel Xeon Gold 6326 @ 2.9 GHz    | 32             | 512 GB        | FDR        | 8x RTX A5000 (24 GB GPU memory per GPU)      | By Core              |
| savio4_gpu                                | 3     | n0[386-388].savio4                                                                                                            | AMD EPYC 9554 @ 3.1 GHz           | 64             | 792 GB        | FDR        | 8x L40 (46 GB GPU memory per GPU)            | By Core              |

### GPU Job Script

https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/#7-gpu-job-script

Scheduler Examples
Here we show some example job scripts that allow for various kinds of parallelization, jobs that use fewer cores than available on a node, GPU jobs, low-priority condo jobs, and long-running FCA jobs.

1. Threaded/OpenMP job script¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Request one node:
#SBATCH --nodes=1
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=4
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./a.out
Here --cpus-per-task should be no more than the number of cores on a Savio node in the partition you request. You may want to experiment with the number of threads for your job to determine the optimal number, as computational speed does not always increase with more threads. Note that if --cpus-per-task is fewer than the number of cores on a node, your job will not make full use of the node. Strictly speaking the --nodes and --ntasks-per-node arguments are optional here because they default to 1.

2. Simple multi-core job script (multiple processes on one node)¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Request one node:
#SBATCH --nodes=1
#
# Specify number of tasks for use case (example):
#SBATCH --ntasks-per-node=20
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
./a.out
This job script would be appropriate for multi-core R, Python, or MATLAB jobs. In the commands that launch your code and/or within your code itself, you can reference the SLURM_NTASKS environment variable to dynamically identify how many tasks (i.e., processing units) are available to you.

Here the number of CPUs used by your code at at any given time should be no more than the number of cores on a Savio node.

For a way to run many individual jobs on one or more nodes (more jobs than cores), see this information on using GNU parallel.

3. MPI job script¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=40
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
module load gcc openmpi
mpirun ./a.out
As noted in the introduction, for partitions in Savio2 and Savio3 scheduled on a per-node basis, you probably want to set the number of tasks to be a multiple of the number of cores per node in that partition, thereby making use of all the cores on the node(s) to which your job is assigned.

This example assumes that each task will use a single core; otherwise there could be resource contention amongst the tasks assigned to a node.

Optimizing MPI on savio4_htc using UCX

savio4 nodes use HDR, under which optimal MPI performance will generally be obtained by using UCX. To do so, make sure to load the ucx module and an openmpi module that uses ucx. You'll need to do this both when building MPI-based software and when running it. At the moment, you'll need to use these (non-default) modules:


module load gcc/11.3.0 ucx/1.14.0 openmpi/5.0.0-ucx
4. Alternative MPI job script¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Number of nodes needed for use case:
#SBATCH --nodes=2
#
# Tasks per node based on number of cores per node (example):
#SBATCH --ntasks-per-node=20
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
module load gcc openmpi
mpirun ./a.out
This alternative explicitly specifies the number of nodes, tasks per node, and CPUs per task rather than simply specifying the number of tasks and having SLURM determine the resources needed. As before, one would generally want the number of tasks per node to equal a multiple of the number of cores on a node, assuming only one CPU per task.

5. Hybrid OpenMP+MPI job script¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Number of nodes needed for use case (example):
#SBATCH --nodes=2
#
# Tasks per node based on --cpus-per-task below and number of cores
# per node (example):
#SBATCH --ntasks-per-node=4
#
# Processors per task needed for use case (example):
#SBATCH --cpus-per-task=5
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
module load gcc openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun ./a.out
Here we request a total of 8 (=2x4) MPI tasks, with 5 cores per task. When using partitions scheduled on a per-node basis, one would generally want to use all the cores on each node (i.e., that --ntasks-per-node multiplied by --cpus-per-task equals the number of cores on a node.

6. Jobs scheduled on a per-core basis (jobs that use fewer cores than available on a node)¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=savio3_htc
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=4
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
./a.out
In the HTC and GPU partitions you are only charged for the actual number of cores used, so the notion of making best use of resources by saturating a node is not relevant.

7. GPU job script¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=savio4_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu and A5000 in savio4_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this should generally be in the form "gpu:A5000:[1-4] with the type included
#SBATCH --gres=gpu:A5000:1
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
./a.out
Requesting a GPU type in savio3_gpu and savio4_gpu

savio3_gpu and savio4_gpu regular condo jobs (those not using the low priority queue) should request the specific type of GPU bought for the condo as detailed here.

savio3_gpu and savio4_gpu regular FCA jobs (those not using the low priority queue) should request either the GTX2080TI, A40, or V100 GPU type, e.g., --gres=gpu:GTX2080TI:1 for savio3_gpu and the A5000 type for savio4_gpu. Also, if requesting an A40 or V100 GPU, note that you also need to specifically specify the QoS via -q a40_gpu3_normal or -q v100_gpu3_normal, respectively.

To help the job scheduler effectively manage the use of GPUs, your job submission script must request multiple CPU cores for each GPU you use. Jobs submitted that do not request sufficient CPUs for every GPU will be rejected by the scheduler. Please see the table here for the ratio of CPU cores to GPUs.

Here’s how to request two CPUs for each GPU: the total of CPUs requested results from multiplying two settings: the number of tasks (--ntasks=) and CPUs per task (--cpus-per-task=).

For instance, in the above example, one GPU was requested via --gres=gpu:1, and the required total of two CPUs was thus requested via the combination of --ntasks=1 and --cpus-per-task=2 . Similarly, if your job script requests four GPUs via --gres=gpu:4, and uses --ntasks=8, it should also include --cpus-per-task=1 in order to request the required total of eight CPUs.

Note that in the --gres=gpu:n specification, n must be between 1 and the number of GPUs on a single node (which is provided here for the various GPU types). This is because the feature is associated with how many GPUs per node to request. If you wish to use more than the number of GPUs available on a node, your --gres=gpu:type:n specification should include how many GPUs to use per node requested. For example, if you wish to use four savio3_gpu A40 GPUs across two nodes (for which there are either 2 or 4 GPUs per node), your job script should include options to the effect of --gres=gpu:A40:2, --nodes=2, --ntasks=4, and --cpus-per-task=8.

8. Long-running jobs (up to 10 days and 4 cores per job)¶

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# QoS: must be savio_long for jobs &gt; 3 days
#SBATCH --qos=savio_long
#
# Partition:
#SBATCH --partition=savio2_htc
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=2
#
# Processors per task:
#SBATCH --cpus-per-task=1
#
# Wall clock limit (7 days in this case):
#SBATCH --time=7-00:00:00
#
## Command(s) to run (example):
./a.out
A given job in the long queue can use no more than 4 cores and a maximum of 10 days. Collectively across the entire Savio cluster, at most 24 cores are available for long-running jobs, so you may find that your job may sit in the queue for a while before it starts.

In the savio2_htc pool you are only charged for the actual number of cores used, so the notion of making best use of resources by saturating a node is not relevant.

9. Low-priority jobs¶
Low-priority jobs can only be run using condo accounts. By default any jobs run in a condo account will use the default QoS (generally savio_normal) if not specified. To use the low-priority queue, you need to specify the low-priority QoS, as follows.


#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Quality of Service:
#SBATCH --qos=savio_lowprio
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run:
echo "hello world"
You may wish to add #SBATCH --requeue as well so that low-priority jobs that are preempted are automatically resubmitted.


### Savio Partitions and QoS

https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-config/

| Partition     | Nodes | Node Features                                           | Nodes shared? | SU/core hour ratio |
|---------------|-------|---------------------------------------------------------|---------------|--------------------|
| savio2        | 163   | savio2 or savio2_c24 or savio2_c28                      | exclusive     | 0.75               |
| savio2_bigmem | 36    | savio2_bigmem or savio2_m128                            | exclusive     | 1.20               |
| savio2_htc    | 20    | savio2_htc                                              | shared        | 1.20               |
| savio2_1080ti | 8     | savio2_1080ti                                           | shared        | 1.67 (3.34 / GPU)  |
| savio2_knl    | 28    | savio2_knl                                              | exclusive     | 0.40               |
| savio3        | 192   | savio3 or savio3_c40                                    | exclusive     | 1.00               |
| savio3_bigmem | 20    | savio3_bigmem or savio3_m384; (savio3_c40 for 40 cores) | exclusive     | 2.67               |
| savio3_htc    | 24    | savio3_htc or savio3_c40                                | shared        | 2.67               |
| savio3_xlmem  | 4     | savio3_xlmem or savio3_c52                              | exclusive     | 4.67               |
| savio3_gpu    | 2     | savio3_gpu (2x V100)                                    | shared        | 3.67               |
| savio3_gpu    | 9     | 4rtx (4x GTX2080TI)                                     | shared        | 3.67               |
| savio3_gpu    | 6     | 8rtx (8x TITAN)                                         | shared        | 3.67               |
| savio3_gpu    | 16    | a40 (2x A40)                                            | shared        | 3.67               |
| savio3_gpu    | 6     | a40 (4x A40)                                            | shared        | 3.67               |
| savio4_htc    | 212   | savio4_m256 or savio4_m512                              | shared        | 3.67               |
| savio4_gpu    | 26    | a5000 (8x A5000)                                        | shared        | 4.67               |
| savio4_gpu    | 3     | L40 (8x L40)                                            | shared        | 4.67               |

Overview of QoS Configurations for Savio¶


| QoS           | Accounts allowed | QoS Limits                                                                   | Partitions   |
|---------------|------------------|------------------------------------------------------------------------------|--------------|
| savio_normal  | FCA*, ICA        | 24 nodes max per job, 72 hour (72:00:00) wallclock limit                     | all          |
| savio_debug   | FCA*, ICA        | 4 nodes max per job, 4 nodes in total, 3 hour (03:00:00) wallclock limit     | all          |
| savio_long    | FCA*, ICA        | 4 cores max per job, 24 cores in total, 10 day (10-00:00:00) wallclock limit | savio2_htc   |
| Condo QoS     | condos           | specific to each condo, see next section                                     | as purchased |
| savio_lowprio | condos, FCA**    | 24 nodes max per job, 72 hour (72:00:00) wallclock limit                     | all          |

QoS Configurations for Savio Condos¶

```
sacctmgr show qos format=Name%24,Priority%8,GrpTRES%22,MinTRES%26
```

## Software modules available

https://docs-research-it.berkeley.edu/services/high-performance-computing/overview/system-overview/

```
module avail

--------------------------------------------- /global/software/rocky-8.x86_64/modfiles/langs ----------------------------------------------
   anaconda3/2024.02-1-11.4 (D)    julia/1.10.2-11.4                    python/3.10.12-gcc-11.4.0        rust/1.70.0-gcc-11.4.0
   anaconda3/2024.10-1-11.4        openjdk/8.0.442_b06                  python/3.11.6-gcc-11.4.0  (D)
   intelpython/3.9.19              openjdk/17.0.8.1_1-gcc-11.4.0 (D)    r-spatial/4.4.0
   java/22.0.1                     perl/5.38.0-gcc-11.4.0               r/4.4.0-gcc-11.4.0

--------------------------------------------- /global/software/rocky-8.x86_64/modfiles/tools ----------------------------------------------
   automake/1.16.5             ffmpeg/6.0              mariadb/10.8.2           proj/9.2.1                          tcl/8.6.12
   awscli/1.29.41              gdal/3.7.3              matlab/r2022a     (D)    protobuf/3.24.3                     texlive/2024
   bazel/6.1.1                 glog/0.6.0              matlab/r2023a            qt/5.15.11                          tmux/3.3a      (D)
   cmake/3.27.7                gmake/4.4.1             matlab/r2024a            rclone/1.63.1                       unixodbc/2.3.4
   code-server/4.12.0          gnuplot/6.0.1           matlab/r2024b            rclone/1.68.1                (D)    vim/9.0.0045
   code-server/4.91.1          gurobi/10.0.0           mercurial/6.4.5          rstudio-server/2024.04.2-764        vim/9.1.0437   (D)
   code-server/4.93.1          imagemagick/7.1.1-11    nano/7.2                 snappy/1.1.10
   code-server/4.99.3 (L,D)    leveldb/1.23            ninja/1.11.1             spack/0.21.1
   eigen/3.4.0                 lmdb/0.9.31             parallel/20220522        sq/0.1.0
   emacs/29.1         (L)      m4/1.4.19               pdsh/2.31         (D)    swig/4.1.1

------------------------------------------- /global/software/rocky-8.x86_64/modfiles/compilers --------------------------------------------
   gcc/10.5.0        gcc/13.2.0                         llvm/17.0.4    nvhpc/23.11-spack        pdsh/2.31
   gcc/11.4.0 (D)    intel-oneapi-compilers/2023.1.0    nvhpc/23.9     nvhpc/23.11       (D)

---------------------------------------------- /global/software/rocky-8.x86_64/modfiles/apps ----------------------------------------------
   ai/jupyter-ai/2.31.4                    bio/ezclermont/0.7.0              bio/qiime2-amplicon/2024.5
   ai/ollama/0.6.8                         bio/fastqc/0.12.1-gcc-11.4.0      bio/raxml-ng/1.2.2
   ai/omni-engineer/0.1.1                  bio/featurecount/2.0.6            bio/relion/4.0.1-gcc-11.4.0
   bio/abricate/1.0.1-jgrg                 bio/gatk/4.4.0.0-gcc-11.4.0       bio/samtools/1.17-gcc-11.4.0
   bio/abricate/1.0.1               (D)    bio/grace/5.1.25-gcc-11.4.0       bio/snippy/4.6.0
   bio/alphafold3/3.0.1                    bio/hisat2/2.2.1-gcc-11.4.0       bio/trimmomatic/0.39-gcc-11.4.0
   bio/bamtools/2.5.2-gcc-11.4.0           bio/hmmer/3.4-gcc-11.4.0          bio/unicycler/0.5.0
   bio/bamutil/1.0.15-gcc-11.4.0           bio/idba/1.1.3-gcc-11.4.0         bio/vcftools/0.1.16-gcc-11.4.0
   bio/bcftools/1.16-gcc-11.4.0            bio/integron_finder/2.0.2         ml/alphafold3/3.0.1
   bio/beast/2.6.4                         bio/kallisto/0.48.0-gcc-11.4.0    ml/pytorch/2.0.1-py3.11.7
   bio/bedtools2/2.31.0-gcc-11.4.0         bio/mafft/7.526                   ml/pytorch/2.3.1-py3.11.7       (D)
   bio/blast-plus/2.13.0-gcc-11.4.0        bio/mlst/2.19.0                   ml/tensorflow/2.14.0-py3.9.0
   bio/blast-plus/2.14.1-gcc-11.4.0 (D)    bio/paup/4.0a                     ml/tensorflow/2.15.0-py3.10.0   (D)
   bio/bowtie2/2.5.1-gcc-11.4.0            bio/phylotool/u20.04              ms/ambertools/2022
   bio/bwa-mem2/2.2.1                      bio/picard/3.0.0-gcc-11.4.0       ms/qchem/5.3.2
   bio/bwa/0.7.18                          bio/plink/1.07-gcc-11.4.0         ms/qchem/6.2.1                  (D)
   bio/cellranger/8.0.1                    bio/prodigal/2.6.3-gcc-11.4.0

----------------------------------------------------- /global/software/site/modfiles ------------------------------------------------------
   mathematica/13.0.1    mathematica/14.1.0 (D)

------------------------------------------ /global/home/groups/consultsw/rocky-8.x86_64/modfiles ------------------------------------------
   GTDB-Tk/2.4.1    checkm/1.2.3          iqtree/3.0.0     nextflow/23.10.0        r-brms/2.22.0    tmux/2.9a
   STAR/2.7.11b     clonalframeml/1.13    jags/4.3.2       nextflow/24.10.4 (D)    raxml/8.2.12     tmux/3.1b
   angsd/0.94       cutadapt/4.9          kraken2/2.1.4    py-cupy/13.4.1          spades/4.1.0     tmux/3.1c
   beastx/10.5.0    fastp/0.23.4          ncurses/6.1      quast/5.2.0             tmux/2.8         tmux/3.2a

  Where:
   L:  Module is loaded
   D:  Default Module

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```

```
module spider

----------------------------------------------------------------------------------------------------------------------------------------
The following is a list of the modules and extensions currently available:
----------------------------------------------------------------------------------------------------------------------------------------
  GTDB-Tk: GTDB-Tk/2.4.1

  STAR: STAR/2.7.11b

  adios: adios/1.13.1

  ai/jupyter-ai: ai/jupyter-ai/2.31.4

  ai/ollama: ai/ollama/0.6.8

  ai/omni-engineer: ai/omni-engineer/0.1.1

  anaconda3: anaconda3/2024.02-1-11.4, anaconda3/2024.10-1-11.4

  angsd: angsd/0.94

  ant: ant/1.14.0

  antlr: antlr/2.7.7

  automake: automake/1.16.5

  awscli: awscli/1.29.41

  bamutil: bamutil/1.0.15

  bazel: bazel/6.1.1

  beastx: beastx/10.5.0

  bedtools2: bedtools2/2.31.0

  berkeleygw: berkeleygw/3.1.0

  bio/abricate: bio/abricate/1.0.1-jgrg, bio/abricate/1.0.1

  bio/alphafold3: bio/alphafold3/3.0.1

  bio/bamtools: bio/bamtools/2.5.2-gcc-11.4.0

  bio/bamutil: bio/bamutil/1.0.15-gcc-11.4.0

  bio/bcftools: bio/bcftools/1.16-gcc-11.4.0

  bio/beast: bio/beast/2.6.4

  bio/bedtools2: bio/bedtools2/2.31.0-gcc-11.4.0

  bio/blast-plus: bio/blast-plus/2.13.0-gcc-11.4.0, bio/blast-plus/2.14.1-gcc-11.4.0

  bio/bowtie2: bio/bowtie2/2.5.1-gcc-11.4.0

  bio/bwa: bio/bwa/0.7.18

  bio/bwa-mem2: bio/bwa-mem2/2.2.1

  bio/cellranger: bio/cellranger/8.0.1

  bio/ezclermont: bio/ezclermont/0.7.0

  bio/fastqc: bio/fastqc/0.12.1-gcc-11.4.0

  bio/featurecount: bio/featurecount/2.0.6

  bio/gatk: bio/gatk/4.4.0.0-gcc-11.4.0

  bio/grace: bio/grace/5.1.25-gcc-11.4.0

  bio/hisat2: bio/hisat2/2.2.1-gcc-11.4.0

  bio/hmmer: bio/hmmer/3.4-gcc-11.4.0

  bio/idba: bio/idba/1.1.3-gcc-11.4.0

  bio/integron_finder: bio/integron_finder/2.0.2

  bio/kallisto: bio/kallisto/0.48.0-gcc-11.4.0

  bio/mafft: bio/mafft/7.526

  bio/mlst: bio/mlst/2.19.0

  bio/paup: bio/paup/4.0a

  bio/phylotool: bio/phylotool/u20.04

  bio/picard: bio/picard/3.0.0-gcc-11.4.0

  bio/plink: bio/plink/1.07-gcc-11.4.0

  bio/prodigal: bio/prodigal/2.6.3-gcc-11.4.0

  bio/qiime2-amplicon: bio/qiime2-amplicon/2024.5

  bio/raxml-ng: bio/raxml-ng/1.2.2

  bio/relion: bio/relion/4.0.1-gcc-11.4.0

  bio/samtools: bio/samtools/1.17-gcc-11.4.0

  bio/snippy: bio/snippy/4.6.0

  bio/trimmomatic: bio/trimmomatic/0.39-gcc-11.4.0

  bio/unicycler: bio/unicycler/0.5.0

  bio/vcftools: bio/vcftools/0.1.16-gcc-11.4.0

  blast-plus: blast-plus/2.13.0, blast-plus/2.14.1

  boost: boost/1.83.0

  checkm: checkm/1.2.3

  clonalframeml: clonalframeml/1.13

  cmake: cmake/3.27.7

  code-server: code-server/4.12.0, code-server/4.91.1, code-server/4.93.1, code-server/4.99.3

  cp2k: cp2k/2023.2-cpu, cp2k/2023.2

  cuda: cuda/11.8.0, cuda/12.2.1, cuda/12.6.0

  cudnn: cudnn/8.7.0.84-11.8, cudnn/8.9.0-12.2.1

  cutadapt: cutadapt/4.9

  eigen: eigen/3.4.0

  emacs: emacs/29.1

  fastp: fastp/0.23.4

  ffmpeg: ffmpeg/6.0

  fftw: fftw/2.1.5, fftw/3.3.10

  gatk: gatk/4.4.0.0

  gcc: gcc/10.5.0, gcc/11.4.0, gcc/13.2.0

  gdal: gdal/3.7.3

  geos: geos/3.12.0

  glog: glog/0.6.0

  gmake: gmake/4.4.1

  gmt: gmt/6.4.0

  gnuplot: gnuplot/6.0.1

  grace: grace/5.1.25

  gromacs: gromacs/2023.3

  gsl: gsl/2.7.1

  gurobi: gurobi/10.0.0

  hdf5: hdf5/1.12.2, hdf5/1.14.3

  hisat2: hisat2/2.2.1

  hmmer: hmmer/3.4

  hpl: hpl/2.3

  imagemagick: imagemagick/7.1.1-11

  intel-oneapi-compilers: intel-oneapi-compilers/2023.1.0

  intel-oneapi-mkl: intel-oneapi-mkl/2023.2.0

  intel-oneapi-mpi: intel-oneapi-mpi/2021.10.0

  intel-oneapi-tbb: intel-oneapi-tbb/2021.10.0

  intelpython: intelpython/3.9.19

  ior: ior/3.3.0

  iqtree: iqtree/3.0.0

  jags: jags/4.3.2

  java: java/22.0.1

  julia: julia/1.10.2-11.4

  kraken2: kraken2/2.1.4

  lammps: lammps/20230802

  leveldb: leveldb/1.23

  llvm: llvm/17.0.4

  lmdb: lmdb/0.9.31

  m4: m4/1.4.19

  mariadb: mariadb/10.8.2

  mathematica: mathematica/13.0.1, mathematica/14.1.0

  matlab: matlab/r2022a, matlab/r2023a, matlab/r2024a, matlab/r2024b

  mercurial: mercurial/6.4.5

  ml/alphafold3: ml/alphafold3/3.0.1

  ml/pytorch: ml/pytorch/2.0.1-py3.11.7, ml/pytorch/2.3.1-py3.11.7

  ml/tensorflow: ml/tensorflow/2.14.0-py3.9.0, ml/tensorflow/2.15.0-py3.10.0

  ms/ambertools: ms/ambertools/2022

  ms/qchem: ms/qchem/5.3.2, ms/qchem/6.2.1

  mumps: mumps/5.5.1

  nano: nano/7.2

  ncl: ncl/6.6.2

  nco: nco/5.1.6

  ncurses: ncurses/6.1

  ncview: ncview/2.1.9

  netcdf-c: netcdf-c/4.9.2

  netcdf-fortran: netcdf-fortran/4.6.1

  netlib-lapack: netlib-lapack/3.11.0

  netlib-scalapack: netlib-scalapack/2.2.0

  nextflow: nextflow/23.10.0, nextflow/24.10.4

  ninja: ninja/1.11.1

  nvhpc: nvhpc/23.9, nvhpc/23.11-spack, nvhpc/23.11

  openblas: openblas/0.3.24

  openjdk: openjdk/8.0.442_b06, openjdk/17.0.8.1_1-gcc-11.4.0

  openmpi: openmpi/4.1.3, openmpi/4.1.6-internal-testing, openmpi/4.1.6

  orca: orca/5.0.3, orca/6.0.1

  osu-micro-benchmarks: osu-micro-benchmarks/7.3

  parallel: parallel/20220522

  parallel-netcdf: parallel-netcdf/1.12.3

  pdsh: pdsh/2.31

  perl: perl/5.38.0-gcc-11.4.0, perl/5.38.0

  perl-xml-libxml: perl-xml-libxml/2.0201

  petsc: petsc/3.20.1-complex, petsc/3.20.1

  plink: plink/1.07

  proj: proj/9.2.1

  protobuf: protobuf/3.24.3

  py-cupy: py-cupy/13.4.1

  python: python/3.10.12-gcc-11.4.0, python/3.11.6-gcc-11.4.0

  qt: qt/5.15.11

  quantum-espresso: quantum-espresso/7.2

  quast: quast/5.2.0

  r: r/4.4.0-gcc-11.4.0

  r-brms: r-brms/2.22.0

  r-ggplot2: r-ggplot2/3.4.2

  r-spatial: r-spatial/4.4.0

  raxml: raxml/8.2.12

  rclone: rclone/1.63.1, rclone/1.68.1

  relion: relion/4.0.1

  rstudio-server: rstudio-server/2024.04.2-764

  rust: rust/1.70.0-gcc-11.4.0

  slepc: slepc/3.20.0

  snappy: snappy/1.1.10

  spack: spack/0.21.1

  spades: spades/4.1.0

  sq: sq/0.1.0

  sqlite: sqlite/3.43.2

  swig: swig/4.1.1

  tcl: tcl/8.6.12

  texlive: texlive/2024

  tmux: tmux/2.8, tmux/2.9a, tmux/3.1b, tmux/3.1c, tmux/3.2a, tmux/3.3a

  ucx: ucx/1.14.1

  udunits: udunits/2.2.28

  unixodbc: unixodbc/2.3.4

  vasp: vasp/6.4.1-cpu-intel, vasp/6.4.1-cpu, vasp/6.4.1-gpu-general, vasp/6.4.1-gpu

  vim: vim/9.0.0045, vim/9.1.0437

  wannier90: wannier90/3.1.0

  xerces-c: xerces-c/3.2.4

----------------------------------------------------------------------------------------------------------------------------------------

To learn more about a package execute:

   $ module spider Foo

where "Foo" is the name of a module.

To find detailed information about a particular package you
must specify the version if there is more than one version:

   $ module spider Foo/11.1

----------------------------------------------------------------------------------------------------------------------------------------

```
