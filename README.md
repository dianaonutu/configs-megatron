# Pre-training LLMs with Megatron-LM on Snellius using Virtual Environment

This guide is for Snellius users to quickly set up a virtual environment for pretraining LLMs using Nvidia's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## Environment setup
The tasks below require a specific set of libraries and versions installed on Snellius. The module combination listed here is compatible
with **PyTorch 2.6** (PyTorch 2.5.1 also works) and **CUDA 12.6**, and is optimal for tasks such as installing FlashAttention fast. The virtual environment 
used for pre-training with Megatron-LM requires the following modules loaded:
```
module purge # Clear all previously loaded modules
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0    # Includes CUDA 12.6.0 dependencies
module load cuDNN/9.5.0.50-CUDA-12.6.0                # Includes CUDA 12.6.0 dependencies
```
### Module Descriptions
- **Python 3.12.3**: Python interpreter and standard libraries.
- **CUDA 12.6.0**: Required for GPU acceleration.
- **NCCL 2.22.3**: Enables efficient multi-GPU communication.
- **cuDNN 9.5.0.50**: Provides optimized GPU kernels for deep learning.

## Table of Contents

1. [Create Virtual Environment](#create-virtual-environment)  
2. [Tokenize & Preprocess Data](#tokenize--preprocess-data)  
   2.1 [Download FineWeb Dataset](#download-fineweb-dataset)  
   2.2 [Tokenization & Preprocessing](#tokenization--preprocessing)
3. [Pretraining a GPT Model](#pretraining-a-gpt-model)
      - [Configuration](#configuration)
      - [Option 1: Start Pretraining](#option-1-start-pretraining)
      - [Option 2: Run Tests and Debug on a Single GPU](#option-2-run-tests-and-debug-on-a-single-gpu)
4. [Acknowledgments](#acknowledgments)

## Create Virtual Environment 
**Estimated time:** 10 minutes

Clone this repository and navigate into its directory:
```
git clone https://github.com/dianaonutu/Megatron-LM-Snellius-venv.git
cd Megatron-LM-Snellius-venv
```
Ensure the required modules are loaded (see [Environment Setup](#environment-setup)). Then, create the virtual environment:
```
python -m venv megatron-venv
```
Allocate a compute node, activate the virtual environment, and install packages.
> **Note**: Login nodes are intended only for lightweight tasks (e.g., job submission and monitoring). Installing libraries can be
resource-intensive and may be terminated automatically if it consumes too many resources on the login nodes. Always allocate
a compute node for installing libraries.
> 
**Estimated time:** 7 minutes
```
salloc -n 16 -t 30:00
source megatron-venv/bin/activate
./install.sh
```
Once finished, exit node allocation:
```
exit
```

## Tokenize & Preprocess Data
**Estimated time:** 45 minutes

Ensure required modules are loaded (see [Environment Setup](#environment-setup)). Allocate a compute node, activate your virtual environment, and set the path to your own project directory:
```
salloc -p gpu_h100 --gpus-per-node 1 -t 1:00:00
source megatron-venv/bin/activate
export PROJECT_SPACE=/projects/0/prjsXXXX   # Replace with your project ID
```

### Download FineWeb Dataset
Download the 10B shard from HuggingFace's
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

**Estimated time:** 8 minutes
```
python load_fineweb.py
```

### Tokenization & Preprocessing
Set environment variables for input and output paths and worker count.
```
export FINEWEB_INPUT=$PROJECT_SPACE/datasets/FineWeb/raw/fineweb-10BT.jsonl
export FINEWEB_OUTPUT=$PROJECT_SPACE/datasets/FineWeb/fineweb-10BT
export WORKERS=${SLURM_CPUS_PER_TASK:-16}
```
Run the tokenizer. 

**Estimated time:** 34 minutes
```
python Megatron-LM/tools/preprocess_data.py \
    --input $FINEWEB_INPUT \
    --output-prefix $FINEWEB_OUTPUT \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --append-eod \
    --log-interval 10000 \
    --workers $WORKERS
```

The output is an index file (`.idx`) and the binary (`.bin`) of the tokenizer model.

Exit the node when done: 

```
exit
```

## Pretraining a GPT model
### Configuration
Clone the Megatron-LM repository, if you haven't done it already. This codebase is based on commit [`8a9e8644`](https://github.com/NVIDIA/Megatron-LM/commit/8a9e8644).
```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 8a9e8644
cd ..
```
Make scripts executable. You only need to run this command once.
```
chmod +x launch.sh
chmod +x train-gpt-venv.job
```
Set your project and virtual environment paths in `train-gpt-venv.job`:
```
export PROJECT_SPACE=/projects/0/prjsXXXX   # Replace with your project ID
export VENV_PATH=/path/to/your/venv         # Replace with actual venv path
```
**Optional**: To enable Weights & Biases logging, add the following to the `OUTPUT_ARGS` section in `train-gpt-venv.job`:
```
--wandb_exp_name <your_experiment_name>
--wandb_project <your_project_name>
--wandb_save_dir <your_wandb_dir>
```
To enable Weights & Biases logging, you must also set your API key as an environment variable. This authenticates your session with Weights & Biases.
Replace <your_wandb_key> with your actual Weights & Biases API key, which you can find at https://wandb.ai/authorize after logging in.
```
export WANDB_API_KEY=<your_wandb_key>
```

### Option 1: Start Pretraining
Submit the job.
```
sbatch train-gpt-venv.job
```

### Option 2: Run Tests and Debug on a Single GPU
This option is meant for quickly testing your setup and code or debugging issues on a single GPU.

Allocate a single GPU:
```
salloc -p gpu_h100 --gpus-per-node 1 -t 1:00:00
export SLURM_CPUS_PER_TASK=1
export SLURM_NTASKS=1
```
Now, you can start your testing session within the `salloc` environment by running:
```
./train-gpt-venv.job
```

## Acknowledgments
Thanks to [@spyysalo](https://github.com/spyysalo)'s original [LUMI Megatron-LM guide](https://github.com/spyysalo/lumi-fineweb-replication) and [@tvosch](https://github.com/tvosch)'s [Snellius guide](https://github.com/SURF-ML/Megatron-LM-Snellius) that helped in creating this one. 
