#!/bin/bash
#SBATCH --job-name=baseline_hf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/home/pbairedd/vlm_project/logs/baseline_hf_%j.out

module purge
module load cuda12.4/toolkit/12.4.1
module load python39
unset PYTHONHOME
unset PYTHONPATH

source /mmfs1/home/pbairedd/vlm_project/vlm_env/bin/activate

python - << 'PYEOF'
import transformers, tokenizers
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
from transformers import AutoProcessor
AutoProcessor.from_pretrained("/mmfs1/home/pbairedd/vlm_project/models/llava-1.5-7b")
print("PROCESSOR_OK")
PYEOF

python /mmfs1/home/pbairedd/vlm_project/scripts/run_pope_hf.py
