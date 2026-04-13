#!/bin/bash
#SBATCH --job-name=download_llava
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=user
#SBATCH --output=/mmfs1/home/pbairedd/vlm_project/logs/download_%j.out

export PATH=/cm/shared/apps/slurm/current/bin:$PATH
module purge
module load cuda12.4/toolkit/12.4.1
module load python39
unset PYTHONHOME
unset PYTHONPATH
source /mmfs1/home/pbairedd/vlm_project/vlm_env/bin/activate

echo "Job started on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python3 --version)"

python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os

save_dir = "/mmfs1/home/pbairedd/vlm_project/models/llava-1.5-7b"
os.makedirs(save_dir, exist_ok=True)
print(f"Downloading to {save_dir}...")

snapshot_download(
    repo_id="llava-hf/llava-1.5-7b-hf",
    local_dir=save_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"]
)
print("Download complete.")
PYEOF
