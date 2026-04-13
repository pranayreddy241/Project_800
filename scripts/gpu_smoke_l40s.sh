#!/bin/bash
#SBATCH -J gpu_smoke_l40s
#SBATCH -p gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:05:00
#SBATCH --output=/mmfs1/home/pbairedd/vlm_project/logs/gpu_smoke_l40s_%j.out

module load cuda12.4/toolkit/12.4.1
module load python39

cd /mmfs1/home/pbairedd/vlm_project || exit 1
source /mmfs1/home/pbairedd/vlm_project/vlm_env/bin/activate

echo "HOSTNAME: $(hostname)"
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
