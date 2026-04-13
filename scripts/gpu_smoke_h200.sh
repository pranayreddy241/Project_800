#!/bin/bash
#SBATCH -J gpu_smoke_h200
#SBATCH -p gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_smoke_h200_%j.out

module load cuda12.4/toolkit/12.4.1
module load python39

cd /mmfs1/home/pbairedd/vlm_project
source /mmfs1/home/pbairedd/vlm_env/bin/activate

echo "HOSTNAME: $(hostname)"
echo "PWD: $(pwd)"
which python
python --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
