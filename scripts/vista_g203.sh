#!/bin/bash
#SBATCH -J vista_rand
#SBATCH -p gpu-l40s
#SBATCH -w g203
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/home/pbairedd/vlm_project/logs/vista_rand_%j.out

module load cuda12.4/toolkit/12.4.1
module load python39

cd /mmfs1/home/pbairedd/vlm_project || exit 1
source /mmfs1/home/pbairedd/vlm_project/vlm_env/bin/activate

echo "HOSTNAME: $(hostname)"
nvidia-smi

python /mmfs1/home/pbairedd/vlm_project/scripts/run_pope_vista.py \
  --data /mmfs1/home/pbairedd/vlm_project/data/coco_pope_random.json \
  --image-folder /mmfs1/home/pbairedd/vlm_project/data/val2014 \
  --output /mmfs1/home/pbairedd/vlm_project/results/vista_random.jsonl
