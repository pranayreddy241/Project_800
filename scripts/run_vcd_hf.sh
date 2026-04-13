#!/bin/bash
#SBATCH --job-name=vcd_hf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/home/pbairedd/vlm_project/logs/vcd_hf_%j.out

module purge
module load cuda12.4/toolkit/12.4.1
module load python39
unset PYTHONHOME
unset PYTHONPATH

source /mmfs1/home/pbairedd/vlm_project/vlm_env/bin/activate

python /mmfs1/home/pbairedd/vlm_project/scripts/run_pope_vcd.py
