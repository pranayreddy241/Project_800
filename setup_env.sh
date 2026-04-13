module purge
module load cuda12.4/toolkit/12.4.1
module load python39
unset PYTHONHOME
unset PYTHONPATH
source ~/vlm_project/vlm_env/bin/activate
echo "Environment ready. Python: $(python3 --version)"
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
