#!/bin/bash
#SBATCH --job-name=baseline_pope
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --account=user
#SBATCH --output=/mmfs1/home/pbairedd/vlm_project/logs/baseline_%j.out

export PATH=/cm/shared/apps/slurm/current/bin:$PATH
module purge
module load cuda12.4/toolkit/12.4.1
module load python39
unset PYTHONHOME
unset PYTHONPATH
source /mmfs1/home/pbairedd/vlm_project/vlm_env/bin/activate

MODEL=/mmfs1/home/pbairedd/vlm_project/models/llava-1.5-7b
DATA=/mmfs1/home/pbairedd/vlm_project/data
RESULTS=/mmfs1/home/pbairedd/vlm_project/results
LLAVA=/mmfs1/home/pbairedd/vlm_project/LLaVA

echo "==== BASELINE EVALUATION ===="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python3 --version)"
echo "Model: $MODEL"

# Add LLaVA to Python path
export PYTHONPATH=$LLAVA:$PYTHONPATH

for SPLIT in random popular adversarial; do
    echo ""
    echo "--- POPE-${SPLIT} ---"

    python3 $LLAVA/llava/eval/model_vqa_loader.py \
        --model-path $MODEL \
        --question-file $DATA/coco_pope_${SPLIT}.json \
        --image-folder $DATA/val2014 \
        --answers-file $RESULTS/baseline_pope_${SPLIT}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

    python3 - << PYEOF
import json

questions = {str(item["question_id"]): item
             for item in json.load(open("$DATA/coco_pope_${SPLIT}.json"))}
answers = [json.loads(l) for l in open("$RESULTS/baseline_pope_${SPLIT}.jsonl")]

TP = FP = TN = FN = yes_count = 0
for ans in answers:
    qid = str(ans["question_id"])
    pred = ans["text"].strip().lower()
    gt = questions[qid]["label"].strip().lower()
    pred_yes = pred.startswith("yes")
    gt_yes = (gt == "yes")
    yes_count += int(pred_yes)
    if pred_yes and gt_yes:       TP += 1
    elif pred_yes and not gt_yes: FP += 1
    elif not pred_yes and gt_yes: FN += 1
    else:                         TN += 1

total = TP + FP + TN + FN
acc  = (TP + TN) / total
prec = TP / (TP + FP) if (TP + FP) > 0 else 0
rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
yes_r = yes_count / total

print(f"=== BASELINE POPE-${SPLIT} ===")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Yes Ratio: {yes_r*100:.2f}%")
print(f"TP:{TP} FP:{FP} TN:{TN} FN:{FN} Total:{total}")

import os
os.makedirs("$RESULTS", exist_ok=True)
with open("$RESULTS/baseline_pope_${SPLIT}_score.txt", "w") as f:
    f.write(f"Method: baseline\nSplit: ${SPLIT}\n")
    f.write(f"Accuracy: {acc*100:.4f}\nPrecision: {prec*100:.4f}\n")
    f.write(f"Recall: {rec*100:.4f}\nF1: {f1*100:.4f}\n")
    f.write(f"Yes_ratio: {yes_r*100:.4f}\n")
    f.write(f"TP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}\n")
PYEOF

done

echo ""
echo "==== ALL SPLITS DONE ===="
ls -lh $RESULTS/
