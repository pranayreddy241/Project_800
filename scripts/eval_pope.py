import json
import sys

question_file = sys.argv[1]
answer_file = sys.argv[2]

with open(question_file) as f:
    questions = [json.loads(line) for line in f if line.strip()]
with open(answer_file) as f:
    answers = [json.loads(line) for line in f if line.strip()]

qmap = {str(q["question_id"]): q for q in questions}

TP = FP = TN = FN = yes_count = 0

for ans in answers:
    qid = str(ans["question_id"])
    pred_text = ans.get("text", "").strip().lower()
    gt = qmap[qid]["label"].strip().lower()

    pred_yes = pred_text.startswith("yes")
    gt_yes = (gt == "yes")

    yes_count += int(pred_yes)

    if pred_yes and gt_yes:
        TP += 1
    elif pred_yes and not gt_yes:
        FP += 1
    elif not pred_yes and gt_yes:
        FN += 1
    else:
        TN += 1

total = TP + FP + TN + FN
acc = (TP + TN) / total if total else 0
prec = TP / (TP + FP) if (TP + FP) else 0
rec = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
yes_ratio = yes_count / total if total else 0

print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Yes Ratio: {yes_ratio*100:.2f}%")
print(f"TP:{TP} FP:{FP} TN:{TN} FN:{FN} Total:{total}")
