import os
import json
import csv
from collections import defaultdict

BASE = "/home/pbairedd/vlm_project"
DATA_DIR = f"{BASE}/data"
RES_DIR = f"{BASE}/results"
OUT_DIR = f"{RES_DIR}/ensemble_search"
os.makedirs(OUT_DIR, exist_ok=True)

splits = ["random", "popular", "adversarial"]

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def norm(x):
    x = str(x).strip().lower()
    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return "no"

def metrics(gt_rows, pred_rows):
    qmap = {str(r["question_id"]): r for r in gt_rows}
    pmap = {str(r["question_id"]): r for r in pred_rows}

    TP = FP = TN = FN = yes_count = 0
    for qid, gt in qmap.items():
        pred = norm(pmap[qid]["text"])
        gold = norm(gt["label"])

        pred_yes = pred == "yes"
        gold_yes = gold == "yes"
        yes_count += int(pred_yes)

        if pred_yes and gold_yes:
            TP += 1
        elif pred_yes and not gold_yes:
            FP += 1
        elif not pred_yes and gold_yes:
            FN += 1
        else:
            TN += 1

    total = TP + FP + TN + FN
    acc = (TP + TN) / total if total else 0
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    yes_ratio = yes_count / total if total else 0
    return {
        "accuracy": acc * 100,
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "yes_ratio": yes_ratio * 100,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN, "total": total
    }

def build_strategy(name, b, o, v):
    """
    b=baseline, o=opera, v=vcd
    """
    if name == "majority_vote":
        votes = [b, o, v]
        return "yes" if votes.count("yes") >= 2 else "no"

    if name == "baseline_opera_trust":
        # If baseline and opera agree, trust them; else fall back to baseline
        return b if b == o else b

    if name == "opera_primary_vcd_veto":
        # OPERA is strong; VCD only vetoes risky positives
        if o == "yes" and v == "no":
            return "no"
        return o

    if name == "baseline_primary_vcd_double_veto":
        # Flip baseline yes only if both other methods reject it
        if b == "yes" and o == "no" and v == "no":
            return "no"
        return b

    if name == "agreement_then_conservative":
        # Use agreement if present; if all disagree pattern resolves conservatively
        if b == o:
            return b
        if b == v:
            return b
        if o == v:
            return o
        return "no"

    if name == "opera_baseline_hybrid":
        # Trust OPERA when it matches baseline; otherwise prefer OPERA unless VCD vetoes a yes
        if o == b:
            return o
        if o == "yes" and v == "no":
            return "no"
        return o

    raise ValueError(f"Unknown strategy {name}")

strategies = [
    "majority_vote",
    "baseline_opera_trust",
    "opera_primary_vcd_veto",
    "baseline_primary_vcd_double_veto",
    "agreement_then_conservative",
    "opera_baseline_hybrid",
]

summary_rows = []

for split in splits:
    gt = load_jsonl(f"{DATA_DIR}/coco_pope_{split}.json")
    baseline = load_jsonl(f"{RES_DIR}/baseline_pope_{split}.jsonl")
    opera = load_jsonl(f"{RES_DIR}/opera_pope_{split}.jsonl")
    vcd = load_jsonl(f"{RES_DIR}/vcd_pope_{split}.jsonl")

    bmap = {str(r["question_id"]): r for r in baseline}
    omap = {str(r["question_id"]): r for r in opera}
    vmap = {str(r["question_id"]): r for r in vcd}

    for strat in strategies:
        out_rows = []
        for q in gt:
            qid = str(q["question_id"])
            b = norm(bmap[qid]["text"])
            o = norm(omap[qid]["text"])
            v = norm(vmap[qid]["text"])

            pred = build_strategy(strat, b, o, v)
            out_rows.append({
                "question_id": q["question_id"],
                "image": q["image"],
                "question": q["text"],
                "label": q["label"],
                "text": pred,
                "baseline": b,
                "opera": o,
                "vcd": v,
                "strategy": strat
            })

        out_path = f"{OUT_DIR}/{strat}_{split}.jsonl"
        with open(out_path, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")

        m = metrics(gt, out_rows)
        summary_rows.append({
            "method": strat,
            "split": split,
            **m
        })

csv_path = f"{OUT_DIR}/ensemble_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["method","split","accuracy","precision","recall","f1","yes_ratio","TP","FP","TN","FN","total"]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("Saved:", csv_path)

# Print best by average F1
by_method = defaultdict(list)
for r in summary_rows:
    by_method[r["method"]].append(r["f1"])

ranked = sorted(
    [(m, sum(vals)/len(vals)) for m, vals in by_method.items()],
    key=lambda x: x[1],
    reverse=True
)

print("\nAverage F1 by strategy:")
for m, avgf1 in ranked:
    print(f"{m}: {avgf1:.2f}")

best = ranked[0][0]
print(f"\nBEST_STRATEGY={best}")
