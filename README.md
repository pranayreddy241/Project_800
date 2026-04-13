## Key Results (Comparison Across Methods)

The following results highlight the behavior of each mitigation strategy on the POPE benchmark.

### Popular Split

| Method        | Accuracy | Precision | Recall | F1 Score | Yes Ratio |
|--------------|----------|----------|--------|----------|-----------|
| Baseline     | 85.90%   | 93.11%   | 77.53% | 84.61%   | 41.63%    |
| OPERA-lite   | 85.70%   | 93.57%   | 76.67% | 84.28%   | 40.97%    |
| VCD-lite     | 79.60%   | 96.25%   | 61.60% | 75.12%   | 32.00%    |
| Combined     | 79.13%   | 96.59%   | 60.40% | 74.32%   | 31.27%    |

---

### Adversarial Split

| Method        | Accuracy | Precision | Recall | F1 Score | Yes Ratio |
|--------------|----------|----------|--------|----------|-----------|
| Baseline     | 83.83%   | 88.71%   | 77.53% | 82.75%   | 43.70%    |
| OPERA-lite   | 83.60%   | 89.01%   | 76.67% | 82.38%   | 43.07%    |
| VCD-lite     | 78.30%   | 92.49%   | 61.60% | 73.95%   | 33.30%    |
| Combined     | 77.80%   | 92.64%   | 60.40% | 73.12%   | 32.60%    |

---

## What This Shows

- **VCD-lite**
  - Highest precision (strong hallucination suppression)
  - Significant drop in recall (misses real objects)
  - Much lower Yes Ratio → overly conservative

- **OPERA-lite**
  - Maintains a balanced trade-off
  - Very close to baseline F1
  - Slight reduction in hallucination without large recall loss

- **Combined (VCD + OPERA)**
  - Does not improve performance
  - Further reduces recall
  - Becomes overly conservative
  - Confirms that these methods are not directly composable

---

## Key Insight

> Applying multiple hallucination mitigation strategies together does not necessarily improve performance.

Instead, these methods:
- reinforce the same bias (predicting “no”)
- reduce recall more than they improve precision
- lead to **non-additive behavior**

This motivates moving toward **adaptive mitigation strategies** instead of static combinations.
