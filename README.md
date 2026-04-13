# Hallucination Mitigation in Vision-Language Models

This repository contains my graduate research work on understanding and mitigating object hallucination in vision-language models (VLMs), using **LLaVA-1.5-7B** and the **POPE benchmark**.

The focus of this work is not just applying mitigation methods, but studying how they behave, interact, and whether they can be combined effectively.

---

## Problem Context

Vision-language models often generate responses that are **linguistically plausible but not visually grounded**. 

In object-centric settings, this typically appears as a false affirmative:

> “Is there a bicycle in the image?” → *Yes* (when no bicycle exists)

This type of hallucination is especially problematic because:
- it appears confident
- it is hard to detect downstream
- it propagates into decision-making systems

---

## Goal of This Work

Instead of proposing a new model, this project focuses on:

- building a **controlled evaluation pipeline**
- understanding how hallucination emerges in binary queries
- evaluating **training-free mitigation strategies**
- analyzing trade-offs between precision, recall, and bias
- studying whether mitigation methods are **composable**

---

## Methods Explored

All experiments are conducted using **LLaVA-1.5-7B** on the POPE benchmark under five conditions:

### 1. Baseline
Standard model inference without any mitigation.

### 2. VCD-lite
A simplified version of Visual Contrastive Decoding:
- compares predictions on original vs blurred image
- suppresses unstable affirmative responses

### 3. OPERA-lite
A constrained decoding setup inspired by OPERA:
- deterministic decoding
- prompt-level grounding constraints
- reduced output drift

### 4. Combined (VCD-lite + OPERA-lite)
Direct composition of both mitigation strategies.

### 5. Ensemble Methods
Rule-based fusion across baseline, VCD-lite, and OPERA-lite outputs.

---

## Benchmark: POPE

The evaluation is performed on the **POPE benchmark**, which tests hallucination through binary yes/no object-existence queries.

Three splits are used:

- **Random** → general behavior  
- **Popular** → tests bias toward frequent objects  
- **Adversarial** → stresses co-occurrence priors (most challenging)

---

## Key Results

### Popular Split
- Accuracy: 85.90%
- Precision: 93.11%
- Recall: 77.53%
- F1 Score: 84.61%
- Yes Ratio: 41.63%

### Adversarial Split
- Accuracy: 83.83%
- Precision: 88.71%
- Recall: 77.53%
- F1 Score: 82.75%
- Yes Ratio: 43.70%

---

## Core Observation

Across all experiments, a consistent pattern emerges:

> **Reducing hallucination tends to reduce recall.**

Methods that suppress false positives (hallucinations) also suppress true positives.

This leads to a **precision–recall trade-off**, rather than a clear improvement.

---

## Main Finding

The most important result from this work is:

> **Training-free hallucination mitigation methods are not directly composable.**

Specifically:

- VCD-lite reduces hallucination but becomes overly conservative  
- OPERA-lite maintains a better balance  
- combining them **does not improve performance**
- instead, it amplifies conservativeness

This suggests that these methods:
- do not address independent error sources
- instead act on overlapping failure modes

---

## Interpretation

Rather than being complementary, these mitigation methods tend to push the model in the same direction:

> “Say *no* more often”

This explains why combining them fails to produce additive gains.

---

## Next Direction

Based on these findings, the next step is:

### Adaptive Mitigation

Instead of applying mitigation uniformly:

- use model confidence (e.g., logit margin)
- apply contrastive suppression only when needed
- preserve recall for high-confidence predictions

This shifts the problem from:
> “Which method is best?”

to:
> “When should each method be applied?”

---

## Repository Structure

```text
.
├── scripts/              # inference and evaluation pipelines
├── results/              # metrics and prediction outputs
├── data/                 # lightweight setup files
├── logs/                 # experiment logs
├── LLaVA/                # supporting code
├── OPERA/                # supporting code
├── VCD/                  # supporting code
├── setup_env.sh
├── requirements.txt
└── README.md
