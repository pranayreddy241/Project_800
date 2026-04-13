# Hallucination Mitigation in Vision-Language Models

This repository contains my graduate research work on object hallucination mitigation in vision-language models, with a focus on training-free inference-time methods evaluated on the POPE benchmark.

The project uses **LLaVA-1.5-7B** as the main model and studies how different mitigation strategies behave in binary object-existence queries. A central question in this work is whether existing hallucination mitigation methods can be combined effectively, or whether they interfere with each other.

## Project Summary

Vision-language models often generate confident answers that are not supported by the image. In object-centric settings, this usually appears as a false "yes" response to a question asking whether an object is present.

In this project, I built an evaluation pipeline around the **POPE** benchmark and tested the following conditions:

- **Baseline**
- **VCD-lite**
- **OPERA-lite**
- **Combined VCD-lite + OPERA-lite**
- **Simple ensemble methods**

The experiments showed a clear precision-recall trade-off. In particular, methods that reduce affirmative bias can also suppress correct positive predictions. One of the main findings from this work is that these mitigation strategies are not directly composable in a naive way.

## Main Finding

From the experiments, I observed that:

- **VCD-lite** improves precision by suppressing unsupported affirmative predictions
- but it also causes a noticeable drop in recall
- **OPERA-lite** gives a more balanced trade-off and stays closer to baseline F1
- directly combining these methods does not produce additive gains
- instead, the combination tends to make the model overly conservative

This suggests that hallucination mitigation may need to be adaptive and confidence-aware, rather than relying on fixed combinations of multiple conservative methods.

## Benchmark

The main evaluation benchmark used in this project is **POPE**, which tests object hallucination through binary yes/no questions.

The standard POPE splits used here are:

- Random
- Popular
- Adversarial

These splits help reveal different types of hallucination, especially those driven by object popularity and co-occurrence priors.

## Repository Structure

```text
.
├── scripts/              # experiment and evaluation scripts
├── data/                 # small data files or setup notes
├── results/              # metrics and sample outputs
├── logs/                 # lightweight logs
├── LLaVA/                # supporting code used in experiments
├── OPERA/                # supporting code used in experiments
├── VCD/                  # supporting code used in experiments
├── setup_env.sh          # environment setup helper
├── .gitignore
├── requirements.txt
└── README.md
