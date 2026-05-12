---
title: Causality and Machine Learning
subtitle: NORA Summer Research School 2026
layout: default
---

# Causality and Machine Learning

This site contains the materials for the session about **Causal Estimation and Inference (day 3)** for the *NORA Summer Research School 2026* on **Causality and Machine Learning**, given by [Johan de Aguas](https://johandh2o.github.io).

## Recommended readings 
- Rohrer, J. M. (2018). [Thinking clearly about correlations and causation: Graphical causal models for observational data](https://journals.sagepub.com/doi/10.1177/2515245917745629). Advances in Methods and Practices in Psychological Science, 1(1), 27–42. doi:10.1177/2515245917745629
- Funk, M. J., Westreich, D., Wiesen, C., Stürmer, T., Brookhart, M. A., & Davidian, M. (2011). [Doubly robust estimation of causal effects](https://academic.oup.com/aje/article/173/7/761/103691). American Journal of Epidemiology, 173(7), 761–767. doi:10.1093/aje/kwq439
- **Optional**: Hernán, M. A., & Robins, J. M. (2020). [Causal inference: What if](https://static1.squarespace.com/static/675db8b0dd37046447128f5f/t/677676888e31cc50c2c33877/1735816881944/hernanrobins_WhatIf_2jan25.pdf) (**chapters: 1,3,7,12,13,15**). Chapman & Hall / CRC.

## Advance readings 

## Available material

- [Day 3: Estimation for causal effects](day3.html)

## Computational setup

For Day 3, students should have Python installed and be able to run Jupyter notebooks. The recommended minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib statsmodels jupyterlab
jupyter lab
```

On Windows, replace the activation line with:

```bash
.venv\Scripts\activate
```

## Repository structure

```text
docs/
├── _config.yml
├── index.md
├── day3.md
├── assets/
│   └── css/
│       └── style.scss
└── notebooks/
    └── day3_estimation_1.ipynb
```
