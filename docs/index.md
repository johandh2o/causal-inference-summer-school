---
title: Causality and Machine Learning
subtitle: NORA Summer Research School 2026
layout: default
---

# Causality and Machine Learning

This site contains the materials for the session on **Causal Estimation and Inference** (day 3) for the *NORA Summer Research School 2026* on **Causality and Machine Learning*.

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
