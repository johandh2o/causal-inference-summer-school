---
title: Causal Inference Summer School
layout: default
---

# Causal Inference Summer School

This site contains the materials for the causal inference summer school.

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
    └── day3_estimation.ipynb
```
