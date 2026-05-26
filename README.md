# Causal Inference Course: Day 3 Materials

This repository contains a minimal GitHub Pages website and a Jupyter notebook for **Day 3: Estimation and Causal Inference**.

The Day 3 session focuses on the workflow

```text
causal question
→ causal estimand
→ identification assumptions
→ identifying functional
→ estimators
→ diagnostics and uncertainty
```

## Repository structure

```text
docs/       GitHub Pages website files
notebooks/  Jupyter notebook for the Day 3 coding session
data/       Placeholder folder for optional datasets
```

## Computational setup

Create the course environment with:

```bash
conda env create -f environment.yml
conda activate causal-course
jupyter lab
```

Then open:

```text
notebooks/day3_identification_estimation.ipynb
```

## Website

The website is designed to be published with GitHub Pages from the `docs/` folder.

In GitHub:

```text
Settings → Pages → Deploy from a branch → main → /docs
```
