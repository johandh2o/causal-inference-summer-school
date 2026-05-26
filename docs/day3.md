---
title: "Day 3: Estimation for Causal Effects"
layout: default
---

## Advanced readings

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). [Estimation of regression coefficients when some regressors are not always observed](https://www.jstor.org/stable/2290910). Journal of the American Statistical Association, 89(427), 846–866. doi:10.1080/01621459.1994.10476818
- van der Laan, M. J., & Rubin, D. (2006). [Targeted maximum likelihood learning](https://biostats.bepress.com/cgi/viewcontent.cgi?article=1214&context=ucbbiostat). The International Journal of Biostatistics, 2(1), Article 11. doi:10.2202/1557-4679.1043
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). [Double/debiased machine learning for treatment and structural parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401). The Econometrics Journal, 21(1), C1–C68. doi:10.1111/ectj.12097


## Tutorials, online books and resources

- [Applied causal inference](https://appliedcausalinference.github.io/aci_book/01-intro-to-causality.html)
- [Causal inference for the brave and true](https://matheusfacure.github.io/python-causality-handbook/landing-page.html)
- [Causal inference and machine learning](https://www.causalmlbook.com)
- [Machine learning-based causal inference](https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/)
- [Applied causal inference powered by ML and AI](https://github.com/CausalAIBook)
- [Awesome causal inference 1](https://github.com/matteocourthoud/awesome-causal-inference)
- [Awesome causal inference 2](https://github.com/imirzadeh/awesome-causal-inference)
- [Awesome causality](https://shubhanshu.com/awesome-causality/)
- [An illustrated guide to TMLE](https://www.khstats.com/blog/tmle/tutorial)
- [TMLE for binary outcome](https://migariane.github.io/TMLE.nb.html)
- [Theory behind TMLE (CrossValidated)](https://stats.stackexchange.com/questions/442569/theory-behind-targeted-maximum-likelihood-estimation-tmle)
- [EconML and CausalML](https://causal-machine-learning.github.io/kdd2021-tutorial/)
- [The basics of double/debiased machine learning](https://docs.doubleml.org/stable/guide/basics.html)




# Day 3: Estimation for Causal Effects

This day focuses on the transition from identification formulas to estimators. The goal is to understand how causal estimands can be estimated from observed data using outcome regression, inverse probability weighting, doubly robust estimators, and flexible machine-learning nuisance estimation.

## Learning objectives

By the end of this session, students should be able to:

1. Define the average treatment effect under standard causal assumptions.
2. Implement the g-computation estimator.
3. Implement inverse probability weighting.
4. Explain why doubly robust estimators combine outcome and treatment models.
5. Diagnose practical problems such as positivity violations, misspecification, and unstable weights.
6. Use cross-fitting to reduce overfitting bias when nuisance functions are estimated flexibly.

## Prerequisites

Students should be familiar with:

- Random variables and conditional expectations.
- Conditional independence.
- Basic regression models.
- The backdoor adjustment formula.
- Python basics, including `numpy`, `pandas`, and `scikit-learn`.

## Schedule

| Time | Topic | Format |
|---:|---|---|
| 09:00--09:30 | Recap: identification versus estimation | Lecture |
| 09:30--10:30 | Outcome-regression and g-computation | Lecture + coding |
| 10:30--10:45 | Break |  |
| 10:45--11:45 | Propensity scores and inverse probability weighting | Lecture + coding |
| 11:45--12:30 | Doubly robust estimation | Lecture |
| 12:30--13:30 | Lunch |  |
| 13:30--14:45 | Cross-fitting and flexible nuisance estimation | Coding lab |
| 14:45--15:00 | Break |  |
| 15:00--16:00 | Simulation study and diagnostics | Coding lab |
| 16:00--16:30 | Discussion and wrap-up | Group discussion |

## Main estimand

For a binary treatment `A`, baseline covariates `W`, and outcome `Y`, the central estimand is the average treatment effect

\[
\psi = \mathbb{E}\{Y(1)-Y(0)\}.
\]

Under consistency, positivity, and conditional exchangeability,

\[
\psi
=
\mathbb{E}\left[Q(1,W)-Q(0,W)\right],
\qquad
Q(a,w)=\mathbb{E}(Y\mid A=a,W=w).
\]

## Notebook

The coding lab is available here:

- [Download/open the Day 3 estimation notebook](notebooks/day3_estimation.ipynb)

The notebook contains:

- a simulated observational dataset;
- a g-computation estimator;
- an inverse-probability-weighted estimator;
- an augmented inverse-probability-weighted estimator;
- a short cross-fitting exercise;
- diagnostic plots for propensity scores and weights.

## Recommended readings

### Short and accessible

Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapters 11--13.

### Intermediate

Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Sections on propensity scores and inverse probability weighting.

### Complete reference

van der Laan, M. J., & Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer.

## Exercises

1. Modify the treatment assignment mechanism to create weaker positivity.
2. Compare the variance of the IPW estimator under mild and severe positivity violations.
3. Fit misspecified outcome and propensity models. Which estimators remain approximately centered?
4. Replace logistic regression with random forests for nuisance estimation.
5. Implement two-fold cross-fitting for the AIPW estimator.

## Expected outputs

At the end of the lab, students should produce:

- a table comparing estimators;
- a plot of the estimated propensity-score distribution by treatment arm;
- a plot showing the impact of weight truncation;
- a short interpretation of the simulation results.
