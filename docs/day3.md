---
layout: default
title: "Day 3: Estimation and causal inference"
permalink: /day3.html
---

# Day 3: Estimation and causal inference

This day focuses on the transition from identification formulas to estimators. The goal is to understand how causal estimands can be estimated from observed data using outcome regression, inverse probability weighting, doubly robust estimators, and flexible machine-learning nuisance estimation.

This session is given by [Johan de Aguas](https://johandh2o.github.io).

## Learning objectives

By the end of this session, students should be able to:

- Define the average treatment effect under standard causal assumptions.
- Implement the g-computation estimator.
- Implement inverse probability weighting.
- Explain why doubly robust estimators combine outcome and treatment models.
- Diagnose practical problems such as positivity violations, misspecification, and unstable weights.
- Use cross-fitting to reduce overfitting bias when nuisance functions are estimated flexibly.

## Schedule

| Time | Topic | Format |
|---:|---|---|
| 10:00--10:50 | Introduction, formalisms and causal effects | Lecture |
| 10:50--11:00 | 10 minute break | |
| 11:00--11:30 | Common causal estimands | Lab  |
| 11:30--11:50 | The causal workflow, identification | Lecture  |


## Advanced readings

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). [Estimation of regression coefficients when some regressors are not always observed](https://www.jstor.org/stable/2290910). *Journal of the American Statistical Association*, 89(427), 846--866. doi:10.1080/01621459.1994.10476818
- van der Laan, M. J., & Rubin, D. (2006). [Targeted maximum likelihood learning](https://biostats.bepress.com/cgi/viewcontent.cgi?article=1214&context=ucbbiostat). *The International Journal of Biostatistics*, 2(1), Article 11. doi:10.2202/1557-4679.1043
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). [Double/debiased machine learning for treatment and structural parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401). *The Econometrics Journal*, 21(1), C1--C68. doi:10.1111/ectj.12097

## Notebooks

The coding labs are available here:

- [Common causal estimands](https://molab.marimo.io/github/causal-inference-summer-school/blob/main/docs/marimo/causal_estimands_visual_simulation_marimo.py/wasm)
- [Plug-in bias](https://molab.marimo.io/github/causal-inference-summer-school/blob/main/docs/marimo/ate_plugin_first_order_bias_marimo.py/wasm)


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
