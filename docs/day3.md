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
- Implement plug-in estimators.
- Implement AIPW estimators.
- Explain how doubly robust estimators combine outcome and treatment models.
- Use cross-fitting to reduce overfitting bias when nuisance functions are estimated flexibly.

## Schedule and notebooks

| Time | Topic | Format |
|---:|---|---|
| 10:05--10:45 | Introduction, formalisms and causal effects | [Lecture](https://github.com/johandh2o/causal-inference-summer-school/blob/main/docs/slides/day3.pdf) |
| 10:45--11:00 | Common causal estimands | [Marimo lab](https://molab.marimo.io/github/johandh2o/causal-inference-summer-school/blob/main/docs/marimo/1_causal_estimands_visual_simulation_marimo.py/wasm) |
| 11:00--11:15 | The causal roadmap, identification | Lecture  |
| 11:15--11:45 | Estimation: parametric and ML plug-in | Lecture |
| 11:45--12:00 | Estimation: parametric and ML plug-in | [Marimo lab](https://molab.marimo.io/github/johandh2o/causal-inference-summer-school/blob/main/docs/marimo/2_estimation_s_t_parametric_marimo.py/wasm) |
| 12:00--13:00 | Lunch | |
| 13:05--13:45 | AIPW and double robustness | Lecture  |
| 13:45--14:15 | State-of-the-art and NN architectures  | Lecture  |
| 14:15--14:30 | AIPW | [Marimo lab](https://molab.marimo.io/github/johandh2o/causal-inference-summer-school/blob/main/docs/marimo/3b_aipw_double_robustness_marimo.py/wasm)  |
| 14:30--15:00 | Influence function, DML (and TMLE*)  | Lecture |
| 15:00--15:30 | AIPW, DML (and TMLE*) | [Marimo lab](https://molab.marimo.io/github/johandh2o/causal-inference-summer-school/blob/main/docs/marimo/4_dml_tmle_marimo.py/wasm)  | 
| 15:30--16:00 | Heterogeneous treatment effect  |   |
| 16:00--16:30 | Heterogeneous treatment effect  | Marimo lab  |

## Advanced readings

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). [Estimation of regression coefficients when some regressors are not always observed](https://www.jstor.org/stable/2290910). *Journal of the American Statistical Association*, 89(427), 846--866. doi:10.1080/01621459.1994.10476818
- van der Laan, M. J., & Rubin, D. (2006). [Targeted maximum likelihood learning](https://biostats.bepress.com/cgi/viewcontent.cgi?article=1214&context=ucbbiostat). *The International Journal of Biostatistics*, 2(1), Article 11. doi:10.2202/1557-4679.1043
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). [Double/debiased machine learning for treatment and structural parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401). *The Econometrics Journal*, 21(1), C1--C68. doi:10.1111/ectj.12097

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
