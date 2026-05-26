---
title: "Day 3: Estimation and causal inference"
layout: default
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


## Advanced readings

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). [Estimation of regression coefficients when some regressors are not always observed](https://www.jstor.org/stable/2290910). Journal of the American Statistical Association, 89(427), 846–866. doi:10.1080/01621459.1994.10476818
- van der Laan, M. J., & Rubin, D. (2006). [Targeted maximum likelihood learning](https://biostats.bepress.com/cgi/viewcontent.cgi?article=1214&context=ucbbiostat). The International Journal of Biostatistics, 2(1), Article 11. doi:10.2202/1557-4679.1043
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). [Double/debiased machine learning for treatment and structural parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401). The Econometrics Journal, 21(1), C1–C68. doi:10.1111/ectj.12097


## Notebooks

The coding lab is available here:

- [Download/open the Day 3 estimation notebook](notebooks/day3_estimation.ipynb)


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