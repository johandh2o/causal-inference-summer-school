---
title: Causality and Machine Learning
subtitle: NORA Summer Research School 2026
layout: default
---

# NORA Summer Research School 2026: Causality and Machine Learning:

## Course description

The course is part of the [NORA Summer Research School 2026](https://www.nora.ai/nora-research-school/education-programs/summer-school/summer-school-2026/). 

**Causality and Machine Learning** introduces PhD candidates and researchers to foundational concepts and modern methods at the intersection of causality and machine learning. It is designed to bridge traditional causal inference, including potential outcomes, structural causal models, graphical reasoning, identification, and semiparametric estimation, with modern machine learning methodologies for flexible nuisance estimation, policy learning, and data-driven causal analysis.

A central goal of the course is to help students formulate causal questions precisely, understand the assumptions required for identification, and implement estimation methods using modern computational tools. The course emphasizes both mathematical clarity and practical implementation

**Format**: Lectures, guided examples, hands-on coding sessions, group discussion, and final project.

**Credits**: 5

**Instructors**

- [Prof. Fabio Massimo Zennaro](https://www4.uib.no/en/find-employees/fabio-massimo.zennaro), Dept. of Informatics, University of Bergen.
- [Prof. Pekka Parviainen](https://www4.uib.no/en/find-employees/pekka.parviainen), Dept. of Informatics, University of Bergen.
- [Prof. Johan Pensar](https://www.mn.uio.no/math/english/people/aca/johanpen/), Statistics and Data Science (Dept. of Mathematics), University of Oslo.
- [Johan de Aguas](https://johandh2o.github.io), Statistics and Data Science (Dept. of Mathematics), University of Oslo.

## Learning objectives

By the end of the course, students should be able to:

- Represent probabilistic systems using graphical models, including nodes, edges, paths, parents, ancestors, descendants, and Markov properties.
- Use d-separation and Bayesian network factorization to reason about conditional independence, prediction, and probabilistic modeling.
- Interpret directed acyclic graphs as causal models and relate them to structural causal models, interventions, and potential outcomes.
- Apply graphical criteria to reason about causal identification, including adjustment, backdoor paths, front-door intuition, selection bias, and transportability.
- Formulate causal estimands and connect them to observed-data estimands that can be estimated from data.
- Use statistical and machine learning methods for causal estimation, including outcome regression, IPW, AIPW, TMLE, DML, nuisance estimation, and cross-fitting.
- Explain the goals, assumptions, and limitations of causal discovery from observational and interventional data.
- Compare basic structure learning approaches, including constraint-based, score-based, and hybrid methods, while accounting for uncertainty and Markov equivalence.
- Critically evaluate causal inference papers, empirical analyses, and simulation studies.
- Communicate a causal analysis clearly in written and oral form.

## Syllabus

| Day | Topic | 
|---:|---|
| 1 | Probability theory and random variables | 
| 1 | Graphs and conditional independence | 
| 1 | Bayesian networks |
| 1 | d-separation and I-equivalences | 
| 2 | Pearl's causal hierarchy | 
| 2 | Structural causal models | 
| 2 | Identifiability of causal queries | 
| 2 | Backdoor adjustment, confounders and selection bias | 
| 3 | Potential outcomes and assumptions for identification | 
| 3 | Causal estimands: ATE, ATT, CATE and policy effects | 
| 3 | Classical estimation: outcome regression and IPW | 
| 3 | Semiparametric estimation: AIPW, DML and TMLE | 
| 4 | Where do causal models come from? Expert knowledge, RCTs, observational data | 
| 4 | Constraint-based structure learning | 
| 4 | Score-based structure learning | 
| 4 | Learning beyond equivalence classes | 

## Recommended readings 

Here is a list of references (including papers and textbooks) covering the topics presented at the summer school. You are invited to have a look at these sources to get a foundation of the ideas we will work on during the week.

### Day 1: Probabilistic graphical models

- Koller, D., & Friedman, N. (2009). [Probabilistic Graphical Models: Principles and Techniques](http://mcb111.org/w06/KollerFriedman.pdf) (**chapters 1-3**). MIT Press.

### Day 2: Causal models

- Peters, J., Janzing, D., & Scholkopf, B. (2017). [Elements of causal inference: foundations and learning algorithms](https://library.oapen.org/bitstream/handle/20.500.12657/26040/1/11283.pdf) (**chapters: 1-3**). MIT press.
- **Optional**: Pearl, J. (2015). [Trygve Haavelmo and the emergence of causal calculus](https://ftp.cs.ucla.edu/pub/stat_ser/r391.pdf). Econometric Theory, 31(1), 152-179
- **Optional**: Pearl, J. (2018). [Theoretical impediments to machine learning with seven sparks from the causal revolution](https://arxiv.org/pdf/1801.04016). arXiv preprint arXiv:1801.04016

### Day 3: Estimation and causal inference

- Cinelli, C., Forney, A., & Pearl, J. (2024). [A crash course in good and bad controls](https://journals.sagepub.com/doi/full/10.1177/00491241221099552). Sociological Methods & Research, 53(3), 1071-1104. doi: 10.1177/0049124122109955
- Funk, M. J., Westreich, D., Wiesen, C., Stürmer, T., Brookhart, M. A., & Davidian, M. (2011). [Doubly robust estimation of causal effects](https://academic.oup.com/aje/article/173/7/761/103691). American Journal of Epidemiology, 173(7), 761–767. doi:10.1093/aje/kwq439
- **Optional**: Hernán, M. A., & Robins, J. M. (2020). [Causal inference: What if](https://static1.squarespace.com/static/675db8b0dd37046447128f5f/t/677676888e31cc50c2c33877/1735816881944/hernanrobins_WhatIf_2jan25.pdf) (**chapters: 1,3,7,12,13,15**). Chapman & Hall / CRC.

### Day 4: Causal discovery

- Koller, D., & Friedman, N. (2009). [Probabilistic Graphical Models: Principles and Techniques](http://mcb111.org/w06/KollerFriedman.pdf) (**chapters 16-18,21**). MIT Press.







## Advanced readings

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). [Estimation of regression coefficients when some regressors are not always observed](https://www.jstor.org/stable/2290910). Journal of the American Statistical Association, 89(427), 846–866. doi:10.1080/01621459.1994.10476818
- van der Laan, M. J., & Rubin, D. (2006). [Targeted maximum likelihood learning](https://biostats.bepress.com/cgi/viewcontent.cgi?article=1214&context=ucbbiostat). The International Journal of Biostatistics, 2(1), Article 11. doi:10.2202/1557-4679.1043
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). [Double/debiased machine learning for treatment and structural parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401). The Econometrics Journal, 21(1), C1–C68. doi:10.1111/ectj.12097

## Available materials

- [Day 3: Estimation for causal effects](day3.html)

## Projects

Hey

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

## Computational setup

For day 3, students should have Python installed and be able to run Jupyter notebooks. The recommended minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib statsmodels jupyterlab marimo pgmpy
jupyter lab
```
