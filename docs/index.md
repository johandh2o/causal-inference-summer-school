---
title: Causality and Machine Learning
subtitle: NORA Summer Research School 2026
layout: default
---

# NORA Summer School 2026: Causality and Machine Learning

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

| Day | Topic | Lecturer |
|---:|---|
| 1 | Probability theory and random variables | Prof. Johan Pensar |
| 1 | Graphs and conditional independence | Prof. Johan Pensar |
| 1 | Bayesian networks | Prof. Johan Pensar |
| 1 | d-separation and I-equivalences | Prof. Johan Pensar |
| 2 | Pearl's causal hierarchy | Prof. Fabio Massimo Zennaro |
| 2 | Structural causal models | Prof. Fabio Massimo Zennaro |
| 2 | Identifiability of causal queries | Prof. Fabio Massimo Zennaro |
| 2 | Backdoor adjustment, confounders and selection bias | Prof. Fabio Massimo Zennaro |
| 3 | Potential outcomes and assumptions for identification | Johan de Aguas |
| 3 | Causal estimands: ATE, ATT, CATE and policy effects | Johan de Aguas |
| 3 | Classical estimation: outcome regression and IPW | Johan de Aguas |
| 3 | Semiparametric estimation: AIPW, DML and TMLE | Johan de Aguas |
| 4 | Where do causal models come from? Expert knowledge, RCTs, observational data | Prof. Pekka Parviainen | 
| 4 | Constraint-based structure learning | Prof. Pekka Parviainen | 
| 4 | Score-based structure learning | Prof. Pekka Parviainen | 
| 4 | Learning beyond equivalence classes | Prof. Pekka Parviainen | 

## Recommended readings 

Here is a list of references (including papers and textbooks) covering the topics presented at the summer school. You are invited to have a look at these sources to get a foundation of the ideas we will work on during the week.

#### Day 1: Probabilistic graphical models

- Koller, D., & Friedman, N. (2009). [Probabilistic Graphical Models: Principles and Techniques](http://mcb111.org/w06/KollerFriedman.pdf) (**chapters 1-3**). MIT Press.

#### Day 2: Causal models

- Peters, J., Janzing, D., & Scholkopf, B. (2017). [Elements of causal inference: foundations and learning algorithms](https://library.oapen.org/bitstream/handle/20.500.12657/26040/1/11283.pdf) (**chapters: 1-3**). MIT press.
- **Optional**: Pearl, J. (2015). [Trygve Haavelmo and the emergence of causal calculus](https://ftp.cs.ucla.edu/pub/stat_ser/r391.pdf). Econometric Theory, 31(1), 152-179
- **Optional**: Pearl, J. (2018). [Theoretical impediments to machine learning with seven sparks from the causal revolution](https://arxiv.org/pdf/1801.04016). arXiv preprint arXiv:1801.04016

#### Day 3: Estimation and causal inference

- Cinelli, C., Forney, A., & Pearl, J. (2024). [A crash course in good and bad controls](https://journals.sagepub.com/doi/full/10.1177/00491241221099552). Sociological Methods & Research, 53(3), 1071-1104. doi: 10.1177/0049124122109955
- Funk, M. J., Westreich, D., Wiesen, C., Stürmer, T., Brookhart, M. A., & Davidian, M. (2011). [Doubly robust estimation of causal effects](https://academic.oup.com/aje/article/173/7/761/103691). American Journal of Epidemiology, 173(7), 761–767. doi:10.1093/aje/kwq439
- **Optional**: Hernán, M. A., & Robins, J. M. (2020). [Causal inference: What if](https://static1.squarespace.com/static/675db8b0dd37046447128f5f/t/677676888e31cc50c2c33877/1735816881944/hernanrobins_WhatIf_2jan25.pdf) (**chapters: 1,3,7,12,13,15**). Chapman & Hall / CRC.

#### Day 4: Causal discovery

- Koller, D., & Friedman, N. (2009). [Probabilistic Graphical Models: Principles and Techniques](http://mcb111.org/w06/KollerFriedman.pdf) (**chapters 16-18,21**). MIT Press.

## Computational setup

Working Python 3.11 or 3.12 environment. We will use the following libraries:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib statsmodels jupyterlab marimo pgmpy
```

## Final project

To complete the summer school, students will be asked to develop a project in the field of **causality and machine learning**. Depending on the scope of the project, students may work individually or in pairs; however, each student must submit an individual final report.

We envision three main types of projects, which may focus **on one or several of the topics covered across the different days of the course**.

#### Option A: Benchmark-based comparison of causal methods

Students may work with a publicly available  benchmark dataset. The goal is to compare different causal learning or estimation methods discussed during the course. Students are encouraged to discuss practical issues such as model misspecification, nuisance estimation, overlap, finite-sample behavior, and sensitivity to tuning choices. Benchmark data include for instance:

- For causal discovery: [Bayesian network repository](https://www.bnlearn.com/bnrepository/).
- For benchmark estimation: [ACIC data challenge 2019](https://sites.google.com/view/acic2019datachallenge/data-challenge) and [IHDP / Jobs datasets](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks).
- For physical data testbed [Causal Chambers](https://github.com/juangamella/causal-chamber).

#### Option B: Bring your own data and formulate a causal analysis problem

Students may bring their own dataset, or use a dataset from their field of interest, and formulate a causal question that can be studied using the tools from the course. The project should clearly define the learning goal. For instance, for estimation this includes specifying treatment, outcome, covariates, target population, and causal estimand. The main emphasis should be on translating a substantive question into a precise causal problem, rather than only on obtaining numerical results.

#### Option C: Reproduce, implement, or extend a causal machine learning paper

Students may choose a paper on causal graphical models, causal inference or causal machine learning,  and work on reproducing, implementing, or extending some of its main ideas. This may involve coding the proposed method, reproducing a simulation study, applying the method to a new dataset, or critically analyzing the assumptions and theoretical results. Students may also focus on a theoretical or mathematical problem inspired by the paper

Each project should include a final report containing:
- A clear statement of the causal question or methodological problem.
- A discussion of the assumptions for identification, estimation, learning and interpretation.
- A description of the methods used.
- Simulations, theoretical analysis, or a critical discussion, depending on the selected option.
- A short conclusion summarizing what was learned.

A LaTeX template for the final report will be made available on the course website. Information about more detailed evaluation criteria and the policy for the use of LLMs will be provided during the course.

## Instructions for Registering the Credits

During the summer school, attendance will be registered. After the successful submission of the final project, participants in the course will be registered in the UiB system so that their credits will be available nationally through FSWeb.

For international students, a formal certificate granting the credits can be requested by contacting the organizers <fabio.zennaro@uib.no> or UiB administration.

## Contact

Questions about the course, readings, computational setup, or final project can be directed to: <fabio.zennaro@uib.no>

## Available materials for Day 3

- [Day 3: Estimation and causal inference]({{ "/day3.html" | relative_url }})