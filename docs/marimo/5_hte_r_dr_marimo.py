# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "scikit-learn"
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # These are the only packages used in the notebook.
    # marimo builds the app, numpy simulates arrays, pandas stores tables,
    # and matplotlib makes the plots.
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import KFold

    return KFold, LinearRegression, LogisticRegression, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HTE estimation: outcome plug-in vs R-learner vs DR-learner

    **Goal.** Estimate heterogeneous treatment effects using three CATE learners:

    \[
    \widehat\tau_{\rm plug}(w),
    \qquad
    \widehat\tau_{\rm R}(w),
    \qquad
    \widehat\tau_{\rm DR}(w).
    \]

    The observed data are

    \[
    O=(W,A,Y),
    \qquad
    W=(X,G,Z_1,Z_2,Z_3),
    \qquad A\in\{0,1\}.
    \]

    We use the notation from the slides:

    \[
    Q(w,a)=E[Y\mid W=w,A=a],
    \qquad
    \varpi(w)=P(A=1\mid W=w),
    \qquad
    \tau(w)=E[Y^1-Y^0\mid W=w].
    \]

    The lab focuses on two lower-dimensional summaries of the CATE:

    \[
    \tau_G(g)=E\{\tau(W)\mid G=g\},
    \qquad
    \tau_X(x)=E\{\tau(W)\mid X=x\}.
    \]

    The central lesson is that all three learners try to estimate the same CATE function, but they construct the regression target differently:

    - the **outcome plug-in** directly uses \(\widehat Q(W,1)-\widehat Q(W,0)\), with no propensity score;
    - the **R-learner** uses residualized outcome variation;
    - the **DR-learner** first builds a doubly robust pseudo-outcome and then regresses it on covariates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data-generating process

    The **true DGP is fixed throughout the lab**. The sliders below do **not** change the data-generating process. They only change the feature dictionaries used by the fitted learners.

    Treatment assignment follows

    \[
    A\mid W\sim \operatorname{Bernoulli}\{\varpi_\lambda(W)\},
    \]

    where larger \(\lambda\) means more lack of overlap:

    \[
    \operatorname{logit}\{\varpi_\lambda(W)\}
    =\lambda\Big[-0.15+1.00X-1.05Z_1+0.75\sin(Z_2)
    +0.70G+0.60XG-0.65\mathbb{I}(Z_3>0)\Big].
    \]

    Potential outcomes are generated as

    \[
    Y^a=\mu_0(W)+a\tau(W)+\varepsilon,
    \qquad
    \varepsilon\sim N(0,1),
    \]

    with the fixed nonlinear baseline outcome

    \[
    \begin{aligned}
    \mu_0(W)
    &= 1 + 0.50X +0.30Z_3 +0.55G +0.20XG \\
    &\quad + 2\{2.20\sin(2X)+1.50(Z_1^2-1)+1.35Z_2Z_3
    -1.10\cos(2Z_3)+1.35XGZ_1 \\
    &\hspace{2.9cm}+1.00(X^2-1)Z_2-0.90G\sin(2.5Z_1)\}.
    \end{aligned}
    \]

    The true CATE is also fixed:

    \[
    \begin{aligned}
    \tau(W)
    &= 0.80+0.55X+0.65G-0.70XG \\
    &\quad +1.90\{ -0.35(X^2-1)+0.20X^3+0.50(X^2-1)G
    +0.45Z_1-0.30Z_2+0.40GZ_1 \\
    &\hspace{1.7cm}+1.20\sin(1.5X)+0.95Z_1Z_2-0.65\cos(Z_3)
    +0.85X\mathbb{I}(Z_3>0)+0.90XGZ_2 \\
    &\hspace{1.7cm}+0.95G\sin(2X)-0.65(X^2-1)Z_2+0.65G(Z_1^2-1)\}.
    \end{aligned}
    \]

    Thus, the true DGP always contains the important variables \(Z_1,Z_2\), nonlinearities, and interactions. The feature-misspecification lever controls whether the fitted learners are allowed to use those terms.

    Compared with the previous version, the coefficients of the omitted \(Z_1,Z_2\)-driven terms are only moderately larger. The point is not to make the example pathological, but to make the three estimators separate visually under the relevant levers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the learner misspecification lever works

    The notebook uses one feature-misspecification lever, \(\rho\in[0,1]\).
    The same \(\rho\) controls the features used for
    \(\widehat\mu(W)\) in the R-learner, \(\widehat Q(W,A)\) in the plug-in and DR-learner,
    and the final \(\widehat\tau(W)\) regressions of the R-learner and DR-learner:

    \[
    \widehat f_\rho(W)
    =(1-\rho)\widehat f_{\rm correct}(W)
    +\rho\widehat f_{\rm wrong}(W).
    \]

    Therefore, increasing \(\rho\) simultaneously makes the nuisance learners and the final CATE learners rely more heavily on the deliberately incomplete feature dictionaries. This keeps the lab simple while still accentuating differences among the estimators.

    The propensity score has a separate misspecification lever \(\zeta\). At \(\zeta=0\), the propensity is estimated with a correctly specified logistic model. At \(\zeta=1\), it is estimated with a misspecified logistic model that ignores \(Z_1\), \(\sin(Z_2)\), and \(\mathbb{I}(Z_3>0)\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    hte_n = mo.ui.slider(
        start=600,
        stop=6000,
        step=300,
        value=2400,
        label="sample size n",
    )
    hte_feature_rho = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.10,
        value=0.00,
        label="outcome/feature misspecification ρ: 0 = correct, 1 = wrong",
    )
    hte_prop_zeta = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.10,
        value=0.00,
        label="propensity misspecification ζ: 0 = correct, 1 = wrong",
    )
    hte_overlap = mo.ui.slider(
        start=0.25,
        stop=2.50,
        step=0.25,
        value=0.50,
        label="lack of overlap λ",
    )
    hte_wrong_tau_basis = mo.ui.dropdown(
        options=["moderately wrong τ basis", "sparse wrong τ basis"],
        value="moderately wrong τ basis",
        label="wrong final τ learner used when ρ=1",
    )
    hte_seed = mo.ui.number(value=2026, start=1, stop=999999, label="random seed")
    hte_run_button = mo.ui.run_button(label="Run HTE experiment")

    mo.vstack([
        mo.md(r"""
        ## Choose the experiment

        Suggested first run: \(n=2400\), \(\rho=0\), \(\zeta=0\), \(\lambda=0.5\). This is the clean reference setting: correct nuisance features, correct final CATE features, and correctly specified estimated propensity.
        Then increase \(\rho\) to misspecify both the outcome/CATE feature dictionaries, increase \(\zeta\) to misspecify the propensity, and increase \(\lambda\) to weaken overlap.

        """),
        mo.hstack([hte_n, hte_overlap]),
        mo.hstack([hte_feature_rho, hte_prop_zeta]),
        mo.hstack([hte_wrong_tau_basis, hte_seed]),
        hte_run_button,
    ])
    return (
        hte_feature_rho,
        hte_n,
        hte_overlap,
        hte_prop_zeta,
        hte_run_button,
        hte_seed,
        hte_wrong_tau_basis,
    )


@app.cell(hide_code=True)
def _(KFold, LinearRegression, LogisticRegression, np, pd, plt):
    DGP_RHO = 2.0

    def expit(x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x)))

    def logit(p):
        p = np.clip(np.asarray(p), 1e-5, 1.0 - 1e-5)
        return np.log(p / (1.0 - p))

    def as_vector(value, n):
        if np.isscalar(value):
            return np.full(int(n), float(value))
        return np.asarray(value, dtype=float)

    def covariate_arrays(df, x_override=None, g_override=None):
        n = len(df)
        X = as_vector(df["X"].to_numpy() if x_override is None else x_override, n)
        G = as_vector(df["G"].to_numpy() if g_override is None else g_override, n)
        Z1 = df["Z1"].to_numpy()
        Z2 = df["Z2"].to_numpy()
        Z3 = df["Z3"].to_numpy()
        return X, G, Z1, Z2, Z3

    def true_propensity_from_covariates(X, G, Z1, Z2, Z3, overlap_lambda):
        eta = float(overlap_lambda) * (
            -0.15
            + 1.00 * X
            - 1.05 * Z1
            + 0.75 * np.sin(Z2)
            + 0.70 * G
            + 0.60 * X * G
            - 0.65 * (Z3 > 0)
        )
        return np.clip(expit(eta), 0.01, 0.99)

    def mu0_true_from_arrays(X, G, Z1, Z2, Z3):
        X2 = X**2 - 1.0
        mu0_linear = 1.00 + 0.50 * X + 0.30 * Z3 + 0.55 * G + 0.20 * X * G
        mu0_extra = (
            2.20 * np.sin(2.0 * X)
            + 1.50 * (Z1**2 - 1.0)
            + 1.35 * Z2 * Z3
            - 1.10 * np.cos(2.0 * Z3)
            + 1.35 * X * G * Z1
            + 1.00 * X2 * Z2
            - 0.90 * G * np.sin(2.5 * Z1)
        )
        return mu0_linear + DGP_RHO * mu0_extra

    def tau_true_from_arrays(X, G, Z1, Z2, Z3):
        X2 = X**2 - 1.0
        tau_base = 0.80 + 0.55 * X + 0.65 * G - 0.70 * X * G
        tau_extra = (
            -0.35 * X2
            + 0.20 * X**3
            + 0.50 * X2 * G
            + 0.45 * Z1
            - 0.30 * Z2
            + 0.40 * G * Z1
            + 1.20 * np.sin(1.5 * X)
            + 0.95 * Z1 * Z2
            - 0.65 * np.cos(Z3)
            + 0.85 * X * (Z3 > 0)
            + 0.90 * X * G * Z2
            + 0.95 * G * np.sin(2.0 * X)
            - 0.65 * X2 * Z2
            + 0.65 * G * (Z1**2 - 1.0)
        )
        return tau_base + 0.95 * DGP_RHO * tau_extra

    def generate_data(n, overlap_lambda=1.0, seed=2026):
        rng = np.random.default_rng(int(seed))
        n = int(n)

        X = rng.normal(0.0, 1.0, size=n)
        G = rng.binomial(1, 0.45, size=n)
        Z1 = 0.30 + 0.55 * X + 0.45 * G + rng.normal(0.0, 0.90, size=n)
        Z2 = -0.20 + 0.45 * X * (1 - G) - 0.35 * G + rng.normal(0.0, 0.95, size=n)
        Z3 = 0.20 * X - 0.25 * Z1 + rng.normal(0.0, 1.00, size=n)

        varpi = true_propensity_from_covariates(X, G, Z1, Z2, Z3, overlap_lambda)
        A = rng.binomial(1, varpi)

        mu0 = mu0_true_from_arrays(X, G, Z1, Z2, Z3)
        tau = tau_true_from_arrays(X, G, Z1, Z2, Z3)
        mu_true = mu0 + varpi * tau

        eps = rng.normal(0.0, 1.0, size=n)
        Y_zero = mu0 + eps
        Y_one = mu0 + tau + eps
        Y = A * Y_one + (1 - A) * Y_zero

        return pd.DataFrame(
            {
                "X": X,
                "G": G,
                "Z1": Z1,
                "Z2": Z2,
                "Z3": Z3,
                "A": A,
                "Y": Y,
                "Y_zero_true": Y_zero,
                "Y_one_true": Y_one,
                "tau_true": tau,
                "mu_true": mu_true,
                "varpi_true": varpi,
            }
        )

    def tau_features_correct(df, x_override=None, g_override=None):
        X, G, Z1, Z2, Z3 = covariate_arrays(df, x_override=x_override, g_override=g_override)
        X2 = X**2 - 1.0
        return np.column_stack([
            np.ones(len(df)),
            X,
            G,
            X * G,
            X2,
            X**3,
            X2 * G,
            Z1,
            Z2,
            G * Z1,
            np.sin(1.5 * X),
            Z1 * Z2,
            np.cos(Z3),
            X * (Z3 > 0),
            X * G * Z2,
            G * np.sin(2.0 * X),
            X2 * Z2,
            G * (Z1**2 - 1.0),
        ])

    def tau_features_wrong(df, basis="moderately wrong τ basis", x_override=None, g_override=None):
        X, G, Z1, Z2, Z3 = covariate_arrays(df, x_override=x_override, g_override=g_override)
        if basis == "sparse wrong τ basis":
            return np.column_stack([
                np.ones(len(df)),
                X,
                G,
                X * G,
            ])
        X2 = X**2 - 1.0
        return np.column_stack([
            np.ones(len(df)),
            X,
            G,
            X * G,
            X2,
            X2 * G,
            np.sin(1.5 * X),
            G * np.sin(2.0 * X),
            Z3,
        ])

    def mu0_features_correct(df):
        X, G, Z1, Z2, Z3 = covariate_arrays(df)
        X2 = X**2 - 1.0
        return np.column_stack([
            np.ones(len(df)),
            X,
            Z3,
            G,
            X * G,
            np.sin(2.0 * X),
            Z1**2 - 1.0,
            Z2 * Z3,
            np.cos(2.0 * Z3),
            X * G * Z1,
            X2 * Z2,
            G * np.sin(2.5 * Z1),
        ])

    def design_mu_correct(df):
        # E[Y | W] = mu0(W) + varpi(W) tau(W). Since the DGP is fixed,
        # a correctly specified direct mu regression needs both mu0 features
        # and varpi(W) times the tau features.
        pi = df["varpi_true"].to_numpy()
        return np.column_stack([
            mu0_features_correct(df),
            pi[:, None] * tau_features_correct(df),
        ])

    def design_mu_wrong(df):
        X, G, Z1, Z2, Z3 = covariate_arrays(df)
        X2 = X**2 - 1.0
        return np.column_stack([
            np.ones(len(df)),
            X,
            G,
            Z3,
            X * G,
            X2,
            X2 * G,
            np.sin(2.0 * X),
            np.cos(2.0 * Z3),
        ])

    def design_q_correct(df, a_value=None):
        d = df.copy()
        if a_value is not None:
            d["A"] = int(a_value)
        A = d["A"].to_numpy()
        return np.column_stack([
            mu0_features_correct(d),
            A[:, None] * tau_features_correct(d),
        ])

    def design_q_wrong(df, a_value=None):
        d = df.copy()
        if a_value is not None:
            d["A"] = int(a_value)
        X, G, Z1, Z2, Z3 = covariate_arrays(d)
        A = d["A"].to_numpy()
        X2 = X**2 - 1.0
        return np.column_stack([
            np.ones(len(d)),
            X,
            G,
            Z3,
            X * G,
            X2,
            np.sin(2.0 * X),
            np.cos(2.0 * Z3),
            A,
            A * X,
            A * G,
            A * X * G,
            A * X2,
            A * np.sin(1.5 * X),
            A * G * np.sin(2.0 * X),
        ])

    def design_propensity_correct(df):
        X, G, Z1, Z2, Z3 = covariate_arrays(df)
        return np.column_stack([
            X,
            Z1,
            np.sin(Z2),
            G,
            X * G,
            (Z3 > 0).astype(float),
        ])

    def design_propensity_mis(df):
        X, G, Z1, Z2, Z3 = covariate_arrays(df)
        return np.column_stack([
            X,
            G,
            X * G,
        ])

    def fit_ridge(X, y, penalty=1e-6, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            sw = np.ones(len(y))
        else:
            sw = np.clip(np.asarray(sample_weight, dtype=float), 1e-8, np.inf)
        Xw = X * np.sqrt(sw)[:, None]
        yw = y * np.sqrt(sw)
        ridge = np.eye(X.shape[1])
        ridge[0, 0] = 0.0
        lhs = Xw.T @ Xw + float(penalty) * ridge
        rhs = Xw.T @ yw
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def predict_ridge(X, beta):
        return np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float)

    def fit_linear_predict(train_df, test_df, design_function, outcome_name):
        model = LinearRegression(fit_intercept=False)
        model.fit(design_function(train_df), train_df[outcome_name].to_numpy())
        return model.predict(design_function(test_df)), model

    def safe_logistic_predict(train_df, test_df, design_function):
        y = train_df["A"].to_numpy()
        if len(np.unique(y)) < 2:
            return np.full(len(test_df), float(np.mean(y)))
        model = LogisticRegression(max_iter=3000, C=10.0)
        model.fit(design_function(train_df), y)
        return model.predict_proba(design_function(test_df))[:, 1]

    def crossfit_nuisances(df, feature_misspec_rho=0.0, prop_misspec_zeta=0.0, n_splits=3, seed=2026):
        n = len(df)
        mu_correct_hat = np.zeros(n)
        mu_wrong_hat = np.zeros(n)
        q1_correct_hat = np.zeros(n)
        q0_correct_hat = np.zeros(n)
        q1_wrong_hat = np.zeros(n)
        q0_wrong_hat = np.zeros(n)
        pi_correct_hat = np.zeros(n)
        pi_mis_hat = np.zeros(n)

        kfold = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        for train_idx, test_idx in kfold.split(df):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            mu_correct_hat[test_idx], _ = fit_linear_predict(train_df, test_df, design_mu_correct, "Y")
            mu_wrong_hat[test_idx], _ = fit_linear_predict(train_df, test_df, design_mu_wrong, "Y")

            q_correct_model = LinearRegression(fit_intercept=False)
            q_correct_model.fit(design_q_correct(train_df), train_df["Y"].to_numpy())
            q1_correct_hat[test_idx] = q_correct_model.predict(design_q_correct(test_df, a_value=1))
            q0_correct_hat[test_idx] = q_correct_model.predict(design_q_correct(test_df, a_value=0))

            q_wrong_model = LinearRegression(fit_intercept=False)
            q_wrong_model.fit(design_q_wrong(train_df), train_df["Y"].to_numpy())
            q1_wrong_hat[test_idx] = q_wrong_model.predict(design_q_wrong(test_df, a_value=1))
            q0_wrong_hat[test_idx] = q_wrong_model.predict(design_q_wrong(test_df, a_value=0))

            pi_correct_hat[test_idx] = safe_logistic_predict(train_df, test_df, design_propensity_correct)
            pi_mis_hat[test_idx] = safe_logistic_predict(train_df, test_df, design_propensity_mis)

        rho = float(feature_misspec_rho)
        zeta = float(prop_misspec_zeta)

        pi_correct_hat = np.clip(pi_correct_hat, 0.01, 0.99)
        pi_mis_hat = np.clip(pi_mis_hat, 0.01, 0.99)
        pi_hat = expit((1.0 - zeta) * logit(pi_correct_hat) + zeta * logit(pi_mis_hat))
        pi_hat = np.clip(pi_hat, 0.01, 0.99)

        mu_hat = (1.0 - rho) * mu_correct_hat + rho * mu_wrong_hat
        q1_hat = (1.0 - rho) * q1_correct_hat + rho * q1_wrong_hat
        q0_hat = (1.0 - rho) * q0_correct_hat + rho * q0_wrong_hat
        A = df["A"].to_numpy()
        qA_hat = A * q1_hat + (1 - A) * q0_hat

        return {
            "mu_hat": mu_hat,
            "mu_correct_hat": mu_correct_hat,
            "mu_wrong_hat": mu_wrong_hat,
            "q1_hat": q1_hat,
            "q0_hat": q0_hat,
            "qA_hat": qA_hat,
            "q1_correct_hat": q1_correct_hat,
            "q0_correct_hat": q0_correct_hat,
            "q1_wrong_hat": q1_wrong_hat,
            "q0_wrong_hat": q0_wrong_hat,
            "pi_hat": pi_hat,
            "pi_correct_hat": pi_correct_hat,
            "pi_mis_hat": pi_mis_hat,
        }

    def fit_tau_regression_from_design(df, outcome, treatment_resid=None, basis="correct", wrong_basis="moderately wrong τ basis"):
        if basis == "correct":
            X_tau = tau_features_correct(df)
        else:
            X_tau = tau_features_wrong(df, basis=wrong_basis)
        if treatment_resid is None:
            design = X_tau
        else:
            design = treatment_resid[:, None] * X_tau
        beta = fit_ridge(design, outcome, penalty=1e-6)
        return beta

    def fit_r_learner(df, feature_misspec_rho, prop_misspec_zeta, seed, wrong_basis):
        nuis = crossfit_nuisances(
            df,
            feature_misspec_rho=feature_misspec_rho,
            prop_misspec_zeta=prop_misspec_zeta,
            n_splits=3,
            seed=seed,
        )
        A = df["A"].to_numpy()
        Y = df["Y"].to_numpy()
        treatment_resid = A - nuis["pi_hat"]
        outcome_resid = Y - nuis["mu_hat"]

        beta_correct = fit_tau_regression_from_design(
            df,
            outcome=outcome_resid,
            treatment_resid=treatment_resid,
            basis="correct",
            wrong_basis=wrong_basis,
        )
        beta_wrong = fit_tau_regression_from_design(
            df,
            outcome=outcome_resid,
            treatment_resid=treatment_resid,
            basis="wrong",
            wrong_basis=wrong_basis,
        )
        diagnostics = {
            "mu_hat": nuis["mu_hat"],
            "pi_hat": nuis["pi_hat"],
            "pi_correct_hat": nuis["pi_correct_hat"],
            "pi_mis_hat": nuis["pi_mis_hat"],
            "mean_abs_mu_correct_error": float(np.mean(np.abs(nuis["mu_correct_hat"] - df["mu_true"].to_numpy()))),
            "mean_abs_mu_wrong_error": float(np.mean(np.abs(nuis["mu_wrong_hat"] - df["mu_true"].to_numpy()))),
            "mean_abs_mu_selected_error": float(np.mean(np.abs(nuis["mu_hat"] - df["mu_true"].to_numpy()))),
            "r_residual_sd": float(np.std(outcome_resid)),
            "r_weight_mean": float(np.mean(treatment_resid**2)),
            "r_weight_05": float(np.quantile(treatment_resid**2, 0.05)),
            "r_weight_95": float(np.quantile(treatment_resid**2, 0.95)),
        }
        return {
            "name": "R-learner",
            "beta_correct": beta_correct,
            "beta_wrong": beta_wrong,
            "rho": float(feature_misspec_rho),
            "wrong_basis": wrong_basis,
            "diagnostics": diagnostics,
        }

    def fit_dr_learner(df, feature_misspec_rho, prop_misspec_zeta, seed, wrong_basis):
        nuis = crossfit_nuisances(
            df,
            feature_misspec_rho=feature_misspec_rho,
            prop_misspec_zeta=prop_misspec_zeta,
            n_splits=3,
            seed=seed,
        )
        A = df["A"].to_numpy()
        Y = df["Y"].to_numpy()
        phi = (
            nuis["q1_hat"]
            - nuis["q0_hat"]
            + A / nuis["pi_hat"] * (Y - nuis["q1_hat"])
            - (1 - A) / (1.0 - nuis["pi_hat"]) * (Y - nuis["q0_hat"])
        )
        beta_correct = fit_tau_regression_from_design(
            df,
            outcome=phi,
            treatment_resid=None,
            basis="correct",
            wrong_basis=wrong_basis,
        )
        beta_wrong = fit_tau_regression_from_design(
            df,
            outcome=phi,
            treatment_resid=None,
            basis="wrong",
            wrong_basis=wrong_basis,
        )
        diagnostics = {
            "q1_hat": nuis["q1_hat"],
            "q0_hat": nuis["q0_hat"],
            "pi_hat": nuis["pi_hat"],
            "pi_correct_hat": nuis["pi_correct_hat"],
            "pi_mis_hat": nuis["pi_mis_hat"],
            "mean_abs_q_contrast_correct_error": float(np.mean(np.abs((nuis["q1_correct_hat"] - nuis["q0_correct_hat"]) - df["tau_true"].to_numpy()))),
            "mean_abs_q_contrast_wrong_error": float(np.mean(np.abs((nuis["q1_wrong_hat"] - nuis["q0_wrong_hat"]) - df["tau_true"].to_numpy()))),
            "mean_abs_q_contrast_selected_error": float(np.mean(np.abs((nuis["q1_hat"] - nuis["q0_hat"]) - df["tau_true"].to_numpy()))),
            "dr_pseudo_y_sd": float(np.std(phi)),
            "dr_pseudo_y_05": float(np.quantile(phi, 0.05)),
            "dr_pseudo_y_95": float(np.quantile(phi, 0.95)),
        }
        return {
            "name": "DR-learner",
            "beta_correct": beta_correct,
            "beta_wrong": beta_wrong,
            "rho": float(feature_misspec_rho),
            "wrong_basis": wrong_basis,
            "diagnostics": diagnostics,
        }

    def fit_outcome_plugin(df, feature_misspec_rho):
        q_correct_model = LinearRegression(fit_intercept=False)
        q_correct_model.fit(design_q_correct(df), df["Y"].to_numpy())
        q_wrong_model = LinearRegression(fit_intercept=False)
        q_wrong_model.fit(design_q_wrong(df), df["Y"].to_numpy())

        q1_correct = q_correct_model.predict(design_q_correct(df, a_value=1))
        q0_correct = q_correct_model.predict(design_q_correct(df, a_value=0))
        q1_wrong = q_wrong_model.predict(design_q_wrong(df, a_value=1))
        q0_wrong = q_wrong_model.predict(design_q_wrong(df, a_value=0))
        rho = float(feature_misspec_rho)
        tau_hat = (1.0 - rho) * (q1_correct - q0_correct) + rho * (q1_wrong - q0_wrong)
        diagnostics = {
            "plugin_tau_sd": float(np.std(tau_hat)),
            "plugin_tau_05": float(np.quantile(tau_hat, 0.05)),
            "plugin_tau_95": float(np.quantile(tau_hat, 0.95)),
            "plugin_correct_contrast_mae": float(np.mean(np.abs((q1_correct - q0_correct) - df["tau_true"].to_numpy()))),
            "plugin_wrong_contrast_mae": float(np.mean(np.abs((q1_wrong - q0_wrong) - df["tau_true"].to_numpy()))),
            "plugin_selected_contrast_mae": float(np.mean(np.abs(tau_hat - df["tau_true"].to_numpy()))),
        }
        return {
            "name": "Outcome plug-in",
            "q_correct_model": q_correct_model,
            "q_wrong_model": q_wrong_model,
            "rho": rho,
            "diagnostics": diagnostics,
        }

    def predict_tau(model, eval_df, x_override=None, g_override=None):
        if model["name"] == "Outcome plug-in":
            d = eval_df.copy()
            if x_override is not None:
                d["X"] = as_vector(x_override, len(d))
            if g_override is not None:
                d["G"] = as_vector(g_override, len(d))
            q1_correct = model["q_correct_model"].predict(design_q_correct(d, a_value=1))
            q0_correct = model["q_correct_model"].predict(design_q_correct(d, a_value=0))
            q1_wrong = model["q_wrong_model"].predict(design_q_wrong(d, a_value=1))
            q0_wrong = model["q_wrong_model"].predict(design_q_wrong(d, a_value=0))
            return (1.0 - model["rho"]) * (q1_correct - q0_correct) + model["rho"] * (q1_wrong - q0_wrong)

        X_correct = tau_features_correct(eval_df, x_override=x_override, g_override=g_override)
        X_wrong = tau_features_wrong(eval_df, basis=model["wrong_basis"], x_override=x_override, g_override=g_override)
        tau_correct = predict_ridge(X_correct, model["beta_correct"])
        tau_wrong = predict_ridge(X_wrong, model["beta_wrong"])
        return (1.0 - model["rho"]) * tau_correct + model["rho"] * tau_wrong

    def make_reference_grid(data_df, x_grid, seed=999):
        rows = []
        rng = np.random.default_rng(int(seed))
        m_per_x = 350
        for x_value in x_grid:
            G = rng.binomial(1, 0.45, size=m_per_x)
            Z1 = 0.30 + 0.55 * x_value + 0.45 * G + rng.normal(0.0, 0.90, size=m_per_x)
            Z2 = -0.20 + 0.45 * x_value * (1 - G) - 0.35 * G + rng.normal(0.0, 0.95, size=m_per_x)
            Z3 = 0.20 * x_value - 0.25 * Z1 + rng.normal(0.0, 1.00, size=m_per_x)
            rows.append(pd.DataFrame({"X": x_value, "G": G, "Z1": Z1, "Z2": Z2, "Z3": Z3}))
        ref_df = pd.concat(rows, ignore_index=True)
        ref_df["A"] = 0
        ref_df["Y"] = 0.0
        ref_df["varpi_true"] = 0.5
        ref_df["mu_true"] = 0.0
        ref_df["tau_true"] = tau_true_from_arrays(
            ref_df["X"].to_numpy(),
            ref_df["G"].to_numpy(),
            ref_df["Z1"].to_numpy(),
            ref_df["Z2"].to_numpy(),
            ref_df["Z3"].to_numpy(),
        )
        ref_df["x_grid_value"] = np.repeat(x_grid, m_per_x)
        return ref_df

    def true_tau_for_df(df):
        X, G, Z1, Z2, Z3 = covariate_arrays(df)
        return tau_true_from_arrays(X, G, Z1, Z2, Z3)

    def summarize_group_cate(models, eval_df, truth_col="tau_true"):
        rows = []
        for g in [0, 1]:
            in_group = eval_df["G"].to_numpy() == g
            rows.append({"group": f"G={g}", "estimator": "Truth", "estimate": float(eval_df.loc[in_group, truth_col].mean())})
            for model in models:
                tau_hat = predict_tau(model, eval_df.loc[in_group].reset_index(drop=True))
                rows.append({"group": f"G={g}", "estimator": model["name"], "estimate": float(np.mean(tau_hat))})
        return pd.DataFrame(rows)

    def summarize_x_curve(models, ref_grid_df, x_grid):
        curve_rows = []
        ref = ref_grid_df.copy()
        ref["tau_true_eval"] = true_tau_for_df(ref)
        for x_value in x_grid:
            at_x = ref["x_grid_value"].to_numpy() == x_value
            curve_rows.append({"x": x_value, "estimator": "Truth", "estimate": float(ref.loc[at_x, "tau_true_eval"].mean())})
            this_df = ref.loc[at_x].reset_index(drop=True)
            for model in models:
                tau_hat = predict_tau(model, this_df)
                curve_rows.append({"x": x_value, "estimator": model["name"], "estimate": float(np.mean(tau_hat))})
        return pd.DataFrame(curve_rows)

    def fit_models_and_summaries(df, feature_misspec_rho, prop_misspec_zeta, seed, x_grid, ref_grid_df, wrong_basis):
        plugin_model = fit_outcome_plugin(df, feature_misspec_rho=feature_misspec_rho)
        r_model = fit_r_learner(
            df,
            feature_misspec_rho=feature_misspec_rho,
            prop_misspec_zeta=prop_misspec_zeta,
            seed=int(seed) + 17,
            wrong_basis=wrong_basis,
        )
        dr_model = fit_dr_learner(
            df,
            feature_misspec_rho=feature_misspec_rho,
            prop_misspec_zeta=prop_misspec_zeta,
            seed=int(seed) + 31,
            wrong_basis=wrong_basis,
        )
        models = [plugin_model, r_model, dr_model]
        group_table = summarize_group_cate(models, df, truth_col="tau_true")
        x_curve = summarize_x_curve(models, ref_grid_df, x_grid)
        return models, group_table, x_curve

    def bootstrap_summaries(df, feature_misspec_rho, prop_misspec_zeta, seed, x_grid, ref_grid_df, wrong_basis, n_boot=30):
        rng = np.random.default_rng(int(seed) + 5000)
        n = len(df)
        group_samples = []
        curve_samples = []
        for b in range(int(n_boot)):
            sample_ids = rng.integers(0, n, size=n)
            boot_df = df.iloc[sample_ids].reset_index(drop=True)
            _, boot_group, boot_curve = fit_models_and_summaries(
                boot_df,
                feature_misspec_rho=feature_misspec_rho,
                prop_misspec_zeta=prop_misspec_zeta,
                seed=int(seed) + 100 * b + 7,
                x_grid=x_grid,
                ref_grid_df=ref_grid_df,
                wrong_basis=wrong_basis,
            )
            boot_group = boot_group[boot_group["estimator"] != "Truth"].copy()
            boot_curve = boot_curve[boot_curve["estimator"] != "Truth"].copy()
            boot_group["boot"] = b
            boot_curve["boot"] = b
            group_samples.append(boot_group)
            curve_samples.append(boot_curve)

        group_all = pd.concat(group_samples, ignore_index=True)
        curve_all = pd.concat(curve_samples, ignore_index=True)

        group_ci = (
            group_all.groupby(["group", "estimator"], observed=True)["estimate"]
            .quantile([0.025, 0.975])
            .unstack()
            .reset_index()
            .rename(columns={0.025: "lower", 0.975: "upper"})
        )
        curve_ci = (
            curve_all.groupby(["x", "estimator"], observed=True)["estimate"]
            .quantile([0.025, 0.975])
            .unstack()
            .reset_index()
            .rename(columns={0.025: "lower", 0.975: "upper"})
        )
        return group_ci, curve_ci

    def diagnostic_tables(df, models, feature_misspec_rho, prop_misspec_zeta):
        true_pi = df["varpi_true"].to_numpy()
        model_by_name = {m["name"]: m for m in models}
        plugin_model = model_by_name["Outcome plug-in"]
        r_model = model_by_name["R-learner"]
        dr_model = model_by_name["DR-learner"]
        pi_hat = r_model["diagnostics"]["pi_hat"]
        pi_correct_hat = r_model["diagnostics"]["pi_correct_hat"]
        pi_mis_hat = r_model["diagnostics"]["pi_mis_hat"]
        rows = [
            ("learner feature misspecification rho", float(feature_misspec_rho)),
            ("propensity misspecification zeta", float(prop_misspec_zeta)),
            ("treatment prevalence", float(df["A"].mean())),
            ("5% true varpi(W)", float(np.quantile(true_pi, 0.05))),
            ("median true varpi(W)", float(np.median(true_pi))),
            ("95% true varpi(W)", float(np.quantile(true_pi, 0.95))),
            ("5% selected varpi_hat(W)", float(np.quantile(pi_hat, 0.05))),
            ("median selected varpi_hat(W)", float(np.median(pi_hat))),
            ("95% selected varpi_hat(W)", float(np.quantile(pi_hat, 0.95))),
            ("mean abs correct logistic varpi_hat error", float(np.mean(np.abs(pi_correct_hat - true_pi)))),
            ("mean abs misspecified logistic varpi_hat error", float(np.mean(np.abs(pi_mis_hat - true_pi)))),
            ("mean abs selected varpi_hat error", float(np.mean(np.abs(pi_hat - true_pi)))),
            ("mean abs mu_correct error", r_model["diagnostics"]["mean_abs_mu_correct_error"]),
            ("mean abs mu_wrong error", r_model["diagnostics"]["mean_abs_mu_wrong_error"]),
            ("mean abs mu_selected error", r_model["diagnostics"]["mean_abs_mu_selected_error"]),
            ("Outcome plug-in selected contrast MAE", plugin_model["diagnostics"]["plugin_selected_contrast_mae"]),
            ("DR selected Q-contrast MAE", dr_model["diagnostics"]["mean_abs_q_contrast_selected_error"]),
            ("R residual outcome SD", r_model["diagnostics"]["r_residual_sd"]),
            ("DR pseudo-outcome SD", dr_model["diagnostics"]["dr_pseudo_y_sd"]),
            ("mean R residual-treatment weight", r_model["diagnostics"]["r_weight_mean"]),
        ]
        return pd.DataFrame(rows, columns=["quantity", "value"])

    def rmse_table(group_table, x_curve):
        rows = []
        truth_group = group_table[group_table["estimator"] == "Truth"].set_index("group")["estimate"]
        truth_curve = x_curve[x_curve["estimator"] == "Truth"].set_index("x")["estimate"]
        for est in ["Outcome plug-in", "R-learner", "DR-learner"]:
            est_group = group_table[group_table["estimator"] == est].set_index("group")["estimate"]
            est_curve = x_curve[x_curve["estimator"] == est].set_index("x")["estimate"]
            rows.append({
                "estimator": est,
                "group CATE RMSE": float(np.sqrt(np.mean((est_group - truth_group) ** 2))),
                "X-curve CATE RMSE": float(np.sqrt(np.mean((est_curve - truth_curve) ** 2))),
            })
        return pd.DataFrame(rows)

    def run_hte_experiment(n, feature_misspec_rho, prop_misspec_zeta, overlap_lambda, wrong_basis, seed):
        df = generate_data(
            n=int(n),
            overlap_lambda=float(overlap_lambda),
            seed=int(seed),
        )
        x_left, x_right = np.quantile(df["X"], [0.03, 0.97])
        x_grid = np.linspace(x_left, x_right, 65)
        ref_grid_df = make_reference_grid(df, x_grid=x_grid, seed=int(seed) + 900)

        models, group_table, x_curve = fit_models_and_summaries(
            df,
            feature_misspec_rho=float(feature_misspec_rho),
            prop_misspec_zeta=float(prop_misspec_zeta),
            seed=int(seed),
            x_grid=x_grid,
            ref_grid_df=ref_grid_df,
            wrong_basis=wrong_basis,
        )

        group_ci, curve_ci = bootstrap_summaries(
            df,
            feature_misspec_rho=float(feature_misspec_rho),
            prop_misspec_zeta=float(prop_misspec_zeta),
            seed=int(seed),
            x_grid=x_grid,
            ref_grid_df=ref_grid_df,
            wrong_basis=wrong_basis,
            n_boot=30,
        )
        diag = diagnostic_tables(
            df,
            models,
            feature_misspec_rho=float(feature_misspec_rho),
            prop_misspec_zeta=float(prop_misspec_zeta),
        )
        rmse = rmse_table(group_table, x_curve)
        return df, models, group_table, group_ci, x_curve, curve_ci, diag, rmse, x_grid

    def make_group_plot(group_table, group_ci):
        estimators = ["Outcome plug-in", "R-learner", "DR-learner"]
        groups = ["G=0", "G=1"]
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        width = 0.18
        centers = np.arange(len(groups))
        for j, est in enumerate(estimators):
            values = []
            lowers = []
            uppers = []
            for g in groups:
                point = group_table[(group_table["group"] == g) & (group_table["estimator"] == est)]["estimate"].iloc[0]
                ci_row = group_ci[(group_ci["group"] == g) & (group_ci["estimator"] == est)]
                values.append(float(point))
                lowers.append(float(ci_row["lower"].iloc[0]))
                uppers.append(float(ci_row["upper"].iloc[0]))
            pos = centers + (j - 1.0) * width
            yerr = np.vstack([np.asarray(values) - np.asarray(lowers), np.asarray(uppers) - np.asarray(values)])
            ax.errorbar(pos, values, yerr=yerr, fmt="o", capsize=5, linewidth=1.4, markersize=6, label=est)
        truth = [group_table[(group_table["group"] == g) & (group_table["estimator"] == "Truth")]["estimate"].iloc[0] for g in groups]
        ax.scatter(centers, truth, marker="D", s=70, label="truth")
        ax.set_xticks(centers)
        ax.set_xticklabels(groups)
        ax.set_ylabel(r"$\tau_G(g)=E\{\tau(W)\mid G=g\}$")
        ax.set_title("Group CATE point estimates with bootstrap intervals")
        ax.legend()
        fig.tight_layout()
        return fig

    def make_curve_plot(x_curve, curve_ci):
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        truth = x_curve[x_curve["estimator"] == "Truth"].sort_values("x")
        ax.plot(truth["x"], truth["estimate"], linewidth=2.8, linestyle="--", label="truth")
        for est in ["Outcome plug-in", "R-learner", "DR-learner"]:
            curve = x_curve[x_curve["estimator"] == est].sort_values("x")
            ax.plot(curve["x"], curve["estimate"], linewidth=2.0, label=est)
            ci = curve_ci[curve_ci["estimator"] == est].sort_values("x")
            ax.fill_between(ci["x"].to_numpy(), ci["lower"].to_numpy(), ci["upper"].to_numpy(), alpha=0.15)
        ax.set_xlabel("continuous covariate X")
        ax.set_ylabel(r"$\tau_X(x)=E\{\tau(W)\mid X=x\}$")
        ax.set_title("Continuous CATE curve with pointwise bootstrap regions")
        ax.legend()
        fig.tight_layout()
        return fig

    def interpretation_message(rmse, diag, feature_rho, prop_zeta, overlap_lambda):
        d = dict(zip(diag["quantity"], diag["value"]))
        curve_rmse = rmse.set_index("estimator")["X-curve CATE RMSE"].to_dict()
        best_est = min(curve_rmse, key=curve_rmse.get)
        lines = [
            f"For the continuous CATE curve, the smallest RMSE in this run is from **{best_est}**."
        ]

        if float(feature_rho) == 0:
            lines.append(
                "The learner feature misspecification slider is zero. The fitted mu, Q, and final tau regressions use the correct feature dictionaries for the fixed DGP. Remaining error is finite-sample estimation error, regularization, and bootstrap noise."
            )
        elif float(feature_rho) >= 0.7:
            lines.append(
                f"Learner feature misspecification is high. The selected mu model has mean absolute error {d['mean abs mu_selected error']:.3f}, while the selected Q-contrast has mean absolute error {d['DR selected Q-contrast MAE']:.3f}. R and DR are also constrained by the wrong final tau dictionary, especially when the sparse wrong basis is selected."
            )
        else:
            lines.append(
                "Learner feature misspecification is intermediate: mu, Q, and final tau predictions are interpolated between the correctly specified and wrong feature learners."
            )

        if float(prop_zeta) >= 0.7:
            lines.append(
                f"The selected propensity model is close to the deliberately misspecified logistic regression. Its mean absolute error against the simulation truth is {d['mean abs selected varpi_hat error']:.3f}. This affects R and DR, but it does not directly affect the outcome plug-in because the plug-in never uses varpi_hat(W)."
            )
        else:
            lines.append(
                f"The selected propensity model is close to the correctly specified estimated logistic regression. Even at zeta=0 it is estimated, not oracle. Its mean absolute error against the simulation truth is {d['mean abs correct logistic varpi_hat error']:.3f}."
            )

        if float(overlap_lambda) >= 1.75:
            lines.append(
                "Overlap is weak. DR pseudo-outcomes can become noisy because they contain inverse-propensity weights; R-learner residual treatment variation also becomes small in some regions."
            )
        return "\n\n".join(lines)

    return (
        interpretation_message,
        make_curve_plot,
        make_group_plot,
        run_hte_experiment,
    )


@app.cell(hide_code=True)
def _(
    hte_feature_rho,
    hte_n,
    hte_overlap,
    hte_prop_zeta,
    hte_run_button,
    hte_seed,
    hte_wrong_tau_basis,
    mo,
    run_hte_experiment,
):
    mo.stop(
        not hte_run_button.value,
        mo.md("Press **Run HTE experiment** after selecting the levers."),
    )
    (
        hte_df,
        hte_models,
        hte_group_table,
        hte_group_ci,
        hte_x_curve,
        hte_curve_ci,
        hte_diag,
        hte_rmse,
        hte_x_grid,
    ) = run_hte_experiment(
        n=hte_n.value,
        feature_misspec_rho=hte_feature_rho.value,
        prop_misspec_zeta=hte_prop_zeta.value,
        overlap_lambda=hte_overlap.value,
        wrong_basis=hte_wrong_tau_basis.value,
        seed=hte_seed.value,
    )
    return (
        hte_curve_ci,
        hte_diag,
        hte_group_ci,
        hte_group_table,
        hte_rmse,
        hte_x_curve,
    )


@app.cell(hide_code=True)
def _(hte_group_ci, hte_group_table, make_group_plot, mo):
    mo.vstack([
        mo.md(r"## 2. CATE on the binary group $G$"),
        mo.md(r"The target is $\tau_G(g)=E\{\tau(W)\mid G=g\}$. The diamond shows the oracle truth from the simulation. The points show outcome plug-in, R-learner, and DR-learner estimates with percentile bootstrap intervals."),
        make_group_plot(hte_group_table, hte_group_ci),
    ])
    return


@app.cell(hide_code=True)
def _(hte_curve_ci, hte_x_curve, make_curve_plot, mo):
    mo.vstack([
        mo.md(r"## 3. CATE as a function of the continuous variable $X$"),
        mo.md(r"The target is $\tau_X(x)=E\{\tau(W)\mid X=x\}$. The shaded regions are pointwise bootstrap intervals, not simultaneous confidence bands."),
        make_curve_plot(hte_x_curve, hte_curve_ci),
    ])
    return


@app.cell(hide_code=True)
def _(
    hte_diag,
    hte_feature_rho,
    hte_overlap,
    hte_prop_zeta,
    hte_rmse,
    interpretation_message,
    mo,
):
    mo.vstack([
        mo.md("## 4. Diagnostics and current-run interpretation"),
        mo.md("### RMSE against the simulation oracle"),
        hte_rmse.round(3),
        mo.md("### Nuisance and pseudo-outcome diagnostics"),
        hte_diag.round(4),
        mo.md("### Interpretation"),
        mo.md(interpretation_message(hte_rmse, hte_diag, hte_feature_rho.value, hte_prop_zeta.value, hte_overlap.value)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Student discussion prompts

    1. **Start from a clean reference setting.** Set \(\rho=0\), \(\zeta=0\), and choose a setting with good overlap. Are the outcome plug-in, R-learner, and DR-learner all close to the oracle CATE curves?

    2. **Introduce outcome and CATE-model misspecification.** Increase \(\rho\) while keeping \(\zeta=0\). Which estimator deteriorates most clearly? Why?

    3. **Misspecify the propensity score but keep outcome features correct.** Set \(\rho=0\) and increase \(\zeta\), while keeping overlap reasonably good. Which estimators change? Why does the outcome plug-in not directly react to \(\widehat\varpi(W)\)?

    4. **Combine propensity misspecification with weak overlap.** Set \(\rho=0\), then increase both \(\zeta\) and \(\lambda\). How do the R-learner and DR-learner CATE curves compare?

    5. **Connect the behavior to double robustness.** In the previous setting, what does the DR-learner illustrate about double robustness? What does the R-learner lack by comparison?

    6. Feel free to try new configurations, to check the code, to formulate questions, and **to revisit prior Marimo labs**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running this notebook

    From the directory containing this file:

    ```bash
    pip install marimo numpy pandas matplotlib scikit-learn
    marimo edit 5_hte_r_dr_marimo.py
    ```

    To serve it as an app instead of opening the notebook editor:

    ```bash
    marimo run 5_hte_r_dr_marimo.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
