import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Estimating the ATE and CATE: parametric plug-in vs S-learner vs T-learner

        This notebook illustrates the estimation step after identification.

        We simulate observational data with

        $$
        O=(W,A,Y),
        \qquad
        W=(X,Z_1,Z_2,G),
        $$

        where $X$ is the continuous covariate used for the CATE plot, while
        $Z_1,Z_2,G$ are additional pre-treatment covariates that must be adjusted for and marginalized over.

        The notebook compares three plug-in estimators:

        1. a deliberately rigid parametric regression;
        2. an S-learner: one flexible outcome model $\widehat Q(W,A)$;
        3. a T-learner: two flexible outcome models $\widehat Q_1(W)$ and $\widehat Q_0(W)$.

        For each estimator, we show:

        - an ATE point estimate with a bootstrap confidence interval;
        - a CATE curve $x\mapsto E[Y^1-Y^0\mid X=x]$ with pointwise bootstrap intervals;
        - the true simulated values, because the potential outcomes are known in the simulation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data-generating process

        The structural data-generating process is

        $$
        X,Z_1,Z_2,G \longrightarrow A,
        \qquad
        X,Z_1,Z_2,G \longrightarrow Y^0,Y^1,
        \qquad
        Y=AY^1+(1-A)Y^0.
        $$

        Treatment assignment follows

        $$
        A\mid W \sim \operatorname{Bernoulli}\{\pi(W)\}.
        $$

        Increasing the **selection / lack-of-overlap strength** makes treatment more predictable from $W$.
        That makes the treated and control groups less comparable and forces estimators to rely more on adjustment and extrapolation.

        The treatment effect is heterogeneous:

        $$
        Y^1-Y^0
        =
        \tau_0+\tau_X X+\tau_{X^2}(X^2-1)+\tau_G G.
        $$

        Therefore the true CATE curve in $X$ is nonlinear whenever $\tau_{X^2}\neq 0$.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, pd):
    # -------------------------------------------------------------------------
    # Small numerical utilities
    # -------------------------------------------------------------------------

    def expit(z):
        """Numerically stable logistic transform."""
        z = np.asarray(z)
        return 1.0 / (1.0 + np.exp(-z))


    def as_vector(value, n):
        """Return a length-n vector, whether value is already a vector or a scalar."""
        if np.isscalar(value):
            return np.full(n, float(value))
        return np.asarray(value, dtype=float)


    # -------------------------------------------------------------------------
    # Data simulation
    # -------------------------------------------------------------------------

    def generate_data(
        n,
        seed,
        overlap_strength=1.25,
        baseline_nonlinearity=1.00,
        tau0=1.00,
        tau_x=0.50,
        tau_x2=0.30,
        tau_g=0.50,
        noise_sd=1.00,
    ):
        """
        Simulate one observational data set.

        Important teaching point:
        - The analyst observes only W, A, and Y.
        - We keep Y0, Y1, and tau in the data only because this is a simulation,
          so we can compare the estimators against the truth.
        """
        rng = np.random.default_rng(int(seed))
        n = int(n)

        # Pre-treatment covariates.
        # X is the focal continuous covariate for the CATE plot.
        X = rng.normal(loc=0.0, scale=1.0, size=n)
        Z1 = rng.normal(loc=0.0, scale=1.0, size=n)
        Z2 = rng.normal(loc=0.0, scale=1.0, size=n)
        G = rng.binomial(n=1, p=0.45, size=n)

        # Propensity score: probability of naturally receiving treatment.
        # Larger overlap_strength means stronger selection into treatment.
        logit_pi = -0.10 + overlap_strength * (
            0.85 * X - 0.75 * Z1 + 0.55 * np.sin(Z2) + 0.90 * (G - 0.45)
        )
        pi = np.clip(expit(logit_pi), 0.02, 0.98)
        A = rng.binomial(n=1, p=pi)

        # Baseline outcome under control.
        # This is intentionally nonlinear, so a main-effects linear regression is misspecified.
        nonlinear_part = (
            1.10 * np.sin(X)
            + 0.45 * (X**2 - 1.0)
            + 0.50 * np.cos(Z2)
            + 0.35 * X * G
        )
        linear_part = 0.80 * Z1 + 0.65 * G - 0.35 * Z2
        mu0 = 0.20 + baseline_nonlinearity * nonlinear_part + linear_part

        # Individual treatment effect.
        tau = tau0 + tau_x * X + tau_x2 * (X**2 - 1.0) + tau_g * G

        # Potential outcomes and observed outcome.
        epsilon = rng.normal(loc=0.0, scale=noise_sd, size=n)
        Y0 = mu0 + epsilon
        Y1 = Y0 + tau
        Y = A * Y1 + (1 - A) * Y0

        return pd.DataFrame(
            {
                "X": X,
                "Z1": Z1,
                "Z2": Z2,
                "G": G,
                "pi": pi,
                "A": A,
                "Y": Y,
                "Y0": Y0,
                "Y1": Y1,
                "tau": tau,
            }
        )


    # -------------------------------------------------------------------------
    # Design matrices for outcome regression
    # -------------------------------------------------------------------------

    def covariate_arrays(data, x_override=None):
        """
        Extract covariates as numpy arrays.

        x_override is useful for CATE curves: we set X=x for everyone,
        while keeping the empirical distribution of Z1, Z2, and G fixed.
        """
        n = len(data)
        X = as_vector(data["X"].to_numpy() if x_override is None else x_override, n)
        Z1 = data["Z1"].to_numpy()
        Z2 = data["Z2"].to_numpy()
        G = data["G"].to_numpy()
        return X, Z1, Z2, G


    def linear_design(data, a, x_override=None):
        """
        Rigid parametric design matrix.

        This model is intentionally simple:
            Y ~ 1 + A + X + Z1 + Z2 + G

        It omits nonlinearities and treatment-effect interactions, so it is generally
        misspecified for the simulated data-generating process.
        """
        X, Z1, Z2, G = covariate_arrays(data, x_override=x_override)
        A = as_vector(a, len(data))
        return np.column_stack([np.ones(len(data)), A, X, Z1, Z2, G])


    def flexible_w_basis(data, x_override=None):
        """
        A simple hand-built nonlinear basis for W.

        This plays the role of a transparent ML feature map.
        It includes nonlinear transformations and interactions, but avoids any
        external dependency such as scikit-learn.
        """
        X, Z1, Z2, G = covariate_arrays(data, x_override=x_override)
        return np.column_stack(
            [
                np.ones(len(data)),
                X,
                Z1,
                Z2,
                G,
                X**2 - 1.0,
                Z1**2 - 1.0,
                Z2**2 - 1.0,
                X * Z1,
                X * G,
                Z1 * G,
                np.sin(X),
                np.cos(Z2),
            ]
        )


    def s_learner_design(data, a, x_override=None):
        """
        Flexible design matrix for the S-learner.

        The S-learner fits one regression surface Q(W,A).  We include both
        baseline terms in W and treatment-by-W interactions.  This lets the
        single model represent heterogeneous treatment effects.
        """
        basis = flexible_w_basis(data, x_override=x_override)
        A = as_vector(a, len(data))
        return np.column_stack([basis, A[:, None] * basis])


    # -------------------------------------------------------------------------
    # Ridge regression fitting and prediction
    # -------------------------------------------------------------------------

    def fit_ridge(design, outcome, penalty):
        """
        Fit ridge regression by solving the normal equations.

        The intercept is not penalized.  This is not meant to be a production
        implementation; it is written explicitly so students can see the estimator.
        """
        X = np.asarray(design, dtype=float)
        y = np.asarray(outcome, dtype=float)

        ridge_matrix = np.eye(X.shape[1])
        ridge_matrix[0, 0] = 0.0

        lhs = X.T @ X + float(penalty) * ridge_matrix
        rhs = X.T @ y
        return np.linalg.solve(lhs, rhs)


    def predict(design, beta):
        """Linear prediction for a fitted regression coefficient vector."""
        return np.asarray(design, dtype=float) @ np.asarray(beta, dtype=float)


    # -------------------------------------------------------------------------
    # Three plug-in learners
    # -------------------------------------------------------------------------

    def fit_parametric_plugin(data):
        """Fit the deliberately misspecified main-effects linear regression."""
        design = linear_design(data, a=data["A"].to_numpy())
        beta = fit_ridge(design, data["Y"].to_numpy(), penalty=1e-8)
        return {"name": "Parametric", "beta": beta}


    def fit_s_learner(data, penalty):
        """Fit one flexible outcome model Q(W,A)."""
        design = s_learner_design(data, a=data["A"].to_numpy())
        beta = fit_ridge(design, data["Y"].to_numpy(), penalty=penalty)
        return {"name": "S-learner", "beta": beta}


    def fit_t_learner(data, penalty):
        """Fit separate flexible outcome models in the treated and control arms."""
        treated = data[data["A"] == 1]
        control = data[data["A"] == 0]

        beta1 = fit_ridge(
            flexible_w_basis(treated),
            treated["Y"].to_numpy(),
            penalty=penalty,
        )
        beta0 = fit_ridge(
            flexible_w_basis(control),
            control["Y"].to_numpy(),
            penalty=penalty,
        )
        return {"name": "T-learner", "beta1": beta1, "beta0": beta0}


    def predict_under_treatment(model, data, a, x_override=None):
        """Predict E[Y | W, do(A=a)] using one fitted plug-in model."""
        if model["name"] == "Parametric":
            return predict(linear_design(data, a=a, x_override=x_override), model["beta"])

        if model["name"] == "S-learner":
            return predict(s_learner_design(data, a=a, x_override=x_override), model["beta"])

        if model["name"] == "T-learner":
            if int(a) == 1:
                return predict(flexible_w_basis(data, x_override=x_override), model["beta1"])
            return predict(flexible_w_basis(data, x_override=x_override), model["beta0"])

        raise ValueError(f"Unknown model type: {model['name']}")


    # -------------------------------------------------------------------------
    # ATE and CATE estimation
    # -------------------------------------------------------------------------

    def estimate_ate(model, data):
        """
        Plug-in ATE estimate.

        For every observed covariate vector W_i, predict the outcome twice:
        once under A=1 and once under A=0.  Then average the differences.
        """
        q1 = predict_under_treatment(model, data, a=1)
        q0 = predict_under_treatment(model, data, a=0)
        return float(np.mean(q1 - q0))


    def estimate_cate_curve(model, data, x_grid):
        """
        Estimate the marginal CATE curve in X.

        For each x on the grid, set X=x for everyone, keep Z1,Z2,G as observed,
        predict Q(1,x,Z1,Z2,G)-Q(0,x,Z1,Z2,G), and average over Z1,Z2,G.
        """
        curve = []
        for x_value in x_grid:
            q1 = predict_under_treatment(model, data, a=1, x_override=x_value)
            q0 = predict_under_treatment(model, data, a=0, x_override=x_value)
            curve.append(np.mean(q1 - q0))
        return np.asarray(curve)


    def true_ate(data):
        """True finite-sample ATE, available only because this is a simulation."""
        return float(data["tau"].mean())


    def true_cate_curve(data, x_grid, tau0, tau_x, tau_x2, tau_g):
        """
        True CATE curve marginalized over Z1,Z2,G.

        The simulated tau depends only on X and G, so for each fixed x we average
        tau0 + tau_x*x + tau_x2*(x^2-1) + tau_g*G_i over the empirical G_i values.
        """
        mean_g = float(data["G"].mean())
        return tau0 + tau_x * x_grid + tau_x2 * (x_grid**2 - 1.0) + tau_g * mean_g


    def fit_and_evaluate_all(data, x_grid, ridge_penalty):
        """Fit all three estimators and return their ATEs and CATE curves."""
        models = [
            fit_parametric_plugin(data),
            fit_s_learner(data, penalty=ridge_penalty),
            fit_t_learner(data, penalty=ridge_penalty),
        ]

        ate_rows = []
        cate_curves = {}
        for model in models:
            name = model["name"]
            ate_rows.append({"estimator": name, "estimate": estimate_ate(model, data)})
            cate_curves[name] = estimate_cate_curve(model, data, x_grid)

        return pd.DataFrame(ate_rows), cate_curves


    def bootstrap_intervals(data, x_grid, ridge_penalty, n_boot, seed):
        """
        Nonparametric bootstrap intervals.

        These are simple percentile intervals.  For the CATE curves they are
        pointwise intervals, not simultaneous confidence bands.
        """
        n_boot = int(n_boot)
        if n_boot <= 0:
            empty_ate = pd.DataFrame(columns=["estimator", "lower", "upper"])
            empty_curves = {}
            return empty_ate, empty_curves

        rng = np.random.default_rng(int(seed) + 10_000)
        n = len(data)
        names = ["Parametric", "S-learner", "T-learner"]

        ate_samples = {name: [] for name in names}
        curve_samples = {name: [] for name in names}

        for _ in range(n_boot):
            sample_ids = rng.integers(low=0, high=n, size=n)
            boot_data = data.iloc[sample_ids].reset_index(drop=True)

            boot_ate, boot_curves = fit_and_evaluate_all(
                boot_data,
                x_grid=x_grid,
                ridge_penalty=ridge_penalty,
            )

            for name in names:
                value = boot_ate.loc[boot_ate["estimator"] == name, "estimate"].iloc[0]
                ate_samples[name].append(value)
                curve_samples[name].append(boot_curves[name])

        ate_rows = []
        curve_intervals = {}
        for name in names:
            ate_array = np.asarray(ate_samples[name])
            ate_rows.append(
                {
                    "estimator": name,
                    "lower": np.quantile(ate_array, 0.025),
                    "upper": np.quantile(ate_array, 0.975),
                }
            )

            curve_array = np.vstack(curve_samples[name])
            curve_intervals[name] = {
                "lower": np.quantile(curve_array, 0.025, axis=0),
                "upper": np.quantile(curve_array, 0.975, axis=0),
            }

        return pd.DataFrame(ate_rows), curve_intervals


    def treatment_summary(data):
        """Compact table showing treatment imbalance and overlap."""
        pi = data["pi"]
        return pd.DataFrame(
            {
                "quantity": [
                    "Pr(A=1)",
                    "mean X among treated",
                    "mean X among controls",
                    "mean Z1 among treated",
                    "mean Z1 among controls",
                    "Pr(G=1 among treated)",
                    "Pr(G=1 among controls)",
                    "10th percentile of pi(W)",
                    "90th percentile of pi(W)",
                    "Pr(0.1 < pi(W) < 0.9)",
                ],
                "value": [
                    data["A"].mean(),
                    data.loc[data["A"] == 1, "X"].mean(),
                    data.loc[data["A"] == 0, "X"].mean(),
                    data.loc[data["A"] == 1, "Z1"].mean(),
                    data.loc[data["A"] == 0, "Z1"].mean(),
                    data.loc[data["A"] == 1, "G"].mean(),
                    data.loc[data["A"] == 0, "G"].mean(),
                    pi.quantile(0.10),
                    pi.quantile(0.90),
                    ((pi > 0.1) & (pi < 0.9)).mean(),
                ],
            }
        )


    def bin_means(x, y, n_bins=18):
        """Compute binned means for simple diagnostic plots."""
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-8
        edges[-1] += 1e-8
        bin_id = np.digitize(x, edges) - 1

        rows = []
        for j in range(n_bins):
            in_bin = bin_id == j
            if in_bin.sum() > 0:
                rows.append(
                    {
                        "x_mean": np.mean(x[in_bin]),
                        "y_mean": np.mean(y[in_bin]),
                        "n": int(in_bin.sum()),
                    }
                )
        return pd.DataFrame(rows)


    return (
        bin_means,
        bootstrap_intervals,
        estimate_ate,
        estimate_cate_curve,
        expit,
        fit_and_evaluate_all,
        generate_data,
        treatment_summary,
        true_ate,
        true_cate_curve,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive parameters

        Useful classroom exercises:

        1. set treatment-effect heterogeneity close to zero and compare the ATE and CATE plots;
        2. increase heterogeneity in $X$ and watch the parametric model fail to recover the CATE curve;
        3. increase lack of overlap and watch the intervals widen, especially for the T-learner;
        4. increase baseline nonlinearity and see how misspecified adjustment can bias the parametric estimator.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    controls = mo.ui.dictionary(
        {
            "n": mo.ui.slider(
                start=500,
                stop=10_000,
                step=500,
                value=2_500,
                label="sample size",
            ),
            "overlap_strength": mo.ui.slider(
                start=0.0,
                stop=3.0,
                step=0.05,
                value=1.25,
                label="selection / lack-of-overlap strength",
            ),
            "baseline_nonlinearity": mo.ui.slider(
                start=0.0,
                stop=2.0,
                step=0.05,
                value=1.00,
                label="baseline outcome nonlinearity",
            ),
            "tau0": mo.ui.slider(
                start=-1.0,
                stop=3.0,
                step=0.05,
                value=1.00,
                label="baseline treatment effect tau_0",
            ),
            "tau_x": mo.ui.slider(
                start=-1.5,
                stop=1.5,
                step=0.05,
                value=0.50,
                label="linear heterogeneity tau_X",
            ),
            "tau_x2": mo.ui.slider(
                start=-1.0,
                stop=1.0,
                step=0.05,
                value=0.30,
                label="nonlinear heterogeneity tau_X2",
            ),
            "tau_g": mo.ui.slider(
                start=-1.0,
                stop=1.5,
                step=0.05,
                value=0.50,
                label="group heterogeneity tau_G",
            ),
            "noise_sd": mo.ui.slider(
                start=0.25,
                stop=3.0,
                step=0.05,
                value=1.00,
                label="outcome noise SD",
            ),
            "ridge_penalty": mo.ui.slider(
                start=0.0,
                stop=20.0,
                step=0.25,
                value=2.00,
                label="ridge penalty for S/T learners",
            ),
            "n_boot": mo.ui.slider(
                start=0,
                stop=120,
                step=10,
                value=40,
                label="bootstrap repetitions",
            ),
            "seed": mo.ui.number(
                start=1,
                stop=999_999,
                value=2026,
                label="seed",
            ),
        },
        label="Simulation controls",
    )
    controls.vstack()
    return (controls,)


@app.cell(hide_code=True)
def _(controls, generate_data):
    values = controls.value

    data = generate_data(
        n=int(values["n"]),
        seed=int(values["seed"]),
        overlap_strength=float(values["overlap_strength"]),
        baseline_nonlinearity=float(values["baseline_nonlinearity"]),
        tau0=float(values["tau0"]),
        tau_x=float(values["tau_x"]),
        tau_x2=float(values["tau_x2"]),
        tau_g=float(values["tau_g"]),
        noise_sd=float(values["noise_sd"]),
    )

    data.head()
    return data, values


@app.cell(hide_code=True)
def _(data, treatment_summary):
    overlap_table = treatment_summary(data)
    overlap_table.round(3)
    return (overlap_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Treatment selection and overlap

        The propensity score $\pi(W)=P(A=1\mid W)$ is known in the simulation.
        In real data it would have to be estimated.

        When the treated and untreated groups occupy very different regions of covariate space,
        outcome regression has to extrapolate.  This is especially visible when the T-learner fits
        separate models inside each treatment arm.
        """
    )
    return


@app.cell(hide_code=True)
def _(bin_means, data, plt):
    # Plot the empirical treatment probability as a function of the focal covariate X.
    # The two curves correspond to the two levels of G.
    fig_overlap, ax_overlap = plt.subplots(figsize=(8, 4.8))

    for g_value in [0, 1]:
        group_data = data[data["G"] == g_value]
        binned = bin_means(
            group_data["X"].to_numpy(),
            group_data["A"].to_numpy(),
            n_bins=16,
        )
        ax_overlap.plot(
            binned["x_mean"],
            binned["y_mean"],
            marker="o",
            label=f"G={g_value}",
        )

    ax_overlap.set_ylim(-0.02, 1.02)
    ax_overlap.set_xlabel("continuous covariate X")
    ax_overlap.set_ylabel("empirical Pr(A=1 | X bin, G)")
    ax_overlap.set_title("Treatment selection as a function of pre-treatment covariates")
    ax_overlap.legend(title="group")
    fig_overlap.tight_layout()
    fig_overlap

    return (fig_overlap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. The three estimators

        All three estimators use the same identified estimand

        $$
        \psi
        =
        E\{Q(W,1)-Q(W,0)\},
        \qquad
        Q(W,A)=E[Y\mid W,A].
        $$

        They differ only in how they estimate $Q$.

        **Parametric plug-in:** fit one main-effects linear regression

        $$
        Y \sim 1 + A + X + Z_1 + Z_2 + G.
        $$

        **S-learner:** fit one flexible regression surface $\widehat Q(W,A)$ using nonlinear basis terms and treatment interactions.

        **T-learner:** fit two flexible regression surfaces: one among treated observations and one among controls.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    bootstrap_intervals,
    data,
    fit_and_evaluate_all,
    np,
    pd,
    true_ate,
    true_cate_curve,
    values,
):
    # The CATE curve is evaluated on a central grid of X values.
    # We avoid extreme tails because all estimators become unstable there.
    x_left, x_right = np.quantile(data["X"], [0.02, 0.98])
    x_grid = np.linspace(x_left, x_right, 75)

    # Fit the three plug-in estimators on the observed data.
    ate_estimates, cate_estimates = fit_and_evaluate_all(
        data,
        x_grid=x_grid,
        ridge_penalty=float(values["ridge_penalty"]),
    )

    # Compute true values from the simulated potential outcomes.
    ate_truth = true_ate(data)
    cate_truth = true_cate_curve(
        data,
        x_grid=x_grid,
        tau0=float(values["tau0"]),
        tau_x=float(values["tau_x"]),
        tau_x2=float(values["tau_x2"]),
        tau_g=float(values["tau_g"]),
    )

    # Bootstrap intervals.  Set bootstrap repetitions to 0 to turn this off.
    ate_ci, cate_ci = bootstrap_intervals(
        data,
        x_grid=x_grid,
        ridge_penalty=float(values["ridge_penalty"]),
        n_boot=int(values["n_boot"]),
        seed=int(values["seed"]),
    )

    # Merge point estimates, truth, bias, and intervals into one table.
    ate_table = ate_estimates.copy()
    ate_table["truth"] = ate_truth
    ate_table["bias"] = ate_table["estimate"] - ate_truth

    if len(ate_ci) > 0:
        ate_table = ate_table.merge(ate_ci, on="estimator", how="left")
    else:
        ate_table["lower"] = pd.NA
        ate_table["upper"] = pd.NA

    ate_table = ate_table[["estimator", "estimate", "truth", "bias", "lower", "upper"]]

    return ate_ci, ate_estimates, ate_table, ate_truth, cate_ci, cate_estimates, cate_truth, x_grid


@app.cell(hide_code=True)
def _(mo, values):
    ci_text = (
        "Bootstrap intervals are enabled."
        if int(values["n_boot"]) > 0
        else "Bootstrap intervals are turned off. Increase bootstrap repetitions to show intervals."
    )

    mo.md(
        rf"""
        ## 3. ATE estimates

        The target is the finite-sample simulated ATE

        $$
        \psi = \frac{{1}}{{n}}\sum_{{i=1}}^n (Y_i^1-Y_i^0).
        $$

        Each estimator computes

        $$
        \widehat\psi
        =
        \frac{{1}}{{n}}\sum_{{i=1}}^n
        \{{\widehat Q(W_i,1)-\widehat Q(W_i,0)\}}.
        $$

        {ci_text}  The intervals are simple percentile bootstrap intervals.
        """
    )
    return


@app.cell(hide_code=True)
def _(ate_table):
    ate_table.round(3)
    return


@app.cell(hide_code=True)
def _(ate_table, ate_truth, np, plt):
    fig_ate, ax_ate = plt.subplots(figsize=(8, 4.8))

    labels = ate_table["estimator"].tolist()
    estimates = ate_table["estimate"].to_numpy(dtype=float)
    positions = np.arange(len(labels))

    has_intervals = ate_table["lower"].notna().all()
    if has_intervals:
        lower = ate_table["lower"].to_numpy(dtype=float)
        upper = ate_table["upper"].to_numpy(dtype=float)
        yerr = np.vstack([estimates - lower, upper - estimates])
        ax_ate.errorbar(positions, estimates, yerr=yerr, fmt="o", capsize=5, label="estimate with 95% CI")
    else:
        ax_ate.plot(positions, estimates, marker="o", linestyle="", label="estimate")

    ax_ate.axhline(ate_truth, linewidth=2, linestyle="--", label="true ATE")
    ax_ate.set_xticks(positions)
    ax_ate.set_xticklabels(labels)
    ax_ate.set_ylabel("ATE")
    ax_ate.set_title("ATE: point estimates and bootstrap intervals")
    ax_ate.legend()
    fig_ate.tight_layout()
    fig_ate

    return (fig_ate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. CATE curve in the continuous covariate $X$

        The target curve is

        $$
        x \mapsto E[Y^1-Y^0\mid X=x].
        $$

        To estimate this curve, we set $X=x$ for everyone, keep the empirical distribution
        of $Z_1,Z_2,G$, compute the predicted contrast, and average over those remaining covariates.

        The shaded regions are pointwise bootstrap intervals. They are not simultaneous confidence bands.
        """
    )
    return


@app.cell(hide_code=True)
def _(cate_ci, cate_estimates, cate_truth, plt, values, x_grid):
    fig_cate, ax_cate = plt.subplots(figsize=(9, 5.2))

    ax_cate.plot(x_grid, cate_truth, linewidth=2.8, linestyle="--", label="true CATE(x)")

    for name, curve in cate_estimates.items():
        ax_cate.plot(x_grid, curve, linewidth=2.0, label=name)

        if int(values["n_boot"]) > 0 and name in cate_ci:
            ax_cate.fill_between(
                x_grid,
                cate_ci[name]["lower"],
                cate_ci[name]["upper"],
                alpha=0.16,
            )

    ax_cate.set_xlabel("continuous covariate X")
    ax_cate.set_ylabel("CATE(x)")
    ax_cate.set_title("CATE curve: truth, estimates, and pointwise intervals")
    ax_cate.legend()
    fig_cate.tight_layout()
    fig_cate

    return (fig_cate,)


@app.cell(hide_code=True)
def _(ate_table, mo, np, values):
    # A small automatic interpretation of the current run.
    abs_bias = ate_table.assign(abs_bias=lambda d: np.abs(d["bias"].astype(float)))
    best_row = abs_bias.sort_values("abs_bias").iloc[0]
    worst_row = abs_bias.sort_values("abs_bias").iloc[-1]

    mo.md(
        rf"""
        ## Current-run interpretation

        In this run, the smallest absolute ATE bias is from **{best_row['estimator']}**,
        while the largest absolute ATE bias is from **{worst_row['estimator']}**.

        Parameter settings driving the difficulty:

        - selection / lack-of-overlap strength: **{float(values['overlap_strength']):.2f}**;
        - baseline outcome nonlinearity: **{float(values['baseline_nonlinearity']):.2f}**;
        - linear CATE heterogeneity $\tau_X$: **{float(values['tau_x']):.2f}**;
        - nonlinear CATE heterogeneity $\tau_{{X^2}}$: **{float(values['tau_x2']):.2f}**.

        The parametric estimator is expected to struggle when the baseline outcome is nonlinear
        or when the true CATE curve is not approximately flat.  The T-learner can represent more
        heterogeneity, but it may become unstable when overlap is poor because each regression is
        trained on only one treatment arm.
        """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Teaching takeaway

        - Identification gives the formula $E\{Q(W,1)-Q(W,0)\}$, but estimation still requires learning $Q$.
        - A rigid parametric regression can be biased if the outcome surface or treatment-effect heterogeneity is misspecified.
        - The S-learner uses all observations in one regression, which can make it stable, but it depends on how well the single model uses $A$ and its interactions.
        - The T-learner is more flexible for heterogeneous effects, but each model sees only one treatment arm, so it can be unstable under imbalance or poor overlap.
        - Pointwise CATE intervals are uncertainty intervals at each value of $x$ separately; they should not be read as simultaneous bands for the whole curve.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running this notebook

        From the directory containing this file:

        ```bash
        pip install marimo numpy pandas matplotlib
        marimo edit 2_estimation_s_t_parametric_marimo.py
        ```

        To serve it as an app instead of opening the notebook editor:

        ```bash
        marimo run 2_estimation_s_t_parametric_marimo.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
