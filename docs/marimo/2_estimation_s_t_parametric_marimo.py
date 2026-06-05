import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


@app.cell
def _():
    # Only basic packages are used.
    # This keeps the notebook easy to install and easy to explain line by line.
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
        2. an S-learner: one outcome model $\widehat Q(W,A)$;
        3. a T-learner: two outcome models $\widehat Q_1(W)$ and $\widehat Q_0(W)$.

        The data-generating process is designed so that you can create scenarios where the S-learner and
        T-learner behave similarly, and scenarios where they behave very differently.  The key control is
        **S-learner treatment interactions**.  When this is set to zero, the S-learner treats $A$ almost like
        a single additive feature and can miss strong CATE heterogeneity.  When it is increased, the S-learner
        is allowed to look more like a T-learner.
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

        The treatment effect is heterogeneous and can be highly nonlinear:

        $$
        Y^1-Y^0
        =
        \tau_0+\tau_X X+\tau_{X^2}(X^2-1)+\tau_GG
        +\lambda\{1.1\sin(2X)+0.7XG+0.55\sin(Z_1)\}.
        $$

        The parameter $\lambda$ is the **arm-specific CATE complexity**.  Increasing it makes the treated
        outcome surface more different from the control outcome surface.  This is where T-learners can have
        an advantage, provided there is enough data and enough overlap.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, pd):
    # ---------------------------------------------------------------------
    # Small numerical utilities
    # ---------------------------------------------------------------------

    def expit(z):
        """Logistic function: maps any real number to a probability."""
        return 1.0 / (1.0 + np.exp(-np.asarray(z)))


    def as_vector(value, n):
        """Turn a scalar into a length-n vector, or keep an array as an array."""
        if np.isscalar(value):
            return np.full(int(n), float(value))
        return np.asarray(value, dtype=float)


    # ---------------------------------------------------------------------
    # Data simulation
    # ---------------------------------------------------------------------

    def generate_data(
        n,
        seed,
        overlap_strength=1.50,
        treatment_intercept=-0.30,
        baseline_nonlinearity=1.00,
        tau0=1.00,
        tau_x=0.90,
        tau_x2=0.70,
        tau_g=0.70,
        arm_curve_strength=1.50,
        noise_sd=1.00,
    ):
        """
        Simulate one observational data set.

        The analyst observes W, A, and Y.  We also store pi, Y0, Y1, and tau
        because this is a teaching simulation and we want to compare estimators
        with the truth.
        """
        _rng = np.random.default_rng(int(seed))
        _n = int(n)

        # 1. Pre-treatment covariates.
        # X is the focal variable for the CATE plot.
        _X = _rng.normal(loc=0.0, scale=1.0, size=_n)
        _Z1 = _rng.normal(loc=0.0, scale=1.0, size=_n)
        _Z2 = _rng.normal(loc=0.0, scale=1.0, size=_n)
        _G = _rng.binomial(n=1, p=0.45, size=_n)

        # 2. Treatment assignment.
        # The overlap_strength slider controls how strongly W predicts A.
        # Larger values mean worse overlap and more extrapolation.
        _selection_score = (
            0.90 * _X
            - 0.85 * _Z1
            + 0.75 * np.sin(_Z2)
            + 1.00 * (_G - 0.45)
            + 0.25 * _X * _G
        )
        _logit_pi = treatment_intercept + overlap_strength * _selection_score
        _pi = np.clip(expit(_logit_pi), 0.02, 0.98)
        _A = _rng.binomial(n=1, p=_pi)

        # 3. Baseline outcome under control, Y^0.
        # It is nonlinear, so a main-effects linear regression is misspecified.
        _linear_part = 0.65 * _Z1 - 0.35 * _Z2 + 0.60 * _G
        _nonlinear_part = (
            1.20 * np.sin(_X)
            + 0.50 * (_X**2 - 1.0)
            + 0.45 * np.cos(_Z2)
            + 0.35 * _X * _G
            + 0.25 * (_Z1**2 - 1.0)
        )
        _mu0 = 0.20 + _linear_part + baseline_nonlinearity * _nonlinear_part

        # 4. Individual treatment effect tau(W).
        # The last line creates a genuinely arm-specific nonlinear shape.
        # This is the part that makes S- and T-learners separate in some regimes.
        _tau = (
            tau0
            + tau_x * _X
            + tau_x2 * (_X**2 - 1.0)
            + tau_g * _G
            + arm_curve_strength
            * (1.10 * np.sin(2.0 * _X) + 0.70 * _X * _G + 0.55 * np.sin(_Z1))
        )

        # 5. Potential outcomes and observed outcome.
        _epsilon = _rng.normal(loc=0.0, scale=noise_sd, size=_n)
        _Y0 = _mu0 + _epsilon
        _Y1 = _Y0 + _tau
        _Y = _A * _Y1 + (1 - _A) * _Y0

        return pd.DataFrame(
            {
                "X": _X,
                "Z1": _Z1,
                "Z2": _Z2,
                "G": _G,
                "pi": _pi,
                "A": _A,
                "Y": _Y,
                "Y0": _Y0,
                "Y1": _Y1,
                "tau": _tau,
            }
        )


    # ---------------------------------------------------------------------
    # Design matrices for outcome regression
    # ---------------------------------------------------------------------

    def covariate_arrays(data, x_override=None):
        """
        Extract covariates as numpy arrays.

        x_override is used for the CATE curve: we set X=x for everyone,
        while keeping the observed Z1, Z2, and G distribution fixed.
        """
        _n = len(data)
        _X = as_vector(data["X"].to_numpy() if x_override is None else x_override, _n)
        _Z1 = data["Z1"].to_numpy()
        _Z2 = data["Z2"].to_numpy()
        _G = data["G"].to_numpy()
        return _X, _Z1, _Z2, _G


    def linear_design(data, a, x_override=None):
        """
        Rigid parametric design matrix.

        This model is intentionally simple:
            Y ~ 1 + A + X + Z1 + Z2 + G

        It omits nonlinearities and treatment-effect interactions.
        """
        _X, _Z1, _Z2, _G = covariate_arrays(data, x_override=x_override)
        _A = as_vector(a, len(data))
        return np.column_stack([np.ones(len(data)), _A, _X, _Z1, _Z2, _G])


    def flexible_w_basis(data, x_override=None):
        """
        Hand-built nonlinear basis for W.

        This plays the role of a transparent ML feature map.  It includes the
        nonlinear terms used by the data-generating process, but it is still
        simple enough to explain to students.
        """
        _X, _Z1, _Z2, _G = covariate_arrays(data, x_override=x_override)
        return np.column_stack(
            [
                np.ones(len(data)),
                _X,
                _Z1,
                _Z2,
                _G,
                _X**2 - 1.0,
                _Z1**2 - 1.0,
                _Z2**2 - 1.0,
                _X * _Z1,
                _X * _G,
                _Z1 * _G,
                _Z2 * _G,
                np.sin(_X),
                np.sin(2.0 * _X),
                np.cos(_Z2),
                np.sin(_Z1),
            ]
        )


    def s_learner_design(data, a, x_override=None, interaction_level=0):
        """
        Design matrix for the S-learner.

        The S-learner always fits one regression surface.  The interaction_level
        slider controls how much the single surface is allowed to use treatment:

        level 0: baseline basis + A only;
        level 1: baseline basis + A, A*X, A*G;
        level 2: baseline basis + richer treatment interactions.

        Level 0 makes the classic weakness visible: the model may use A only as
        a small additive feature and therefore miss heterogeneous effects.
        """
        _basis = flexible_w_basis(data, x_override=x_override)
        _X, _Z1, _Z2, _G = covariate_arrays(data, x_override=x_override)
        _A = as_vector(a, len(data))
        _level = int(interaction_level)

        if _level <= 0:
            _effect_columns = _A[:, None]
        elif _level == 1:
            _effect_columns = np.column_stack([_A, _A * _X, _A * _G])
        else:
            _effect_basis = np.column_stack(
                [
                    np.ones(len(data)),
                    _X,
                    _G,
                    _X**2 - 1.0,
                    _X * _G,
                    np.sin(2.0 * _X),
                    np.sin(_Z1),
                ]
            )
            _effect_columns = _A[:, None] * _effect_basis

        return np.column_stack([_basis, _effect_columns])


    # ---------------------------------------------------------------------
    # Ridge regression fitting and prediction
    # ---------------------------------------------------------------------

    def fit_ridge(design, outcome, penalty):
        """
        Fit ridge regression by solving the normal equations.

        The intercept is not penalized.  The function is deliberately explicit
        rather than optimized, so students can see the estimator.
        """
        _X = np.asarray(design, dtype=float)
        _y = np.asarray(outcome, dtype=float)
        _penalty = float(penalty)

        _ridge_matrix = np.eye(_X.shape[1])
        _ridge_matrix[0, 0] = 0.0

        _lhs = _X.T @ _X + _penalty * _ridge_matrix
        _rhs = _X.T @ _y

        try:
            return np.linalg.solve(_lhs, _rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(_lhs, _rhs, rcond=None)[0]


    def predict(design, beta):
        """Linear prediction for a fitted coefficient vector."""
        return np.asarray(design, dtype=float) @ np.asarray(beta, dtype=float)


    # ---------------------------------------------------------------------
    # Three plug-in learners
    # ---------------------------------------------------------------------

    def fit_parametric_plugin(data):
        """Fit the deliberately misspecified main-effects linear regression."""
        _design = linear_design(data, a=data["A"].to_numpy())
        _beta = fit_ridge(_design, data["Y"].to_numpy(), penalty=1e-8)
        return {"name": "Parametric", "beta": _beta}


    def fit_s_learner(data, penalty, interaction_level):
        """Fit one outcome model Q(W,A)."""
        _design = s_learner_design(
            data,
            a=data["A"].to_numpy(),
            interaction_level=interaction_level,
        )
        _beta = fit_ridge(_design, data["Y"].to_numpy(), penalty=penalty)
        return {"name": "S-learner", "beta": _beta, "interaction_level": int(interaction_level)}


    def fit_t_learner(data, penalty):
        """Fit separate outcome models in the treated and control arms."""
        _treated_data = data[data["A"] == 1]
        _control_data = data[data["A"] == 0]

        _beta1 = fit_ridge(
            flexible_w_basis(_treated_data),
            _treated_data["Y"].to_numpy(),
            penalty=penalty,
        )
        _beta0 = fit_ridge(
            flexible_w_basis(_control_data),
            _control_data["Y"].to_numpy(),
            penalty=penalty,
        )
        return {"name": "T-learner", "beta1": _beta1, "beta0": _beta0}


    def predict_under_treatment(model, data, a, x_override=None):
        """Predict the outcome under the intervention do(A=a)."""
        if model["name"] == "Parametric":
            return predict(linear_design(data, a=a, x_override=x_override), model["beta"])

        if model["name"] == "S-learner":
            return predict(
                s_learner_design(
                    data,
                    a=a,
                    x_override=x_override,
                    interaction_level=model["interaction_level"],
                ),
                model["beta"],
            )

        if model["name"] == "T-learner":
            _basis = flexible_w_basis(data, x_override=x_override)
            if int(a) == 1:
                return predict(_basis, model["beta1"])
            return predict(_basis, model["beta0"])

        raise ValueError(f"Unknown model type: {model['name']}")


    # ---------------------------------------------------------------------
    # ATE and CATE estimation
    # ---------------------------------------------------------------------

    def estimate_ate(model, data):
        """
        Plug-in ATE estimate.

        For every W_i, predict twice: once under A=1 and once under A=0.
        Then average the predicted differences.
        """
        _q1 = predict_under_treatment(model, data, a=1)
        _q0 = predict_under_treatment(model, data, a=0)
        return float(np.mean(_q1 - _q0))


    def estimate_cate_curve(model, data, x_grid):
        """
        Estimate the marginal CATE curve in X.

        At each x, set X=x for everyone, keep Z1,Z2,G as observed, predict the
        contrast Q(1,x,Z1,Z2,G)-Q(0,x,Z1,Z2,G), and average over Z1,Z2,G.
        """
        _curve = []
        for _x_value in x_grid:
            _q1 = predict_under_treatment(model, data, a=1, x_override=_x_value)
            _q0 = predict_under_treatment(model, data, a=0, x_override=_x_value)
            _curve.append(np.mean(_q1 - _q0))
        return np.asarray(_curve)


    def true_ate(data):
        """True finite-sample ATE, available only in the simulation."""
        return float(data["tau"].mean())


    def true_cate_curve(data, x_grid, tau0, tau_x, tau_x2, tau_g, arm_curve_strength):
        """
        True finite-sample CATE curve marginalized over Z1,Z2,G.

        The DGP makes tau depend on X, G, and Z1.  For each fixed x, we average
        over the empirical distribution of G and Z1.
        """
        _mean_g = float(data["G"].mean())
        _mean_sin_z1 = float(np.sin(data["Z1"]).mean())
        return (
            tau0
            + tau_x * x_grid
            + tau_x2 * (x_grid**2 - 1.0)
            + tau_g * _mean_g
            + arm_curve_strength * (1.10 * np.sin(2.0 * x_grid) + 0.70 * x_grid * _mean_g + 0.55 * _mean_sin_z1)
        )


    def fit_and_evaluate_all(data, x_grid, ridge_penalty, s_interaction_level):
        """Fit all three estimators and return their ATEs and CATE curves."""
        _models = [
            fit_parametric_plugin(data),
            fit_s_learner(data, penalty=ridge_penalty, interaction_level=s_interaction_level),
            fit_t_learner(data, penalty=ridge_penalty),
        ]

        _ate_rows = []
        _cate_curves = {}
        for _model in _models:
            _name = _model["name"]
            _ate_rows.append({"estimator": _name, "estimate": estimate_ate(_model, data)})
            _cate_curves[_name] = estimate_cate_curve(_model, data, x_grid)

        return pd.DataFrame(_ate_rows), _cate_curves


    def bootstrap_intervals(data, x_grid, ridge_penalty, s_interaction_level, n_boot, seed):
        """
        Nonparametric bootstrap intervals.

        These are simple percentile intervals.  For the CATE curves they are
        pointwise intervals, not simultaneous confidence bands.
        """
        _n_boot = int(n_boot)
        if _n_boot <= 0:
            _empty_ate = pd.DataFrame(columns=["estimator", "lower", "upper"])
            return _empty_ate, {}

        _rng = np.random.default_rng(int(seed) + 10_000)
        _n = len(data)
        _names = ["Parametric", "S-learner", "T-learner"]
        _ate_samples = {_name: [] for _name in _names}
        _curve_samples = {_name: [] for _name in _names}

        for _ in range(_n_boot):
            _sample_ids = _rng.integers(low=0, high=_n, size=_n)
            _boot_data = data.iloc[_sample_ids].reset_index(drop=True)

            _boot_ate, _boot_curves = fit_and_evaluate_all(
                _boot_data,
                x_grid=x_grid,
                ridge_penalty=ridge_penalty,
                s_interaction_level=s_interaction_level,
            )

            for _name in _names:
                _value = _boot_ate.loc[_boot_ate["estimator"] == _name, "estimate"].iloc[0]
                _ate_samples[_name].append(_value)
                _curve_samples[_name].append(_boot_curves[_name])

        _ate_rows = []
        _curve_intervals = {}
        for _name in _names:
            _ate_array = np.asarray(_ate_samples[_name])
            _ate_rows.append(
                {
                    "estimator": _name,
                    "lower": np.quantile(_ate_array, 0.025),
                    "upper": np.quantile(_ate_array, 0.975),
                }
            )

            _curve_array = np.vstack(_curve_samples[_name])
            _curve_intervals[_name] = {
                "lower": np.quantile(_curve_array, 0.025, axis=0),
                "upper": np.quantile(_curve_array, 0.975, axis=0),
            }

        return pd.DataFrame(_ate_rows), _curve_intervals


    def treatment_summary(data):
        """Compact table showing treatment imbalance and overlap."""
        _treated = data["A"] == 1
        _control = data["A"] == 0
        _pi = data["pi"]
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
                    data.loc[_treated, "X"].mean(),
                    data.loc[_control, "X"].mean(),
                    data.loc[_treated, "Z1"].mean(),
                    data.loc[_control, "Z1"].mean(),
                    data.loc[_treated, "G"].mean(),
                    data.loc[_control, "G"].mean(),
                    _pi.quantile(0.10),
                    _pi.quantile(0.90),
                    ((_pi > 0.1) & (_pi < 0.9)).mean(),
                ],
            }
        )


    def bin_means(x, y, n_bins=18):
        """Compute binned means for simple diagnostic plots."""
        _edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        _edges[0] -= 1e-8
        _edges[-1] += 1e-8
        _bin_id = np.digitize(x, _edges) - 1

        _rows = []
        for _j in range(n_bins):
            _in_bin = _bin_id == _j
            if _in_bin.sum() > 0:
                _rows.append(
                    {
                        "x_mean": np.mean(x[_in_bin]),
                        "y_mean": np.mean(y[_in_bin]),
                        "n": int(_in_bin.sum()),
                    }
                )
        return pd.DataFrame(_rows)


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

        1. **Make S and T different:** set S-learner treatment interactions to 0, increase arm-specific CATE complexity,
           and increase nonlinear heterogeneity.
        2. **Make S and T similar:** set S-learner treatment interactions to 2 and keep overlap reasonable.
        3. **Make T unstable:** reduce sample size and increase selection / lack-of-overlap strength.
        4. **Make the parametric model fail:** increase baseline nonlinearity and nonlinear treatment-effect heterogeneity.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Every slider controls one feature of the simulation or estimation method.
    est_controls = mo.ui.dictionary(
        {
            "n": mo.ui.slider(
                start=400,
                stop=10_000,
                step=200,
                value=2_000,
                label="sample size",
            ),
            "overlap_strength": mo.ui.slider(
                start=0.0,
                stop=4.0,
                step=0.05,
                value=1.50,
                label="selection / lack-of-overlap strength",
            ),
            "treatment_intercept": mo.ui.slider(
                start=-2.0,
                stop=2.0,
                step=0.05,
                value=-0.30,
                label="treatment intercept / treatment prevalence",
            ),
            "baseline_nonlinearity": mo.ui.slider(
                start=0.0,
                stop=2.5,
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
                start=-2.0,
                stop=2.0,
                step=0.05,
                value=0.90,
                label="linear heterogeneity tau_X",
            ),
            "tau_x2": mo.ui.slider(
                start=-1.5,
                stop=1.5,
                step=0.05,
                value=0.70,
                label="nonlinear heterogeneity tau_X2",
            ),
            "tau_g": mo.ui.slider(
                start=-1.0,
                stop=2.0,
                step=0.05,
                value=0.70,
                label="group heterogeneity tau_G",
            ),
            "arm_curve_strength": mo.ui.slider(
                start=0.0,
                stop=2.5,
                step=0.05,
                value=1.50,
                label="arm-specific CATE complexity lambda",
            ),
            "s_interaction_level": mo.ui.slider(
                start=0,
                stop=2,
                step=1,
                value=0,
                label="S-learner treatment interactions: 0 none, 1 simple, 2 rich",
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
                stop=30.0,
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
    est_controls.vstack()
    return (est_controls,)


@app.cell(hide_code=True)
def _(est_controls, generate_data):
    # Read current slider values.
    est_values = est_controls.value

    # Generate a new data set whenever a slider changes.
    est_data = generate_data(
        n=int(est_values["n"]),
        seed=int(est_values["seed"]),
        overlap_strength=float(est_values["overlap_strength"]),
        treatment_intercept=float(est_values["treatment_intercept"]),
        baseline_nonlinearity=float(est_values["baseline_nonlinearity"]),
        tau0=float(est_values["tau0"]),
        tau_x=float(est_values["tau_x"]),
        tau_x2=float(est_values["tau_x2"]),
        tau_g=float(est_values["tau_g"]),
        arm_curve_strength=float(est_values["arm_curve_strength"]),
        noise_sd=float(est_values["noise_sd"]),
    )

    # Display the first few rows so students see the variables.
    est_data.head()
    return est_data, est_values


@app.cell(hide_code=True)
def _(est_data, treatment_summary):
    overlap_table = treatment_summary(est_data)
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
def _(bin_means, est_data, plt):
    # Plot empirical treatment probability against X, separately by G.
    _fig, _ax = plt.subplots(figsize=(8, 4.8))

    for _g_value in [0, 1]:
        _group_data = est_data[est_data["G"] == _g_value]
        _binned = bin_means(
            _group_data["X"].to_numpy(),
            _group_data["A"].to_numpy(),
            n_bins=16,
        )
        _ax.plot(
            _binned["x_mean"],
            _binned["y_mean"],
            marker="o",
            label=f"G={_g_value}",
        )

    _ax.set_ylim(-0.02, 1.02)
    _ax.set_xlabel("continuous covariate X")
    _ax.set_ylabel("empirical Pr(A=1 | X bin, G)")
    _ax.set_title("Treatment selection as a function of pre-treatment covariates")
    _ax.legend(title="group")
    _fig.tight_layout()
    _fig

    return


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

        **S-learner:** fit one regression surface $\widehat Q(W,A)$.  The interaction slider controls whether
        treatment enters only as $A$, as $A$ plus simple interactions, or with richer treatment interactions.

        **T-learner:** fit two regression surfaces: one among treated observations and one among controls.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    bootstrap_intervals,
    est_data,
    est_values,
    fit_and_evaluate_all,
    np,
    pd,
    true_ate,
    true_cate_curve,
):
    # Evaluate the CATE curve on the central part of the X distribution.
    _x_left, _x_right = np.quantile(est_data["X"], [0.02, 0.98])
    x_grid = np.linspace(_x_left, _x_right, 80)

    # Fit the three estimators on the observed data.
    ate_estimates, cate_estimates = fit_and_evaluate_all(
        est_data,
        x_grid=x_grid,
        ridge_penalty=float(est_values["ridge_penalty"]),
        s_interaction_level=int(est_values["s_interaction_level"]),
    )

    # Compute the true finite-sample ATE and CATE curve.
    ate_truth = true_ate(est_data)
    cate_truth = true_cate_curve(
        est_data,
        x_grid=x_grid,
        tau0=float(est_values["tau0"]),
        tau_x=float(est_values["tau_x"]),
        tau_x2=float(est_values["tau_x2"]),
        tau_g=float(est_values["tau_g"]),
        arm_curve_strength=float(est_values["arm_curve_strength"]),
    )

    # Bootstrap intervals.  Set n_boot to 0 to turn this off.
    ate_ci, cate_ci = bootstrap_intervals(
        est_data,
        x_grid=x_grid,
        ridge_penalty=float(est_values["ridge_penalty"]),
        s_interaction_level=int(est_values["s_interaction_level"]),
        n_boot=int(est_values["n_boot"]),
        seed=int(est_values["seed"]),
    )

    # Merge point estimates, truth, bias, bootstrap intervals, and CATE RMSE.
    ate_table = ate_estimates.copy()
    ate_table["truth"] = ate_truth
    ate_table["bias"] = ate_table["estimate"] - ate_truth
    ate_table["CATE RMSE"] = [
        np.sqrt(np.mean((cate_estimates[_estimator] - cate_truth) ** 2))
        for _estimator in ate_table["estimator"]
    ]

    if len(ate_ci) > 0:
        ate_table = ate_table.merge(ate_ci, on="estimator", how="left")
    else:
        ate_table["lower"] = pd.NA
        ate_table["upper"] = pd.NA

    ate_table = ate_table[
        ["estimator", "estimate", "truth", "bias", "CATE RMSE", "lower", "upper"]
    ]

    return ate_ci, ate_estimates, ate_table, ate_truth, cate_ci, cate_estimates, cate_truth, x_grid


@app.cell(hide_code=True)
def _(est_values, mo):
    _ci_text = (
        "Bootstrap intervals are enabled."
        if int(est_values["n_boot"]) > 0
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

        {_ci_text}  The intervals are simple percentile bootstrap intervals.  The table also reports
        CATE RMSE against the true curve, which helps quantify differences between S- and T-learners.
        """
    )
    return


@app.cell(hide_code=True)
def _(ate_table):
    ate_table.round(3)
    return


@app.cell(hide_code=True)
def _(ate_table, ate_truth, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4.8))

    _labels = ate_table["estimator"].tolist()
    _estimates = ate_table["estimate"].to_numpy(dtype=float)
    _positions = np.arange(len(_labels))

    _has_intervals = ate_table["lower"].notna().all()
    if _has_intervals:
        _lower = ate_table["lower"].to_numpy(dtype=float)
        _upper = ate_table["upper"].to_numpy(dtype=float)
        _yerr = np.vstack([_estimates - _lower, _upper - _estimates])
        _ax.errorbar(
            _positions,
            _estimates,
            yerr=_yerr,
            fmt="o",
            capsize=5,
            label="estimate with 95% CI",
        )
    else:
        _ax.plot(_positions, _estimates, marker="o", linestyle="", label="estimate")

    _ax.axhline(ate_truth, linewidth=2, linestyle="--", label="true ATE")
    _ax.set_xticks(_positions)
    _ax.set_xticklabels(_labels)
    _ax.set_ylabel("ATE")
    _ax.set_title("ATE: point estimates and bootstrap intervals")
    _ax.legend()
    _fig.tight_layout()
    _fig

    return


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
def _(cate_ci, cate_estimates, cate_truth, est_values, plt, x_grid):
    _fig, _ax = plt.subplots(figsize=(9, 5.2))

    _ax.plot(x_grid, cate_truth, linewidth=2.8, linestyle="--", label="true CATE(x)")

    for _name, _curve in cate_estimates.items():
        _ax.plot(x_grid, _curve, linewidth=2.0, label=_name)

        if int(est_values["n_boot"]) > 0 and _name in cate_ci:
            _ax.fill_between(
                x_grid,
                cate_ci[_name]["lower"],
                cate_ci[_name]["upper"],
                alpha=0.16,
            )

    _ax.set_xlabel("continuous covariate X")
    _ax.set_ylabel("CATE(x)")
    _ax.set_title("CATE curve: truth, estimates, and pointwise intervals")
    _ax.legend()
    _fig.tight_layout()
    _fig

    return


@app.cell(hide_code=True)
def _(ate_table, est_values, mo, np):
    # A small automatic interpretation of the current run.
    _abs_bias = ate_table.assign(abs_bias=lambda _d: np.abs(_d["bias"].astype(float)))
    _best_row = _abs_bias.sort_values("abs_bias").iloc[0]
    _worst_row = _abs_bias.sort_values("abs_bias").iloc[-1]
    _best_cate_row = ate_table.sort_values("CATE RMSE").iloc[0]

    mo.md(
        rf"""
        ## Current-run interpretation

        In this run, the smallest absolute ATE bias is from **{_best_row['estimator']}**,
        while the largest absolute ATE bias is from **{_worst_row['estimator']}**.
        The smallest CATE RMSE is from **{_best_cate_row['estimator']}**.

        Settings driving S- versus T-learner differences:

        - S-learner treatment interactions: **{int(est_values['s_interaction_level'])}**;
        - arm-specific CATE complexity $\lambda$: **{float(est_values['arm_curve_strength']):.2f}**;
        - selection / lack-of-overlap strength: **{float(est_values['overlap_strength']):.2f}**;
        - nonlinear heterogeneity $\tau_{{X^2}}$: **{float(est_values['tau_x2']):.2f}**.

        A good way to make the S-learner and T-learner separate is to set interaction level to 0,
        increase $\lambda$, and keep enough sample size for the T-learner to fit each treatment arm.
        A good way to make them similar is to set interaction level to 2.
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
        - The S-learner uses one regression.  This can be stable, but if treatment enters too weakly, it can shrink heterogeneous effects toward a flat effect.
        - The T-learner fits one regression per arm.  This can recover arm-specific outcome surfaces, but it uses fewer observations per regression and can be unstable under imbalance or poor overlap.
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
        marimo edit 2_estimation_s_t_parametric_marimo_more_contrast.py
        ```

        To serve it as an app instead of opening the notebook editor:

        ```bash
        marimo run 2_estimation_s_t_parametric_marimo_more_contrast.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
