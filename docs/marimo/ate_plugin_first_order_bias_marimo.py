import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Plug-in ATE estimation with flexible ML: where first-order bias appears

        This notebook simulates a simple observational causal inference problem with a binary treatment.
        The target is the average treatment effect

        $$
        \psi = \mathbb E\{\mu_1(W)-\mu_0(W)\},
        \qquad
        \mu_a(w)=\mathbb E(Y\mid A=a,W=w).
        $$

        We compare two estimators that use the same flexible outcome-regression fits:

        $$
        \widehat\psi_{\mathrm{plug}}=\mathbb P_n\{\widehat\mu_1(W)-\widehat\mu_0(W)\},
        $$

        and the cross-fitted AIPW / one-step estimator with the known propensity score $\pi(W)$,

        $$
        \widehat\psi_{\mathrm{aipw}}
        =\mathbb P_n\left[
        \widehat\mu_1(W)-\widehat\mu_0(W)
        +\frac{A}{\pi(W)}\{Y-\widehat\mu_1(W)\}
        -\frac{1-A}{1-\pi(W)}\{Y-\widehat\mu_0(W)\}
        \right].
        $$

        The point is not that random forests or boosting are bad. The point is that the **plain plug-in estimator is not Neyman-orthogonal**: its error contains a linear term in the outcome-regression error. The AIPW correction cancels this first-order term when the propensity score is correct.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.model_selection import KFold

    return (
        ExtraTreesRegressor,
        KFold,
        RandomForestRegressor,
        clone,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data-generating process

        We observe $O=(W,A,Y)$, where $W\in\mathbb R^5$, treatment is binary, and

        $$
        A\mid W\sim \mathrm{Bernoulli}\{\pi(W)\},
        \qquad
        Y=\mu_0(W)+A\tau(W)+\varepsilon.
        $$

        The functions $\pi(W)$, $\mu_0(W)$, and $\tau(W)$ are nonlinear. This makes flexible ML useful, but it also creates finite-sample regularization/smoothing error in $\widehat\mu_a$.
        """
    )
    return


@app.cell
def _(np):
    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))


    def dgp_functions(W):
        """Return propensity, mu0, mu1, and heterogeneous treatment effect tau."""
        w0, w1, w2, w3, w4 = W.T

        propensity = expit(
            -0.15
            + 0.65 * w0
            - 0.55 * w1
            + 0.35 * np.sin(w2)
            + 0.20 * w0 * w1
        )
        propensity = np.clip(propensity, 0.05, 0.95)

        mu0 = (
            np.sin(w0)
            + 0.50 * (w1**2 - 1.0)
            - 0.40 * w2
            + 0.35 * w3 * w4
        )
        tau = 1.0 + 0.50 * np.sin(w0 + w1) + 0.30 * (w3 > 0) - 0.20 * w4
        mu1 = mu0 + tau
        return propensity, mu0, mu1, tau


    def generate_observed_data(n, rng, noise_sd=1.0):
        W = rng.normal(size=(n, 5))
        propensity, mu0, mu1, tau = dgp_functions(W)
        A = rng.binomial(1, propensity)
        Y = mu0 + A * tau + rng.normal(loc=0.0, scale=noise_sd, size=n)
        return W, A, Y, propensity, mu0, mu1, tau


    def monte_carlo_true_ate(mc_size=300_000, seed=2026):
        rng = np.random.default_rng(seed)
        W_mc = rng.normal(size=(mc_size, 5))
        _, _, _, tau_mc = dgp_functions(W_mc)
        return float(np.mean(tau_mc))


    true_ate = monte_carlo_true_ate()
    true_ate

    return (
        dgp_functions,
        expit,
        generate_observed_data,
        monte_carlo_true_ate,
        true_ate,
    )


@app.cell(hide_code=True)
def _(mo, true_ate):
    mo.md(
        rf"""
        The population ATE, computed by a large Monte Carlo approximation from the known DGP, is

        $$
        \psi \approx {true_ate:.4f}.
        $$
        """
    ).callout(kind="info")
    return


@app.cell
def _(ExtraTreesRegressor, KFold, RandomForestRegressor, clone, np, pd):
    def make_regressor(model_name, n_trees, min_leaf, max_depth, random_state):
        depth = None if max_depth == 0 else int(max_depth)
        if model_name == "Random forest":
            return RandomForestRegressor(
                n_estimators=int(n_trees),
                min_samples_leaf=int(min_leaf),
                max_depth=depth,
                max_features=1.0,
                bootstrap=True,
                n_jobs=-1,
                random_state=int(random_state),
            )
        if model_name == "Extra trees":
            return ExtraTreesRegressor(
                n_estimators=int(n_trees),
                min_samples_leaf=int(min_leaf),
                max_depth=depth,
                max_features=1.0,
                bootstrap=False,
                n_jobs=-1,
                random_state=int(random_state),
            )
        raise ValueError(f"Unknown model_name={model_name!r}")


    def crossfit_t_learner_predictions(W, A, Y, base_model, n_splits=2, seed=1):
        """Cross-fitted T-learner predictions for mu_0(W) and mu_1(W)."""
        n = W.shape[0]
        q0_hat = np.empty(n)
        q1_hat = np.empty(n)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

        for fold_id, (train_idx, test_idx) in enumerate(folds.split(W)):
            for a_value, storage in [(0, q0_hat), (1, q1_hat)]:
                arm_train_idx = train_idx[A[train_idx] == a_value]
                if len(arm_train_idx) < 20:
                    # Defensive fallback for very small samples or extreme propensities.
                    storage[test_idx] = np.mean(Y[train_idx])
                    continue

                model = clone(base_model)
                if hasattr(model, "random_state"):
                    model.set_params(random_state=int(seed + 1000 * fold_id + a_value))
                model.fit(W[arm_train_idx], Y[arm_train_idx])
                storage[test_idx] = model.predict(W[test_idx])

        return q0_hat, q1_hat


    def estimate_once(
        n,
        seed,
        true_ate,
        generate_observed_data,
        model_name="Random forest",
        n_trees=100,
        min_leaf=5,
        max_depth=0,
        noise_sd=1.0,
    ):
        rng = np.random.default_rng(int(seed))
        W, A, Y, propensity, mu0, mu1, tau = generate_observed_data(
            int(n), rng, noise_sd=float(noise_sd)
        )
        base_model = make_regressor(
            model_name=model_name,
            n_trees=n_trees,
            min_leaf=min_leaf,
            max_depth=max_depth,
            random_state=seed,
        )
        q0_hat, q1_hat = crossfit_t_learner_predictions(
            W, A, Y, base_model, n_splits=2, seed=seed
        )

        plugin = float(np.mean(q1_hat - q0_hat))
        aipw = float(
            np.mean(
                q1_hat
                - q0_hat
                + A / propensity * (Y - q1_hat)
                - (1 - A) / (1 - propensity) * (Y - q0_hat)
            )
        )
        oracle_sample = float(np.mean(tau))
        rmse_q = float(
            np.sqrt(np.mean((q0_hat - mu0) ** 2) + np.mean((q1_hat - mu1) ** 2))
        )

        return {
            "plug-in": plugin,
            "AIPW / one-step": aipw,
            "oracle sample ATE": oracle_sample,
            "plug-in error": plugin - true_ate,
            "AIPW error": aipw - true_ate,
            "oracle sample error": oracle_sample - true_ate,
            "Q RMSE": rmse_q,
        }


    def run_monte_carlo(
        n,
        reps,
        seed,
        true_ate,
        generate_observed_data,
        model_name,
        n_trees,
        min_leaf,
        max_depth,
        noise_sd,
    ):
        rows = []
        for rep in range(int(reps)):
            rows.append(
                estimate_once(
                    n=n,
                    seed=int(seed) + rep,
                    true_ate=true_ate,
                    generate_observed_data=generate_observed_data,
                    model_name=model_name,
                    n_trees=n_trees,
                    min_leaf=min_leaf,
                    max_depth=max_depth,
                    noise_sd=noise_sd,
                )
            )
        return pd.DataFrame(rows)


    return (
        crossfit_t_learner_predictions,
        estimate_once,
        make_regressor,
        run_monte_carlo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Monte Carlo: same flexible ML nuisance, different target estimator

        The next cell fits flexible outcome-regression models separately in the treated and control groups using cross-fitting.
        The plug-in and AIPW estimators use exactly the same fitted $\widehat\mu_0,\widehat\mu_1$.
        """
    )
    return


@app.cell
def _(mo):
    simulation_controls = mo.ui.dictionary(
        {
            "sample size n": mo.ui.slider(
                start=500, stop=5000, step=500, value=1500, label="sample size n"
            ),
            "Monte Carlo reps": mo.ui.slider(
                start=20, stop=250, step=10, value=80, label="Monte Carlo reps"
            ),
            "model": mo.ui.dropdown(
                options=["Random forest", "Extra trees"], value="Random forest", label="ML model"
            ),
            "trees": mo.ui.slider(start=50, stop=300, step=50, value=100, label="trees"),
            "min leaf": mo.ui.slider(
                start=1, stop=30, step=1, value=5, label="minimum leaf size"
            ),
            "max depth": mo.ui.slider(
                start=0,
                stop=20,
                step=1,
                value=0,
                label="max depth; 0 means unlimited",
            ),
            "noise sd": mo.ui.slider(
                start=0.25, stop=2.0, step=0.25, value=1.0, label="outcome noise SD"
            ),
            "seed": mo.ui.number(start=1, stop=999999, value=12345, label="seed"),
        },
        label="Simulation controls",
    )
    run_button = mo.ui.run_button(label="Run Monte Carlo", kind="success")
    mo.vstack([simulation_controls.vstack(), run_button])
    return run_button, simulation_controls


@app.cell
def _(
    generate_observed_data,
    mo,
    run_button,
    run_monte_carlo,
    simulation_controls,
    true_ate,
):
    mo.stop(
        not run_button.value,
        mo.md("Choose the simulation settings, then click **Run Monte Carlo**."),
    )

    control_values = simulation_controls.value
    mc_results = run_monte_carlo(
        n=int(control_values["sample size n"]),
        reps=int(control_values["Monte Carlo reps"]),
        seed=int(control_values["seed"]),
        true_ate=true_ate,
        generate_observed_data=generate_observed_data,
        model_name=control_values["model"],
        n_trees=int(control_values["trees"]),
        min_leaf=int(control_values["min leaf"]),
        max_depth=int(control_values["max depth"]),
        noise_sd=float(control_values["noise sd"]),
    )
    mc_results

    return control_values, mc_results


@app.cell
def _(mc_results, pd):
    mc_summary = pd.DataFrame(
        {
            "mean estimate": [
                mc_results["plug-in"].mean(),
                mc_results["AIPW / one-step"].mean(),
                mc_results["oracle sample ATE"].mean(),
            ],
            "bias vs population ATE": [
                mc_results["plug-in error"].mean(),
                mc_results["AIPW error"].mean(),
                mc_results["oracle sample error"].mean(),
            ],
            "Monte Carlo sd": [
                mc_results["plug-in"].std(),
                mc_results["AIPW / one-step"].std(),
                mc_results["oracle sample ATE"].std(),
            ],
        },
        index=["plug-in", "AIPW / one-step", "oracle sample ATE"],
    )
    mc_summary

    return (mc_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
        ### Monte Carlo summary

        The plug-in estimator usually has visibly larger bias because the smoothing/regularization error in $\widehat\mu_1-\widehat\mu_0$ is averaged directly into the target estimate. The AIPW estimator uses the same $\widehat\mu_a$ but subtracts the leading bias through weighted residuals.
        """
    )
    return


@app.cell
def _(mc_summary):
    mc_summary.round(4)
    return


@app.cell
def _(mc_results, plt, true_ate):
    error_data = [mc_results["plug-in error"], mc_results["AIPW error"]]
    fig_error_box, ax_error_box = plt.subplots(figsize=(7.5, 4.5))
    ax_error_box.axhline(0.0, linewidth=1)
    ax_error_box.boxplot(error_data, labels=["plug-in", "AIPW / one-step"], showmeans=True)
    ax_error_box.set_ylabel("estimator error: estimate - true ATE")
    ax_error_box.set_title(f"Monte Carlo errors around true ATE = {true_ate:.3f}")
    fig_error_box.tight_layout()
    fig_error_box

    return ax_error_box, error_data, fig_error_box


@app.cell
def _(mc_results, plt):
    fig_q_bias, ax_q_bias = plt.subplots(figsize=(7.5, 4.5))
    ax_q_bias.scatter(mc_results["Q RMSE"], mc_results["plug-in error"], alpha=0.7, label="plug-in")
    ax_q_bias.scatter(mc_results["Q RMSE"], mc_results["AIPW error"], alpha=0.7, label="AIPW / one-step")
    ax_q_bias.axhline(0.0, linewidth=1)
    ax_q_bias.set_xlabel("cross-fitted outcome-regression RMSE")
    ax_q_bias.set_ylabel("estimator error")
    ax_q_bias.set_title("Plug-in error tracks first-order outcome-regression error")
    ax_q_bias.legend()
    fig_q_bias.tight_layout()
    fig_q_bias

    return ax_q_bias, fig_q_bias


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Controlled perturbation: directly visualizing first-order bias

        To isolate the mechanism from random forest details, now pretend the outcome regressions are known up to a systematic perturbation:

        $$
        \widehat\mu_0(w)=\mu_0(w)+\delta b_0(w),
        \qquad
        \widehat\mu_1(w)=\mu_1(w)+\delta b_1(w).
        $$

        Then

        $$
        \widehat\psi_{\mathrm{plug}}-\psi
        \approx
        \delta\,\mathbb E\{b_1(W)-b_0(W)\},
        $$

        so the plug-in bias is **linear** in the regression error. By contrast, with the correct propensity score,

        $$
        \mathbb E\left[
        \widehat\mu_1(W)-\widehat\mu_0(W)
        +\frac{A}{\pi(W)}\{Y-\widehat\mu_1(W)\}
        -\frac{1-A}{1-\pi(W)}\{Y-\widehat\mu_0(W)\}
        \mid W
        \right]
        =\mu_1(W)-\mu_0(W),
        $$

        so the first-order regression error cancels.
        """
    )
    return


@app.cell
def _(generate_observed_data, np, pd):
    def perturbation_experiment(generate_observed_data, n=60_000, seed=999):
        rng = np.random.default_rng(seed)
        W, A, Y, propensity, mu0, mu1, tau = generate_observed_data(n, rng, noise_sd=1.0)
        w0, w1, w2, w3, w4 = W.T

        # b1 - b0 has nonzero mean, making the plug-in bias visible.
        b0 = 0.25 * (w0**2 - 1.0) + 0.20 * np.sin(w1)
        b1 = 1.0 + 0.25 * np.sin(w2) - 0.15 * w3

        sample_ate = np.mean(tau)
        rows = []
        for delta in np.linspace(-1.0, 1.0, 41):
            q0_delta = mu0 + delta * b0
            q1_delta = mu1 + delta * b1
            plugin_delta = np.mean(q1_delta - q0_delta)

            # This is the conditional expectation of the AIPW score given W and the nuisance fits.
            aipw_expected_delta = np.mean(
                q1_delta - q0_delta + (mu1 - q1_delta) - (mu0 - q0_delta)
            )

            # This is the realized finite-sample AIPW value using the observed data.
            aipw_realized_delta = np.mean(
                q1_delta
                - q0_delta
                + A / propensity * (Y - q1_delta)
                - (1 - A) / (1 - propensity) * (Y - q0_delta)
            )

            rows.append(
                {
                    "delta": delta,
                    "plug-in error": plugin_delta - sample_ate,
                    "expected AIPW error": aipw_expected_delta - sample_ate,
                    "realized AIPW error": aipw_realized_delta - sample_ate,
                }
            )
        return pd.DataFrame(rows)


    perturbation_results = perturbation_experiment(generate_observed_data)
    perturbation_results.head()

    return perturbation_experiment, perturbation_results


@app.cell
def _(perturbation_results, plt):
    fig_perturbation, ax_perturbation = plt.subplots(figsize=(8, 4.8))
    ax_perturbation.axhline(0.0, linewidth=1)
    ax_perturbation.plot(
        perturbation_results["delta"],
        perturbation_results["plug-in error"],
        marker="o",
        label="plug-in",
    )
    ax_perturbation.plot(
        perturbation_results["delta"],
        perturbation_results["expected AIPW error"],
        marker="s",
        label="AIPW: conditional expectation",
    )
    ax_perturbation.plot(
        perturbation_results["delta"],
        perturbation_results["realized AIPW error"],
        linestyle="--",
        label="AIPW: realized finite sample",
    )
    ax_perturbation.set_xlabel("perturbation size delta")
    ax_perturbation.set_ylabel("error relative to sample ATE")
    ax_perturbation.set_title("Plug-in bias is first-order; AIPW removes the linear term")
    ax_perturbation.legend()
    fig_perturbation.tight_layout()
    fig_perturbation

    return ax_perturbation, fig_perturbation


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Takeaway

        The plug-in estimator is sensitive to the first-order error
        $\mathbb P_n[(\widehat\mu_1-\mu_1)-(\widehat\mu_0-\mu_0)]$.
        Flexible ML can reduce this error, but in finite samples it can also introduce systematic smoothing or regularization bias.

        The AIPW / one-step estimator adds the efficient influence-function correction. With a correct propensity score and cross-fitting, the leading regression error cancels, leaving a second-order product-type remainder rather than a first-order regression-bias term.
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
        pip install marimo numpy pandas matplotlib scikit-learn
        marimo edit ate_plugin_first_order_bias_marimo.py
        ```

        To serve it as an app instead of opening the notebook editor:

        ```bash
        marimo run ate_plugin_first_order_bias_marimo.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
