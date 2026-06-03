import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Causal effects, queries, and estimands: a visual simulation

        This notebook uses a small structural simulation to make the standard causal estimands visible.
        We generate a binary treatment $A$, a continuous pre-treatment covariate $X$, a discrete
        pre-treatment covariate $G$, and two potential outcomes $Y^0,Y^1$.

        The notebook visualizes:

        - the marginal observed distribution $P(Y)$,
        - conditional observed distributions such as $P(Y\mid A=1)$ and $P(Y\mid A=0)$,
        - post-intervention distributions $P(Y\mid do(A=a)) \equiv P(Y^a)$,
        - expected potential outcomes $E[Y^a]$,
        - the ATE, CATE, and ATT.

        The key teaching contrast is:

        $$
        P(Y\mid A=a) \neq P(Y\mid do(A=a))
        $$

        in observational data with confounding. Use the **confounding strength** slider to see how the
        conditional observed bell curves separate from the post-intervention bell curves.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data-generating process

        The simulation follows the simple SCM

        $$
        X \leftarrow U_X,\qquad
        G \leftarrow U_G,\qquad
        A \leftarrow \mathrm{Bernoulli}\{\pi(X,G)\},\qquad
        Y \leftarrow Y^A.
        $$

        The potential outcomes are

        $$
        Y^0 = \mu_0(X,G) + \varepsilon,
        \qquad
        Y^1 = Y^0 + \tau(X,G),
        $$

        with heterogeneous treatment effect

        $$
        \tau(X,G)=\tau_0 + \tau_X X + \tau_G G.
        $$

        Here $X$ and $G$ are pre-treatment variables, so quantities such as
        $E[Y^1-Y^0\mid X=x]$ and $E[Y^1-Y^0\mid G=g]$ are well-defined CATEs.
        """
    )
    return


@app.cell
def _(np, pd):
    def expit(z):
        return 1.0 / (1.0 + np.exp(-z))


    def generate_population(
        n,
        seed,
        confounding_strength=1.25,
        baseline_x_strength=1.00,
        tau_x=0.40,
        tau_g=0.60,
        tau0=1.00,
        treatment_intercept=-0.15,
        noise_sd=1.00,
    ):
        """Generate one finite super-population with observed and potential outcomes."""
        rng = np.random.default_rng(int(seed))

        X = rng.normal(loc=0.0, scale=1.0, size=int(n))
        G = rng.binomial(n=1, p=0.45, size=int(n))

        # Treatment assignment. Increasing confounding_strength makes A more selected by X and G.
        logit_pi = (
            treatment_intercept
            + confounding_strength * (0.90 * X + 0.85 * (G - 0.45))
        )
        pi = np.clip(expit(logit_pi), 0.02, 0.98)
        A = rng.binomial(n=1, p=pi)

        # Outcome mechanism. X and G affect both treatment and outcome when confounding_strength > 0.
        mu0 = (
            0.20
            + baseline_x_strength * (0.90 * X + 0.30 * (X**2 - 1.0))
            + 0.70 * G
            + 0.25 * np.sin(1.5 * X)
        )
        tau = tau0 + tau_x * X + tau_g * G

        epsilon = rng.normal(loc=0.0, scale=noise_sd, size=int(n))
        Y0 = mu0 + epsilon
        Y1 = Y0 + tau
        Y = A * Y1 + (1 - A) * Y0

        data = pd.DataFrame(
            {
                "X": X,
                "G": G,
                "pi": pi,
                "A": A,
                "Y": Y,
                "Y0": Y0,
                "Y1": Y1,
                "tau": tau,
                "mu0": mu0,
            }
        )
        return data


    def smooth_density(values, grid, bins=170):
        """Fast smoothed histogram density; avoids extra dependencies beyond numpy."""
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if len(values) < 5:
            return np.zeros_like(grid)

        left, right = float(np.min(grid)), float(np.max(grid))
        hist, edges = np.histogram(values, bins=bins, range=(left, right), density=False)
        bin_width = edges[1] - edges[0]

        sd = np.std(values)
        bandwidth = max(1.06 * sd * (len(values) ** (-1 / 5)), 1.5 * bin_width)
        sigma_bins = max(bandwidth / bin_width, 1.0)
        radius = int(np.ceil(4.0 * sigma_bins))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()

        smoothed = np.convolve(hist, kernel, mode="same")
        density_at_centers = smoothed / (len(values) * bin_width)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, density_at_centers, left=0.0, right=0.0)


    def estimand_summary(data):
        ate = data["tau"].mean()
        att = data.loc[data["A"] == 1, "tau"].mean()
        atc = data.loc[data["A"] == 0, "tau"].mean()
        e_y0 = data["Y0"].mean()
        e_y1 = data["Y1"].mean()
        naive = data.loc[data["A"] == 1, "Y"].mean() - data.loc[data["A"] == 0, "Y"].mean()

        return pd.DataFrame(
            {
                "estimand or contrast": [
                    r"$E[Y^0]$",
                    r"$E[Y^1]$",
                    r"ATE $=E[Y^1-Y^0]$",
                    r"ATT $=E[Y^1-Y^0\mid A=1]$",
                    r"ATC $=E[Y^1-Y^0\mid A=0]$",
                    r"Observed contrast $E[Y\mid A=1]-E[Y\mid A=0]$",
                ],
                "value": [e_y0, e_y1, ate, att, atc, naive],
            }
        )


    def continuous_cate_bins(data, n_bins=20):
        x = data["X"].to_numpy()
        tau = data["tau"].to_numpy()
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges[0] = edges[0] - 1e-8
        edges[-1] = edges[-1] + 1e-8
        bin_id = np.digitize(x, edges) - 1
        rows = []
        for j in range(n_bins):
            idx = bin_id == j
            if idx.sum() > 0:
                rows.append(
                    {
                        "x_mean": x[idx].mean(),
                        "cate": tau[idx].mean(),
                        "n": idx.sum(),
                    }
                )
        return pd.DataFrame(rows)


    return continuous_cate_bins, estimand_summary, generate_population, smooth_density


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive parameters

        A useful classroom sequence is:

        1. set confounding strength to zero and compare $P(Y\mid A=a)$ with $P(Y^a)$;
        2. increase confounding strength and watch the observed conditional curves move away from the causal curves;
        3. increase treatment-effect heterogeneity in $X$ or $G$ and watch ATE and ATT separate.
        """
    )
    return


@app.cell
def _(mo):
    controls = mo.ui.dictionary(
        {
            "n": mo.ui.slider(start=2_000, stop=80_000, step=2_000, value=30_000, label="sample size"),
            "confounding": mo.ui.slider(start=0.0, stop=3.0, step=0.05, value=1.25, label="confounding strength"),
            "baseline_x": mo.ui.slider(start=0.0, stop=2.0, step=0.05, value=1.0, label="effect of X on baseline outcome"),
            "tau_x": mo.ui.slider(start=-1.0, stop=1.0, step=0.05, value=0.40, label="heterogeneity: tau_X"),
            "tau_g": mo.ui.slider(start=-1.0, stop=1.5, step=0.05, value=0.60, label="heterogeneity: tau_G"),
            "tau0": mo.ui.slider(start=-1.0, stop=3.0, step=0.05, value=1.00, label="baseline treatment effect tau_0"),
            "treatment_intercept": mo.ui.slider(start=-2.0, stop=2.0, step=0.05, value=-0.15, label="treatment intercept"),
            "noise_sd": mo.ui.slider(start=0.25, stop=3.0, step=0.05, value=1.0, label="outcome noise SD"),
            "seed": mo.ui.number(start=1, stop=999_999, value=2026, label="seed"),
        },
        label="DGP controls",
    )
    controls.vstack()
    return (controls,)


@app.cell
def _(controls, generate_population):
    values = controls.value
    data = generate_population(
        n=int(values["n"]),
        seed=int(values["seed"]),
        confounding_strength=float(values["confounding"]),
        baseline_x_strength=float(values["baseline_x"]),
        tau_x=float(values["tau_x"]),
        tau_g=float(values["tau_g"]),
        tau0=float(values["tau0"]),
        treatment_intercept=float(values["treatment_intercept"]),
        noise_sd=float(values["noise_sd"]),
    )
    data.head()
    return data, values


@app.cell
def _(data, pd):
    treatment_summary = pd.DataFrame(
        {
            "quantity": [
                "Pr(A=1)",
                "mean X",
                "mean X among treated",
                "mean X among controls",
                "Pr(G=1)",
                "Pr(G=1 among treated)",
                "Pr(G=1 among controls)",
            ],
            "value": [
                data["A"].mean(),
                data["X"].mean(),
                data.loc[data["A"] == 1, "X"].mean(),
                data.loc[data["A"] == 0, "X"].mean(),
                data["G"].mean(),
                data.loc[data["A"] == 1, "G"].mean(),
                data.loc[data["A"] == 0, "G"].mean(),
            ],
        }
    )
    treatment_summary.round(3)
    return (treatment_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Treatment assignment and confounding

        When confounding strength is positive, the treated and untreated groups have different distributions of
        the pre-treatment causes $X$ and $G$. This is why the passive comparison $P(Y\mid A=1)$ vs.
        $P(Y\mid A=0)$ need not equal the causal comparison $P(Y^1)$ vs. $P(Y^0)$.
        """
    )
    return


@app.cell
def _(data, np, plt):
    x_grid_assign = np.linspace(data["X"].quantile(0.01), data["X"].quantile(0.99), 80)
    fig_assign, ax_assign = plt.subplots(figsize=(8, 4.6))
    for g_value in [0, 1]:
        tmp = data[data["G"] == g_value].copy()
        bins = np.quantile(tmp["X"], np.linspace(0, 1, 16))
        bins[0] -= 1e-8
        bins[-1] += 1e-8
        tmp["x_bin"] = np.digitize(tmp["X"], bins) - 1
        grouped = tmp.groupby("x_bin", observed=True).agg(x_mean=("X", "mean"), a_mean=("A", "mean"))
        ax_assign.plot(grouped["x_mean"], grouped["a_mean"], marker="o", label=f"G={g_value}")

    ax_assign.set_ylim(-0.02, 1.02)
    ax_assign.set_xlabel("continuous pre-treatment covariate X")
    ax_assign.set_ylabel("empirical Pr(A=1 | X bin, G)")
    ax_assign.set_title("Treatment selection as a function of pre-treatment variables")
    ax_assign.legend(title="discrete group")
    fig_assign.tight_layout()
    fig_assign

    return (fig_assign,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Marginal and conditional observed distributions

        The marginal distribution $P(Y)$ mixes treated and untreated outcomes under the natural treatment mechanism.
        The conditional distributions $P(Y\mid A=1)$ and $P(Y\mid A=0)$ are distributions in selected subpopulations,
        not intervention distributions.
        """
    )
    return


@app.cell
def _(data, np, plt, smooth_density):
    y_pool = data["Y"].to_numpy()
    y_left, y_right = np.quantile(y_pool, [0.005, 0.995])
    y_grid = np.linspace(y_left, y_right, 360)

    fig_obs, ax_obs = plt.subplots(figsize=(8, 4.8))
    ax_obs.plot(y_grid, smooth_density(data["Y"], y_grid), linewidth=2.4, label="marginal P(Y)")
    ax_obs.plot(y_grid, smooth_density(data.loc[data["A"] == 0, "Y"], y_grid), label="conditional P(Y | A=0)")
    ax_obs.plot(y_grid, smooth_density(data.loc[data["A"] == 1, "Y"], y_grid), label="conditional P(Y | A=1)")
    ax_obs.set_xlabel("outcome value y")
    ax_obs.set_ylabel("density")
    ax_obs.set_title("Observed distributions: marginal vs conditional")
    ax_obs.legend()
    fig_obs.tight_layout()
    fig_obs

    return (fig_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Post-intervention distributions

        In this simulation the potential outcomes are known, so we can directly draw the post-intervention
        distributions

        $$
        P(Y\mid do(A=0)) = P(Y^0),
        \qquad
        P(Y\mid do(A=1)) = P(Y^1).
        $$

        Compare these curves with the passive conditional curves $P(Y\mid A=0)$ and $P(Y\mid A=1)$.
        With confounding, conditioning on $A=a$ selects a non-representative subpopulation; intervening with
        $do(A=a)$ applies the treatment level to the whole population.
        """
    )
    return


@app.cell
def _(data, np, plt, smooth_density):
    pooled = np.concatenate(
        [
            data["Y"].to_numpy(),
            data["Y0"].to_numpy(),
            data["Y1"].to_numpy(),
        ]
    )
    left, right = np.quantile(pooled, [0.005, 0.995])
    grid = np.linspace(left, right, 380)

    fig_do, ax_do = plt.subplots(figsize=(8.5, 5.0))
    ax_do.plot(grid, smooth_density(data["Y0"], grid), linewidth=2.4, label="interventional P(Y^0) = P(Y | do(A=0))")
    ax_do.plot(grid, smooth_density(data["Y1"], grid), linewidth=2.4, label="interventional P(Y^1) = P(Y | do(A=1))")
    ax_do.plot(grid, smooth_density(data.loc[data["A"] == 0, "Y"], grid), linestyle="--", label="observed P(Y | A=0)")
    ax_do.plot(grid, smooth_density(data.loc[data["A"] == 1, "Y"], grid), linestyle="--", label="observed P(Y | A=1)")
    ax_do.set_xlabel("outcome value y")
    ax_do.set_ylabel("density")
    ax_do.set_title("Observed conditional curves vs post-intervention curves")
    ax_do.legend(fontsize=9)
    fig_do.tight_layout()
    fig_do

    return (fig_do,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Expected potential outcomes, ATE, ATT, and the naive observed contrast

        The estimands are computed using the simulated potential outcomes:

        $$
        E[Y^a] = E[Y\mid do(A=a)],
        \qquad
        ATE = E[Y^1-Y^0],
        \qquad
        ATT = E[Y^1-Y^0\mid A=1].
        $$

        The last row below is not a causal estimand in general. It is the purely observational contrast
        $E[Y\mid A=1]-E[Y\mid A=0]$.
        """
    )
    return


@app.cell
def _(data, estimand_summary):
    estimands = estimand_summary(data)
    estimands.round(3)
    return (estimands,)


@app.cell
def _(estimands, plt):
    contrast_rows = estimands.iloc[[2, 3, 4, 5]].copy()
    labels = ["ATE", "ATT", "ATC", "observed\ncontrast"]
    fig_estimands, ax_estimands = plt.subplots(figsize=(8, 4.8))
    ax_estimands.axhline(0.0, linewidth=1)
    ax_estimands.bar(labels, contrast_rows["value"])
    ax_estimands.set_ylabel("effect or contrast")
    ax_estimands.set_title("Causal effects vs the naive observed contrast")
    fig_estimands.tight_layout()
    fig_estimands

    return (fig_estimands,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. CATE as a function of a continuous pre-treatment variable

        The continuous CATE shown here is

        $$
        CATE(x)=E[Y^1-Y^0\mid X=x].
        $$

        Since $X$ is pre-treatment, this is a causal subgroup estimand. In the simulation, the CATE changes
        with $x$ when the slider $\tau_X$ is not zero.
        """
    )
    return


@app.cell
def _(continuous_cate_bins, data, np, plt, values):
    cate_bins = continuous_cate_bins(data, n_bins=22)
    x_grid = np.linspace(data["X"].quantile(0.01), data["X"].quantile(0.99), 200)
    # Because G is independent of X in this DGP, E[G | X=x] is approximately E[G].
    cate_curve = float(values["tau0"]) + float(values["tau_x"]) * x_grid + float(values["tau_g"]) * data["G"].mean()

    fig_cate_x, ax_cate_x = plt.subplots(figsize=(8, 4.8))
    ax_cate_x.plot(x_grid, cate_curve, linewidth=2.3, label="population CATE(x) curve")
    ax_cate_x.scatter(cate_bins["x_mean"], cate_bins["cate"], label="binned finite-sample CATE")
    ax_cate_x.axhline(data["tau"].mean(), linewidth=1, linestyle="--", label="ATE")
    ax_cate_x.set_xlabel("continuous pre-treatment covariate X")
    ax_cate_x.set_ylabel("E[Y^1 - Y^0 | X=x]")
    ax_cate_x.set_title("CATE as a function of a continuous variable")
    ax_cate_x.legend()
    fig_cate_x.tight_layout()
    fig_cate_x

    return cate_bins, fig_cate_x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. CATE as a function of a discrete pre-treatment variable

        The discrete CATE is

        $$
        CATE(g)=E[Y^1-Y^0\mid G=g].
        $$

        The two bars below are causal contrasts within two pre-treatment subpopulations.
        """
    )
    return


@app.cell
def _(data, pd, plt):
    cate_g = (
        data.groupby("G", observed=True)
        .agg(cate=("tau", "mean"), n=("tau", "size"), mean_x=("X", "mean"))
        .reset_index()
    )

    fig_cate_g, ax_cate_g = plt.subplots(figsize=(7, 4.6))
    ax_cate_g.bar([f"G={int(g)}" for g in cate_g["G"]], cate_g["cate"])
    ax_cate_g.axhline(data["tau"].mean(), linewidth=1, linestyle="--", label="ATE")
    ax_cate_g.set_ylabel("E[Y^1 - Y^0 | G=g]")
    ax_cate_g.set_title("CATE as a function of a discrete variable")
    ax_cate_g.legend()
    fig_cate_g.tight_layout()
    fig_cate_g

    return cate_g, fig_cate_g


@app.cell
def _(cate_g):
    cate_g.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Takeaway

        - $P(Y\mid A=a)$ is an observed conditional distribution. It describes the people who naturally received treatment level $a$.
        - $P(Y\mid do(A=a))=P(Y^a)$ is a post-intervention distribution. It describes what would happen if everyone were set to treatment level $a$.
        - $ATE=E[Y^1-Y^0]$ averages over the whole population.
        - $CATE(x)$ and $CATE(g)$ are treatment-effect functions over pre-treatment subpopulations.
        - $ATT=E[Y^1-Y^0\mid A=1]$ averages over the naturally treated population, so it can differ from the ATE when treatment effects are heterogeneous and treatment assignment is selective.
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
        marimo edit causal_estimands_visual_simulation_marimo.py
        ```

        To serve it as an app instead of opening the notebook editor:

        ```bash
        marimo run causal_estimands_visual_simulation_marimo.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
