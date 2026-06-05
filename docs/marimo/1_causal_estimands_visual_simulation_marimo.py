import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


@app.cell
def _():
    # These are the only packages used in the notebook.
    # marimo builds the app, numpy simulates arrays, pandas stores tables,
    # and matplotlib makes the plots.
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


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


@app.cell(hide_code=True)
def _(np, pd):
    def expit(z):
        """Logistic function: turns any real number into a probability."""
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
        """Simulate observed data and the two potential outcomes for each unit."""

        # The random number generator makes the simulation reproducible.
        rng = np.random.default_rng(int(seed))
        n = int(n)

        # 1. Pre-treatment covariates.
        # X is continuous and G is binary. Both are generated before treatment.
        X = rng.normal(loc=0.0, scale=1.0, size=n)
        G = rng.binomial(n=1, p=0.45, size=n)

        # 2. Treatment assignment.
        # The treatment probability depends on X and G. When the slider
        # confounding_strength is zero, treatment is almost random with respect
        # to X and G. When it is positive, treated and untreated people differ.
        selection_score = 0.90 * X + 0.85 * (G - 0.45)
        logit_pi = treatment_intercept + confounding_strength * selection_score
        pi = expit(logit_pi)

        # Keep probabilities away from exactly 0 and 1. This is not essential
        # for the plots, but it avoids tiny treatment groups at extreme settings.
        pi = np.clip(pi, 0.02, 0.98)
        A = rng.binomial(n=1, p=pi)

        # 3. Baseline potential outcome Y^0.
        # This is the outcome each unit would have under A=0.
        linear_x_part = 0.90 * X
        nonlinear_x_part = 0.30 * (X**2 - 1.0)
        group_part = 0.70 * G
        smooth_x_part = 0.25 * np.sin(1.5 * X)

        mu0 = (
            0.20
            + baseline_x_strength * (linear_x_part + nonlinear_x_part)
            + group_part
            + smooth_x_part
        )
        epsilon = rng.normal(loc=0.0, scale=noise_sd, size=n)
        Y0 = mu0 + epsilon

        # 4. Individual treatment effect tau(X,G).
        # If tau_x or tau_g is nonzero, treatment effects are heterogeneous.
        tau = tau0 + tau_x * X + tau_g * G

        # 5. Treated potential outcome Y^1 and observed outcome Y.
        # We observe Y^1 only if A=1 and Y^0 only if A=0.
        Y1 = Y0 + tau
        Y = A * Y1 + (1 - A) * Y0

        # Store everything in one table. In real data, Y0 and Y1 are not both
        # observed, but here we keep them because this is a teaching simulation.
        return pd.DataFrame(
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


    def smooth_density(values, grid, bins=170):
        """Approximate a smooth density curve using only numpy.

        The idea is simple:
        1. make a histogram of the values;
        2. smooth the histogram with a Gaussian-shaped kernel;
        3. interpolate the smoothed histogram on the requested plotting grid.
        """

        values = np.asarray(values)
        values = values[np.isfinite(values)]

        if len(values) < 5:
            return np.zeros_like(grid)

        grid_left = float(np.min(grid))
        grid_right = float(np.max(grid))

        counts, edges = np.histogram(
            values,
            bins=bins,
            range=(grid_left, grid_right),
            density=False,
        )
        bin_width = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Silverman's rule gives a reasonable automatic bandwidth.
        sample_sd = np.std(values)
        bandwidth = 1.06 * sample_sd * (len(values) ** (-1 / 5))
        bandwidth = max(bandwidth, 1.5 * bin_width)

        # Convert the bandwidth from outcome units into histogram-bin units.
        sigma_bins = max(bandwidth / bin_width, 1.0)
        radius = int(np.ceil(4.0 * sigma_bins))

        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()

        smoothed_counts = np.convolve(counts, kernel, mode="same")
        density = smoothed_counts / (len(values) * bin_width)

        return np.interp(grid, centers, density, left=0.0, right=0.0)


    def estimand_summary(data):
        """Compute the main causal estimands and one non-causal contrast."""

        _treated_mask = data["A"] == 1
        _control_mask = data["A"] == 0

        e_y0 = data["Y0"].mean()
        e_y1 = data["Y1"].mean()
        ate = data["tau"].mean()
        att = data.loc[_treated_mask, "tau"].mean()
        atc = data.loc[_control_mask, "tau"].mean()

        # This contrast is generally biased for the ATE in observational data.
        naive = data.loc[_treated_mask, "Y"].mean() - data.loc[_control_mask, "Y"].mean()

        rows = [
            (r"$E[Y^0]$", e_y0),
            (r"$E[Y^1]$", e_y1),
            (r"ATE $=E[Y^1-Y^0]$", ate),
            (r"ATT $=E[Y^1-Y^0\mid A=1]$", att),
            (r"ATC $=E[Y^1-Y^0\mid A=0]$", atc),
            (r"Observed contrast $E[Y\mid A=1]-E[Y\mid A=0]$", naive),
        ]

        return pd.DataFrame(rows, columns=["estimand or contrast", "value"])


    def continuous_cate_bins(data, n_bins=20):
        """Estimate CATE(X) by averaging tau inside quantile bins of X."""

        x = data["X"].to_numpy()
        tau = data["tau"].to_numpy()

        # Quantile bins keep roughly the same number of observations per bin.
        bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        bin_edges[0] = bin_edges[0] - 1e-8
        bin_edges[-1] = bin_edges[-1] + 1e-8
        bin_id = np.digitize(x, bin_edges) - 1

        rows = []
        for this_bin in range(n_bins):
            in_bin = bin_id == this_bin

            if in_bin.sum() == 0:
                continue

            rows.append(
                {
                    "x_mean": x[in_bin].mean(),
                    "cate": tau[in_bin].mean(),
                    "n": in_bin.sum(),
                }
            )

        return pd.DataFrame(rows)


    return (
        continuous_cate_bins,
        estimand_summary,
        expit,
        generate_population,
        smooth_density,
    )


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


@app.cell(hide_code=True)
def _(mo):
    # Each slider controls one part of the data-generating process.
    controls = mo.ui.dictionary(
        {
            "n": mo.ui.slider(
                start=2_000,
                stop=80_000,
                step=2_000,
                value=30_000,
                label="sample size",
            ),
            "confounding": mo.ui.slider(
                start=0.0,
                stop=3.0,
                step=0.05,
                value=1.25,
                label="confounding strength",
            ),
            "baseline_x": mo.ui.slider(
                start=0.0,
                stop=2.0,
                step=0.05,
                value=1.0,
                label="effect of X on baseline outcome",
            ),
            "tau_x": mo.ui.slider(
                start=-1.0,
                stop=1.0,
                step=0.05,
                value=0.40,
                label="heterogeneity: tau_X",
            ),
            "tau_g": mo.ui.slider(
                start=-1.0,
                stop=1.5,
                step=0.05,
                value=0.60,
                label="heterogeneity: tau_G",
            ),
            "tau0": mo.ui.slider(
                start=-1.0,
                stop=3.0,
                step=0.05,
                value=1.00,
                label="baseline treatment effect tau_0",
            ),
            "treatment_intercept": mo.ui.slider(
                start=-2.0,
                stop=2.0,
                step=0.05,
                value=-0.15,
                label="treatment intercept",
            ),
            "noise_sd": mo.ui.slider(
                start=0.25,
                stop=3.0,
                step=0.05,
                value=1.0,
                label="outcome noise SD",
            ),
            "seed": mo.ui.number(
                start=1,
                stop=999_999,
                value=2026,
                label="seed",
            ),
        },
        label="DGP controls",
    )

    # Show the controls vertically in the app.
    controls.vstack()
    return (controls,)


@app.cell(hide_code=True)
def _(controls, generate_population):
    # Read the current slider values.
    values = controls.value

    # Generate a new data set every time one of the controls changes.
    data = generate_population(
        n=values["n"],
        seed=values["seed"],
        confounding_strength=values["confounding"],
        baseline_x_strength=values["baseline_x"],
        tau_x=values["tau_x"],
        tau_g=values["tau_g"],
        tau0=values["tau0"],
        treatment_intercept=values["treatment_intercept"],
        noise_sd=values["noise_sd"],
    )

    # Display the first rows so students can see the simulated variables.
    data.head()
    return data, values


@app.cell(hide_code=True)
def _(data, pd):
    # Masks make the summaries below easier to read.
    _treated_mask = data["A"] == 1
    _control_mask = data["A"] == 0

    rows = [
        ("Pr(A=1)", data["A"].mean()),
        ("mean X", data["X"].mean()),
        ("mean X among treated", data.loc[_treated_mask, "X"].mean()),
        ("mean X among controls", data.loc[_control_mask, "X"].mean()),
        ("Pr(G=1)", data["G"].mean()),
        ("Pr(G=1 among treated)", data.loc[_treated_mask, "G"].mean()),
        ("Pr(G=1 among controls)", data.loc[_control_mask, "G"].mean()),
    ]

    treatment_summary = pd.DataFrame(rows, columns=["quantity", "value"])
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


@app.cell(hide_code=True)
def _(data, np, plt):
    fig_assign, ax_assign = plt.subplots(figsize=(8, 4.6))

    for g_value in [0, 1]:
        # Look only at one level of the discrete covariate G.
        group_data = data[data["G"] == g_value].copy()

        # Divide X into quantile bins, then estimate Pr(A=1) in each bin.
        bin_edges = np.quantile(group_data["X"], np.linspace(0, 1, 16))
        bin_edges[0] = bin_edges[0] - 1e-8
        bin_edges[-1] = bin_edges[-1] + 1e-8
        group_data["x_bin"] = np.digitize(group_data["X"], bin_edges) - 1

        binned = group_data.groupby("x_bin", observed=True).agg(
            x_mean=("X", "mean"),
            a_mean=("A", "mean"),
        )

        ax_assign.plot(
            binned["x_mean"],
            binned["a_mean"],
            marker="o",
            label=f"G={g_value}",
        )

    ax_assign.set_ylim(-0.02, 1.02)
    ax_assign.set_xlabel("continuous pre-treatment covariate X")
    ax_assign.set_ylabel("empirical Pr(A=1 | X bin, G)")
    ax_assign.set_title("Treatment selection as a function of pre-treatment variables")
    ax_assign.legend(title="discrete group")
    fig_assign.tight_layout()
    fig_assign

    return ax_assign, fig_assign


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


@app.cell(hide_code=True)
def _(data, np, plt, smooth_density):
    # Use the central 99% of observed outcomes as the plotting range.
    y_left, y_right = np.quantile(data["Y"], [0.005, 0.995])
    y_grid = np.linspace(y_left, y_right, 360)

    _treated_mask = data["A"] == 1
    _control_mask = data["A"] == 0

    density_y = smooth_density(data["Y"], y_grid)
    _density_y_a0_obs = smooth_density(data.loc[_control_mask, "Y"], y_grid)
    _density_y_a1_obs = smooth_density(data.loc[_treated_mask, "Y"], y_grid)

    fig_obs, ax_obs = plt.subplots(figsize=(8, 4.8))
    ax_obs.plot(y_grid, density_y, linewidth=2.4, label="marginal P(Y)")
    ax_obs.plot(y_grid, _density_y_a0_obs, label="conditional P(Y | A=0)")
    ax_obs.plot(y_grid, _density_y_a1_obs, label="conditional P(Y | A=1)")
    ax_obs.set_xlabel("outcome value y")
    ax_obs.set_ylabel("density")
    ax_obs.set_title("Observed distributions: marginal vs conditional")
    ax_obs.legend()
    fig_obs.tight_layout()
    fig_obs

    return ax_obs, fig_obs, y_grid


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


@app.cell(hide_code=True)
def _(data, np, plt, smooth_density):
    # To compare all curves on the same x-axis, build the grid from Y, Y0, and Y1.
    all_outcomes = np.concatenate(
        [
            data["Y"].to_numpy(),
            data["Y0"].to_numpy(),
            data["Y1"].to_numpy(),
        ]
    )
    left, right = np.quantile(all_outcomes, [0.005, 0.995])
    grid = np.linspace(left, right, 380)

    _treated_mask = data["A"] == 1
    _control_mask = data["A"] == 0

    density_y0 = smooth_density(data["Y0"], grid)
    density_y1 = smooth_density(data["Y1"], grid)
    _density_y_a0_do = smooth_density(data.loc[_control_mask, "Y"], grid)
    _density_y_a1_do = smooth_density(data.loc[_treated_mask, "Y"], grid)

    fig_do, ax_do = plt.subplots(figsize=(8.5, 5.0))
    ax_do.plot(
        grid,
        density_y0,
        linewidth=2.4,
        label="interventional P(Y^0) = P(Y | do(A=0))",
    )
    ax_do.plot(
        grid,
        density_y1,
        linewidth=2.4,
        label="interventional P(Y^1) = P(Y | do(A=1))",
    )
    ax_do.plot(
        grid,
        _density_y_a0_do,
        linestyle="--",
        label="observed P(Y | A=0)",
    )
    ax_do.plot(
        grid,
        _density_y_a1_do,
        linestyle="--",
        label="observed P(Y | A=1)",
    )
    ax_do.set_xlabel("outcome value y")
    ax_do.set_ylabel("density")
    ax_do.set_title("Observed conditional curves vs post-intervention curves")
    ax_do.legend(fontsize=9)
    fig_do.tight_layout()
    fig_do

    return ax_do, fig_do, grid


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


@app.cell(hide_code=True)
def _(data, estimand_summary):
    estimands = estimand_summary(data)
    estimands.round(3)
    return (estimands,)


@app.cell(hide_code=True)
def _(estimands, plt):
    # Rows 2--5 contain ATE, ATT, ATC, and the observed contrast.
    contrast_rows = estimands.iloc[[2, 3, 4, 5]].copy()
    labels = ["ATE", "ATT", "ATC", "observed\ncontrast"]

    fig_estimands, ax_estimands = plt.subplots(figsize=(8, 4.8))
    ax_estimands.axhline(0.0, linewidth=1)
    ax_estimands.bar(labels, contrast_rows["value"])
    ax_estimands.set_ylabel("effect or contrast")
    ax_estimands.set_title("Causal effects vs the naive observed contrast")
    fig_estimands.tight_layout()
    fig_estimands

    return ax_estimands, contrast_rows, fig_estimands, labels


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


@app.cell(hide_code=True)
def _(continuous_cate_bins, data, np, plt, values):
    # Binned points: empirical averages of the individual treatment effect tau.
    cate_bins = continuous_cate_bins(data, n_bins=22)

    # Smooth line: the theoretical CATE curve implied by the DGP.
    # Because G is independent of X here, E[G | X=x] is approximately E[G].
    x_grid = np.linspace(data["X"].quantile(0.01), data["X"].quantile(0.99), 200)
    mean_g = data["G"].mean()
    cate_curve = values["tau0"] + values["tau_x"] * x_grid + values["tau_g"] * mean_g

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

    return ax_cate_x, cate_bins, cate_curve, fig_cate_x, x_grid


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


@app.cell(hide_code=True)
def _(data, plt):
    # Average the individual treatment effect separately for G=0 and G=1.
    cate_g = (
        data.groupby("G", observed=True)
        .agg(
            cate=("tau", "mean"),
            n=("tau", "size"),
            mean_x=("X", "mean"),
        )
        .reset_index()
    )

    bar_labels = [f"G={int(g)}" for g in cate_g["G"]]

    fig_cate_g, ax_cate_g = plt.subplots(figsize=(7, 4.6))
    ax_cate_g.bar(bar_labels, cate_g["cate"])
    ax_cate_g.axhline(data["tau"].mean(), linewidth=1, linestyle="--", label="ATE")
    ax_cate_g.set_ylabel("E[Y^1 - Y^0 | G=g]")
    ax_cate_g.set_title("CATE as a function of a discrete variable")
    ax_cate_g.legend()
    fig_cate_g.tight_layout()
    fig_cate_g

    return ax_cate_g, cate_g, fig_cate_g


@app.cell(hide_code=True)
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
