# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "scikit-learn",
#   "torch"
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        import torch
        import torch.nn as nn
        TORCH_AVAILABLE = True
    except Exception:
        torch = None
        nn = None
        TORCH_AVAILABLE = False
    return (
        LinearRegression,
        MLPRegressor,
        StandardScaler,
        TORCH_AVAILABLE,
        mo,
        nn,
        np,
        pd,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TARNet as a neural T-learner with shared representation

    **Goal.** Estimate the CATE

    \[
    \tau(w)=Q(w,1)-Q(w,0)
    \]

    using a TARNet-style architecture: a shared representation of \(W\), followed by two treatment-specific outcome heads.

    The code is hidden by default. Use the controls to answer conceptual questions: What should enter the representation? What happens if we omit a confounder? What happens if we include a post-treatment variable? Does a single S-learner recover heterogeneity as well as a two-head architecture?
    """)
    return


@app.cell(hide_code=True)
def _(TORCH_AVAILABLE, mo):
    lab2_n = mo.ui.slider(start=600, stop=3000, step=200, value=1200, label="Sample size")
    lab2_epochs = mo.ui.slider(start=20, stop=160, step=20, value=80, label="Training epochs")
    lab2_overlap = mo.ui.dropdown(
        options=["reasonable overlap", "weak overlap", "almost randomized"],
        value="reasonable overlap",
        label="Treatment assignment"
    )
    lab2_model_choice = mo.ui.dropdown(
        options=[
            "TARNet: shared representation + two heads",
            "T-learner: two separate neural nets",
            "wrong: linear S-learner, no effect-modification terms",
            "wrong: TARNet omits confounder W3",
            "wrong: TARNet includes post-treatment Z"
        ],
        value="TARNet: shared representation + two heads",
        label="Learner"
    )
    lab2_seed = mo.ui.number(value=21, start=1, stop=9999, label="Random seed")
    lab2_run_button = mo.ui.run_button(label="Train learner")
    torch_note = "PyTorch is available." if TORCH_AVAILABLE else "PyTorch was not found, so the lab will fall back to scikit-learn MLPs."
    mo.vstack([
        mo.md(f"""
        ## Choose a neural CATE learner

        {torch_note}

        The default is the architecture discussed in the lecture: shared representation plus two heads. The wrong options are intentionally runnable, but conceptually questionable.
        """),
        mo.hstack([lab2_n, lab2_epochs]),
        mo.hstack([lab2_overlap, lab2_model_choice]),
        mo.hstack([lab2_seed, lab2_run_button]),
    ])
    return (
        lab2_epochs,
        lab2_model_choice,
        lab2_n,
        lab2_overlap,
        lab2_run_button,
        lab2_seed,
    )


@app.cell(hide_code=True)
def _(
    LinearRegression,
    MLPRegressor,
    StandardScaler,
    TORCH_AVAILABLE,
    nn,
    np,
    pd,
    torch,
):
    def lab2_expit(x):
        return 1.0 / (1.0 + np.exp(-x))

    def lab2_generate_data(n, seed=1, overlap="reasonable overlap"):
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(n, 5))
        W1, W2, W3, W4, W5 = W.T
        logits = -0.1 + 1.05 * W1 - 0.9 * W2 + 0.75 * W3 + 0.75 * W1 * W2 + 0.5 * np.sin(W5)
        if overlap == "weak overlap":
            logits = 1.75 * logits
        elif overlap == "almost randomized":
            logits = 0.10 * logits
        e = np.clip(lab2_expit(logits), 0.03, 0.97)
        A = rng.binomial(1, e)
        Z = 0.8 * A + 0.35 * W1 - 0.20 * W2 + rng.normal(scale=0.8, size=n)
        mu0 = 0.5 + 1.8 * np.sin(W1) + W2**2 - 1.1 * W3 + 0.6 * W4 * W5
        tau = 1.0 + 1.2 * W1 - 0.8 * W2 + 0.8 * (W3 > 0) + 0.6 * np.sin(W4)
        Y = mu0 + A * tau + 0.6 * Z + rng.normal(scale=1.0, size=n)
        df = pd.DataFrame(W, columns=[f"W{j}" for j in range(1, 6)])
        df["A"] = A
        df["Z"] = Z
        df["Y"] = Y
        df["tau_true"] = tau + 0.6 * 0.8
        return df

    def lab2_feature_matrix(df, choice):
        cols = ["W1", "W2", "W3", "W4", "W5"]
        if choice == "wrong: TARNet omits confounder W3":
            cols = ["W1", "W2", "W4", "W5"]
        if choice == "wrong: TARNet includes post-treatment Z":
            cols = ["W1", "W2", "W3", "W4", "W5", "Z"]
        return df[cols].to_numpy()

    if TORCH_AVAILABLE:
        class Lab2TARNet(nn.Module):
            def __init__(self, p, hidden=32):
                super().__init__()
                self.rep = nn.Sequential(nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
                self.head0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                self.head1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            def forward(self, x):
                r = self.rep(x)
                return self.head0(r).squeeze(-1), self.head1(r).squeeze(-1)

        class Lab2MLP(nn.Module):
            def __init__(self, p, hidden=32):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            def forward(self, x):
                return self.net(x).squeeze(-1)

    def lab2_train_tarnet_torch(X, A, Y, epochs=80, seed=1):
        torch.manual_seed(seed)
        sx, sy = StandardScaler(), StandardScaler()
        Xs = sx.fit_transform(X)
        Ys = sy.fit_transform(Y.reshape(-1, 1)).ravel()
        tx = torch.tensor(Xs, dtype=torch.float32)
        ta = torch.tensor(A.astype(float), dtype=torch.float32)
        ty = torch.tensor(Ys, dtype=torch.float32)
        model = Lab2TARNet(X.shape[1], hidden=36)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(int(epochs)):
            opt.zero_grad()
            y0, y1 = model(tx)
            pred = ta * y1 + (1.0 - ta) * y0
            loss = torch.mean((ty - pred) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            y0s, y1s = model(tx)
        q0 = sy.inverse_transform(y0s.numpy().reshape(-1, 1)).ravel()
        q1 = sy.inverse_transform(y1s.numpy().reshape(-1, 1)).ravel()
        return q1, q0

    def lab2_train_tlearner_torch(X, A, Y, epochs=80, seed=1):
        torch.manual_seed(seed)
        sx, sy = StandardScaler(), StandardScaler()
        Xs = sx.fit_transform(X)
        Ys = sy.fit_transform(Y.reshape(-1, 1)).ravel()
        qhat = []
        for a in [0, 1]:
            ix = A == a
            tx = torch.tensor(Xs[ix], dtype=torch.float32)
            ty = torch.tensor(Ys[ix], dtype=torch.float32)
            model = Lab2MLP(X.shape[1], hidden=36)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(int(epochs)):
                opt.zero_grad()
                pred = model(tx)
                loss = torch.mean((ty - pred) ** 2)
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred_all = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
            qhat.append(sy.inverse_transform(pred_all.reshape(-1, 1)).ravel())
        return qhat[1], qhat[0]

    def lab2_fit_predict(df, choice, epochs=80, seed=1):
        X = lab2_feature_matrix(df, choice)
        A = df["A"].to_numpy().astype(int)
        Y = df["Y"].to_numpy()
        if choice == "wrong: linear S-learner, no effect-modification terms":
            Xs = np.column_stack([A, X])
            model = LinearRegression().fit(Xs, Y)
            q1 = model.predict(np.column_stack([np.ones(len(df)), X]))
            q0 = model.predict(np.column_stack([np.zeros(len(df)), X]))
            return q1, q0
        if TORCH_AVAILABLE:
            if choice == "T-learner: two separate neural nets":
                return lab2_train_tlearner_torch(X, A, Y, epochs=epochs, seed=seed)
            return lab2_train_tarnet_torch(X, A, Y, epochs=epochs, seed=seed)
        # Fallback if PyTorch is unavailable
        if choice == "T-learner: two separate neural nets":
            sx = StandardScaler().fit(X)
            Xs = sx.transform(X)
            models = []
            for a in [0, 1]:
                model = MLPRegressor(hidden_layer_sizes=(36, 36), max_iter=int(epochs), random_state=seed + a, learning_rate_init=0.01)
                model.fit(Xs[A == a], Y[A == a])
                models.append(model)
            return models[1].predict(Xs), models[0].predict(Xs)
        Xs = np.column_stack([A, X])
        model = MLPRegressor(hidden_layer_sizes=(36, 36), max_iter=int(epochs), random_state=seed, learning_rate_init=0.01)
        model.fit(Xs, Y)
        q1 = model.predict(np.column_stack([np.ones(len(df)), X]))
        q0 = model.predict(np.column_stack([np.zeros(len(df)), X]))
        return q1, q0

    def lab2_metrics(df, q1, q0):
        tau_hat = q1 - q0
        tau_true = df["tau_true"].to_numpy()
        ate_hat = tau_hat.mean()
        ate_true = tau_true.mean()
        return pd.DataFrame({
            "quantity": ["ATE", "CATE RMSE", "CATE correlation"],
            "value": [ate_hat, np.sqrt(np.mean((tau_hat - tau_true) ** 2)), np.corrcoef(tau_hat, tau_true)[0, 1]],
            "target/reference": [ate_true, 0.0, 1.0]
        })

    def lab2_note(choice):
        if choice == "TARNet: shared representation + two heads":
            return "This is the intended TARNet logic: learn a representation of pre-treatment W and use two potential-outcome heads."
        if choice == "T-learner: two separate neural nets":
            return "This is close to a neural T-learner. It can work well, but each treatment arm is learned separately and may be unstable under weak overlap."
        if choice == "wrong: linear S-learner, no effect-modification terms":
            return "This learner can estimate an average treatment effect, but it has almost no ability to learn heterogeneous effects because treatment interactions are excluded."
        if choice == "wrong: TARNet omits confounder W3":
            return "This runs, but W3 is a confounder and an effect modifier. Omitting it can bias both ATE and CATE estimates."
        return "This runs, but Z is post-treatment. Keeping observed Z fixed while changing A changes the causal question and can block part of the effect."

    return lab2_fit_predict, lab2_generate_data, lab2_metrics, lab2_note


@app.cell(hide_code=True)
def _(
    lab2_epochs,
    lab2_fit_predict,
    lab2_generate_data,
    lab2_metrics,
    lab2_model_choice,
    lab2_n,
    lab2_overlap,
    lab2_run_button,
    lab2_seed,
    mo,
):
    mo.stop(not lab2_run_button.value, mo.md("Press **Train learner** after selecting a setup."))
    lab2_df = lab2_generate_data(int(lab2_n.value), seed=int(lab2_seed.value), overlap=lab2_overlap.value)
    lab2_q1, lab2_q0 = lab2_fit_predict(lab2_df, lab2_model_choice.value, epochs=int(lab2_epochs.value), seed=int(lab2_seed.value))
    lab2_tau_hat = lab2_q1 - lab2_q0
    lab2_results = lab2_metrics(lab2_df, lab2_q1, lab2_q0)
    return lab2_df, lab2_results, lab2_tau_hat


@app.cell(hide_code=True)
def _(
    lab2_df,
    lab2_model_choice,
    lab2_note,
    lab2_results,
    lab2_tau_hat,
    mo,
    pd,
    plt,
):
    lab2_tmp = pd.DataFrame({
        "W1": lab2_df["W1"],
        "true_tau": lab2_df["tau_true"],
        "estimated_tau": lab2_tau_hat,
    })
    lab2_tmp["bin"] = pd.qcut(lab2_tmp["W1"], q=12, duplicates="drop")
    lab2_curve = lab2_tmp.groupby("bin", observed=True).agg(W1=("W1", "mean"), true_tau=("true_tau", "mean"), estimated_tau=("estimated_tau", "mean")).reset_index(drop=True)
    lab2_fig, lab2_ax = plt.subplots(figsize=(7, 4))
    lab2_ax.plot(lab2_curve["W1"], lab2_curve["true_tau"], marker="o", label="True τ(w)")
    lab2_ax.plot(lab2_curve["W1"], lab2_curve["estimated_tau"], marker="o", label="Estimated τ(w)")
    lab2_ax.set_xlabel("Binned W1")
    lab2_ax.set_ylabel("CATE")
    lab2_ax.set_title("Does the learner recover heterogeneity over W1?")
    lab2_ax.legend()
    lab2_fig.tight_layout()
    mo.vstack([
        mo.md("## Results"),
        lab2_results.round(3),
        lab2_fig,
        mo.md("## Diagnostic"),
        mo.md(lab2_note(lab2_model_choice.value)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Student discussion prompts

    1. Compare TARNet and the neural T-learner under weak overlap. Which one becomes unstable first?
    2. Why is a two-head architecture closer to a T-learner than to an S-learner?
    3. What does the linear S-learner miss about \(\tau(w)\)?
    4. Why is including post-treatment \(Z\) tempting, and why is it wrong for estimating the total effect?
    """)
    return


if __name__ == "__main__":
    app.run()
