"""Heterogeneous stacking and forecast combinations, built ONLY from earlier out-of-sample forecasts.

Base-model forecasts F[i, b] for i >= WARMUP were each produced from y[:i] alone. At forecast origin t (a test
index) the meta-learner is fitted on rows i in [WARMUP, t-1] (targets y[i], already observed at time t) and applied to
F[t, :]. No row from t or later enters the fit: this is the leak-free replacement of the original procedure
(which fitted the meta-learner on the test actuals).

Output: results/forecasts/<market>_ens.csv (long format, one row per model/seed/origin)."""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import data
from config import MARKETS, MIN_STACK_TRAIN, N_TEST, RESULTS, SEEDS, WARMUP

BASES = ["ARIMA", "SVR", "LSTM", "DT"]
DETERMINISTIC = {"Naive", "SeasonalNaive", "ARIMA", "ETS", "SVR", "DT"}


def load_market(market):
    fr = [pd.read_csv(RESULTS / "forecasts" / f"{market}_{k}.csv", parse_dates=["date"]) for k in ("base", "deep")]
    return pd.concat(fr, ignore_index=True)


def get(df, model, seed):
    s = 0 if model in DETERMINISTIC else seed
    g = df[(df["model"] == model) & (df["seed"] == s)].sort_values("idx")
    return g["idx"].to_numpy(), g["forecast"].to_numpy(), g["actual"].to_numpy(), g["date"].to_numpy()


def stack_forecasts(F, y, idx, test_idx, kind, seed):
    """F: (len(idx), n_bases) forecasts aligned with idx; y: actuals aligned with idx."""
    pos = {i: k for k, i in enumerate(idx)}
    out = []
    for t in test_idx:
        k = pos[t]
        Xtr, ytr = F[:k], y[:k]
        if kind == "mean" or len(ytr) < MIN_STACK_TRAIN:
            out.append(F[k].mean())
        elif kind == "median":
            out.append(np.median(F[k]))
        elif kind == "rf":
            m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1).fit(Xtr, ytr)
            out.append(m.predict(F[k][None])[0])
        elif kind == "nnls":
            w, _ = nnls(Xtr, ytr)
            out.append(float(F[k] @ w))
        elif kind == "ridge":
            m = Ridge(alpha=1.0).fit(Xtr / Xtr.mean(), ytr / Xtr.mean())
            out.append(m.predict(F[k][None] / Xtr.mean())[0] * Xtr.mean())
    return np.array(out)


def run_market(market):
    df = load_market(market)
    rows = []
    for seed in SEEDS:
        cols, idx, y, dates = [], None, None, None
        for b in BASES:
            i, f, a, dt = get(df, b, seed)
            idx, y, dates = i, a, dt
            cols.append(f)
        F = np.column_stack(cols)
        test_idx = idx[-N_TEST:]
        dmap = dict(zip(idx, dates))
        for name, kind in [("HS_RF", "rf"), ("HS_NNLS", "nnls"), ("HS_Ridge", "ridge"),
                           ("Comb_Mean", "mean"), ("Comb_Median", "median")]:
            f =stack_forecasts(F, y, idx, test_idx, kind, seed)
            yy = np.array([y[list(idx).index(t)] for t in test_idx])
            rows.append(pd.DataFrame({"market": market, "model": name, "seed": seed, "idx": test_idx,
                                      "date": [dmap[t] for t in test_idx], "actual": yy, "forecast": f}))
            if name == "Comb_Mean":
                # The original paper's "Custom Boosting" (Algorithm 1) is exactly the equal-weight mean of the
                # four base forecasts, so it is kept under its original name as an alias of Comb_Mean.
                rows.append(pd.DataFrame({"market": market, "model": "Custom_Boosting", "seed": seed,
                                          "idx": test_idx, "date": [dmap[t] for t in test_idx],
                                          "actual": yy, "forecast": f}))
    pd.concat(rows).to_csv(RESULTS / "forecasts" / f"{market}_ens.csv", index=False)


if __name__ == "__main__":
    for m in (sys.argv[1:] or MARKETS):
        run_market(m)
        print("done", m, flush=True)
