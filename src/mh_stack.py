"""Validation-tuned heterogeneous stacking, all horizons h = 1..4.
Stage 'tune'  : per market, evaluate all configurations on the 52 VALIDATION targets (indices N-104..N-53).
Stage 'final' : select configurations, then evaluate once on the
                52 TEST targets, 5 seeds.   Usage: python mh_stack.py tune [market...] | python mh_stack.py final"""
import itertools
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from config import MARKETS, MIN_STACK_TRAIN, RESULTS, WARMUP

import os
H = [int(x) for x in os.environ.get("MH_H", "1,2,3,4").split(",")]
SUF = os.environ.get("MH_SUF", "")
H_SEL = [1, 2, 3, 4]   # horizons used for tuning/selection
N_EVAL = 52
FDIR = RESULTS / "mh" / "forecasts"
TDIR = RESULTS / "mh" / "tuning"
DET = {"Naive", "SeasonalNaive", "ETS", "ARIMA", "SVR", "DT"}
POOL = ["Naive", "ETS", "ARIMA", "SVR", "RNN", "LSTM", "DT", "RF", "GB", "XGB"]
BASE_SETS = {"S1": ["ARIMA", "SVR", "LSTM", "DT"],
             "S2": ["Naive", "ETS", "ARIMA", "SVR", "LSTM", "DT"],
             "S3": ["ETS", "ARIMA", "SVR", "DT", "RF", "XGB"],
             "S4": POOL}
META_RANK = {"mean": 0, "median": 1, "nnls": 2, "ridge": 3, "rf": 4}


# ------------------------------------------------------------------ data
def load_base(market):
    parts = [pd.read_csv(FDIR / f"{market}_mh_{k}{SUF}.csv") for k in ("classical", "tabular", "deep")]
    df = pd.concat(parts, ignore_index=True)
    import data
    y = data.series(data.load_all(), market)["MODAL"].astype(float).to_numpy()
    assert len(y) == int(df["target"].max()) + 1 and np.isfinite(y).all()
    return df, y


def matrix(df, N, h, model, seed):
    s = 0 if model in DET else seed
    g = df[(df["model"] == model) & (df["h"] == h) & (df["seed"] == s)]
    a = np.full(N, np.nan)
    a[g["origin"].to_numpy()] = g["forecast"].to_numpy()
    return a


def build_F(df, N, seed=None):
    """F[h] -> (N, len(POOL)) indexed by origin. seed=None: mean over seeds of stochastic models (tuning)."""
    out = {}
    for h in H:
        cols = []
        for m in POOL:
            if seed is None and m not in DET:
                seeds = sorted(df[(df["model"] == m)]["seed"].unique())
                cols.append(np.mean([matrix(df, N, h, m, s) for s in seeds], axis=0))
            else:
                cols.append(matrix(df, N, h, m, 0 if seed is None else seed))
        out[h] = np.column_stack(cols)
    return out


# ------------------------------------------------------------------ stacker
def stack_forecast(Fh, y, h, origins, cfg, seed=0):
    """Forecast y[o+h-1] for each origin o using rows (F[i], y[i+h-1]) with WARMUP <= i <= o-h only."""
    cols = [POOL.index(b) for b in BASE_SETS[cfg["bases"]]]
    out = []
    for o in origins:
        fo = Fh[o, cols]
        rows = np.arange(WARMUP, o - h + 1)
        if cfg["window"] == "recent":
            rows = rows[-52:]
        meta = cfg["meta"]
        if meta == "mean" or len(rows) < MIN_STACK_TRAIN:
            out.append(float(fo.mean()))
            continue
        if meta == "median":
            out.append(float(np.median(fo)))
            continue
        X, Y = Fh[rows][:, cols], y[rows + h - 1]
        last_i, last_o = y[rows - 1], y[o - 1]
        rel = cfg["repr"] == "rel"
        if rel:
            X, Y, fo = X - last_i[:, None], Y - last_i, fo - last_o
        s = np.mean(np.abs(Y)) if rel else X.mean()
        s = s if s > 0 else 1.0
        if meta == "nnls":
            w, _ = nnls(X, Y)
            p = float(fo @ w)
        elif meta == "ridge":
            m = Ridge(alpha=cfg["alpha"], fit_intercept=not rel).fit(X / s, Y / s)
            p = float(m.predict((fo / s)[None])[0] * s)
        else:
            m = RandomForestRegressor(n_estimators=cfg.get("trees", 50), max_depth=cfg["depth"],
                                      min_samples_leaf=cfg["leaf"], random_state=seed, n_jobs=1).fit(X, Y)
            p = float(m.predict(fo[None])[0])
        out.append(p + (last_o if rel else 0.0))
    return np.array(out)


def enumerate_configs():
    cfgs = []
    for b in BASE_SETS:
        for meta in ("mean", "median"):
            cfgs.append({"bases": b, "meta": meta, "repr": "level", "window": "expanding"})
        for rep, win in itertools.product(("level", "rel"), ("expanding", "recent")):
            cfgs.append({"bases": b, "meta": "nnls", "repr": rep, "window": win})
            for a in (0.1, 1, 10, 100):
                cfgs.append({"bases": b, "meta": "ridge", "alpha": a, "repr": rep, "window": win})
            for dep, leaf in itertools.product((3, None), (3, 5, 10)):
                cfgs.append({"bases": b, "meta": "rf", "depth": dep, "leaf": leaf, "repr": rep, "window": win})
    for i, c in enumerate(cfgs):
        c["id"] = i
    return cfgs


def cfg_name(c):
    extra = {"ridge": f"a{c.get('alpha')}", "rf": f"d{c.get('depth')}l{c.get('leaf')}"}.get(c["meta"], "")
    return f"{c['bases']}|{c['meta']}{extra}|{c['repr']}|{c['window']}"


def simplicity_key(c):
    return (len(BASE_SETS[c["bases"]]), META_RANK[c["meta"]], c["repr"] == "rel", c["window"] == "recent", c["id"])


def pick(scores, cfgs, tol=0.02):
    """scores: array aligned with cfgs. Parsimony rule."""
    best = np.nanmin(scores)
    cand = [c for c, s in zip(cfgs, scores) if s <= best * (1 + tol)]
    return min(cand, key=simplicity_key)


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


# ------------------------------------------------------------------ stage 1
def tune_market(market):
    df, y = load_base(market)
    N = len(y)
    F = build_F(df, N)                     # seed-averaged stochastic bases for tuning
    val_T = np.arange(N - 2 * N_EVAL, N - N_EVAL)
    rows = []
    for c in enumerate_configs():
        r = {"market": market, "id": c["id"], "name": cfg_name(c)}
        for h in H_SEL:
            o = val_T - h + 1
            r[f"mape_h{h}"] = mape(y[val_T], stack_forecast(F[h], y, h, o, c, 0))
        r["mape_mean"] = np.mean([r[f"mape_h{h}"] for h in H_SEL])
        rows.append(r)
    out = pd.DataFrame(rows)
    TDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TDIR / f"{market}_stack_validation.csv", index=False)
    return market


# ------------------------------------------------------------------ stage 2
def naive_val_mape(market):
    df, y = load_base(market)
    N = len(y)
    val_T = np.arange(N - 2 * N_EVAL, N - N_EVAL)
    return np.mean([mape(y[val_T], y[val_T - h]) for h in H_SEL])   # naive: y[o-1] = y[T-h]


def final_market(market, cfgs, chosen):
    df, y = load_base(market)
    N = len(y)
    test_T = np.arange(N - N_EVAL, N)
    dates = df.drop_duplicates("target").set_index("target")["date"]
    named = dict(chosen[market])
    named.update({
        "HS_RF_orig": {"bases": "S1", "meta": "rf", "depth": None, "leaf": 1, "trees": 100, "repr": "level",
                       "window": "expanding"},
        "HS_NNLS_orig": {"bases": "S1", "meta": "nnls", "repr": "level", "window": "expanding"},
        "HS_Ridge_orig": {"bases": "S1", "meta": "ridge", "alpha": 1, "repr": "level", "window": "expanding"},
        "Comb_Mean": {"bases": "S1", "meta": "mean", "repr": "level", "window": "expanding"},
        "Comb_Median": {"bases": "S1", "meta": "median", "repr": "level", "window": "expanding"}})
    rows = []
    for seed in range(5):
        F = build_F(df, N, seed=seed)
        for name, c in named.items():
            if c["meta"] in ("mean", "median") and seed > 0 and name not in ("Comb_Mean", "Comb_Median") and \
                    all(b in DET for b in BASE_SETS[c["bases"]]):
                pass
            for h in H:
                o = test_T - h + 1
                f = stack_forecast(F[h], y, h, o, c, seed)
                rows.append(pd.DataFrame({"market": market, "model": name, "seed": seed, "h": h, "origin": o,
                                          "target": test_T, "date": dates.loc[test_T].to_numpy(),
                                          "actual": y[test_T], "forecast": f}))
    res = pd.concat(rows, ignore_index=True)
    alias = res[res["model"] == "Comb_Mean"].assign(model="Custom_Boosting")
    return pd.concat([res, alias], ignore_index=True)


def select(markets):
    cfgs = enumerate_configs()
    val = {m: pd.read_csv(TDIR / f"{m}_stack_validation.csv").sort_values("id") for m in markets}
    comb_mask = np.array([c["meta"] in ("mean", "median") for c in cfgs])
    chosen, report = {}, []
    # global: score relative to naive validation MAPE, averaged over markets
    rel = np.mean([val[m]["mape_mean"].to_numpy() / naive_val_mape(m) for m in markets], axis=0)
    g = pick(rel, cfgs)
    for m in markets:
        s = val[m]["mape_mean"].to_numpy()
        mk = pick(s, cfgs)
        cb = pick(np.where(comb_mask, s, np.inf), cfgs)
        chosen[m] = {"HS_tuned_mkt": mk, "HS_tuned_glob": g, "Comb_tuned": cb}
        for k, c in chosen[m].items():
            report.append({"market": m, "variant": k, "config": cfg_name(c), "val_mape_mean": s[c["id"]],
                           "val_naive_mape": naive_val_mape(m), **{f"val_h{h}": val[m][f"mape_h{h}"][c["id"]] for h in H_SEL}})
    return chosen, pd.DataFrame(report)


if __name__ == "__main__":
    mode = sys.argv[1]
    mk = sys.argv[2:] or MARKETS
    if mode == "tune":
        for m in Parallel(n_jobs=min(6, len(mk)), return_as="generator")(delayed(tune_market)(m) for m in mk):
            print("tuned", m, flush=True)
    else:
        cfgs = enumerate_configs()
        chosen, rep = select(mk)
        rep.to_csv(TDIR / f"selected_configs{SUF}.csv", index=False)
        print(rep.to_string(), flush=True)
        for m, res in zip(mk, Parallel(n_jobs=min(6, len(mk)))(delayed(final_market)(m, cfgs, chosen) for m in mk)):
            res.to_csv(FDIR / f"{m}_mh_final{SUF}.csv", index=False)
            print("final", m, flush=True)
