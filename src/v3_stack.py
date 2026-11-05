"""Stack over 14 base models (10 per-market + 4 pooled/global), online and shrunk-NNLS meta-learners with
optional gate, horizons 1..8.  Stage 'tune' (validation only) -> 'final' (test, once).
Usage: python v3_stack.py tune | python v3_stack.py final"""
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls

import data
from config import MARKETS, RESULTS, WARMUP

H8 = list(range(1, 9))
N_EVAL = 52
M = len(MARKETS)
FDIR = RESULTS / "mh" / "forecasts"
VDIR = RESULTS / "v3" / "forecasts"
TDIR = RESULTS / "v3" / "tuning"
POOL10 = ["Naive", "ETS", "ARIMA", "SVR", "RNN", "LSTM", "DT", "RF", "GB", "XGB"]
POOL = POOL10 + ["G_Ridge", "G_RF", "G_XGB", "G_MLP"]
DET = {"Naive", "ETS", "ARIMA", "SVR", "DT", "G_Ridge"}
BASESETS = {"B1": POOL, "B2": ["Naive", "ETS", "ARIMA", "G_Ridge", "G_RF", "G_XGB", "G_MLP"], "B3": POOL10,
            "B4": ["Naive", "ETS", "ARIMA", "SVR", "DT", "G_Ridge", "G_XGB"],
            "B5": ["Naive", "G_Ridge", "G_RF", "G_XGB", "G_MLP"]}
MIN_ORIGINS = 20
GATE_WIN = 26
FAMILY_RANK = {"mean": 0, "median": 1, "nnls": 2, "simplex": 2.5, "online": 3}
GATE_MARGIN = {0: None, 1: 0.0, 2: 0.03}
SIMPLEX_ROW = 10.0   # weight of the sum-to-one constraint row in the augmented NNLS


# ---------------------------------------------------------------- loading
def load_Y():
    d = data.load_all()
    S = [data.series(d, m) for m in MARKETS]
    return np.array([s["MODAL"].astype(float).to_numpy() for s in S]), S[0]["date"].to_numpy()


def load_forecast_tensors(Y):
    """dict (model, h, seed) -> (M, N) array of forecasts indexed by origin."""
    N = Y.shape[1]
    cols = ["market", "model", "seed", "h", "origin", "forecast"]
    parts = []
    for mk in MARKETS:
        for k in ("classical", "tabular", "deep"):
            for suf in ("", "_ext"):
                parts.append(pd.read_csv(FDIR / f"{mk}_mh_{k}{suf}.csv", usecols=cols))
    parts.append(pd.read_csv(VDIR / "global.csv", usecols=cols))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["model"].isin(POOL) & df["h"].isin(H8)]
    mi = {m: i for i, m in enumerate(MARKETS)}
    T = {}
    for (model, h, seed), g in df.groupby(["model", "h", "seed"]):
        a = np.full((M, N), np.nan)
        a[g["market"].map(mi).to_numpy(), g["origin"].to_numpy()] = g["forecast"].to_numpy()
        T[(model, h, seed)] = a
    return T


def build(T, h, seed=None):
    """(M, N, len(POOL)); seed=None -> mean over seeds of stochastic models."""
    cols = []
    for m in POOL:
        if m in DET:
            cols.append(T[(m, h, 0)])
        elif seed is None:
            cols.append(np.mean([v for (mm, hh, s), v in T.items() if mm == m and hh == h], axis=0))
        else:
            cols.append(T[(m, h, seed)])
    return np.stack(cols, axis=2)


# ---------------------------------------------------------------- meta-learners
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def stack_path(cfg, Fh, Y, h):
    """Forecast path (M, N) for origins WARMUP..N-h. At origin o only origins i <= o-h (resolved targets) are used."""
    N = Y.shape[1]
    cols = [POOL.index(b) for b in BASESETS[cfg["bases"]]]
    Fb = Fh[:, :, cols]
    k = len(cols)
    ni = cols.index(POOL.index("Naive"))
    out = np.full((M, N), np.nan)
    mean_path = np.full((M, N), np.nan)
    for o in range(WARMUP, N - h + 1):
        mean_path[:, o] = Fb[:, o, :].mean(1)
    fam = cfg["family"]
    if fam == "mean":
        return mean_path
    if fam == "median":
        for o in range(WARMUP, N - h + 1):
            out[:, o] = np.median(Fb[:, o, :], axis=1)
        return out
    idx_all = np.arange(WARMUP, N - h + 1)
    base_last = Y[:, idx_all - 1]                      # y[i-1]
    tgt = Y[:, idx_all + h - 1]
    ell = ((tgt[:, :, None] - Fb[:, idx_all, :]) / base_last[:, :, None]) ** 2      # (M, I, k)
    dev = (Fb[:, idx_all, :] - base_last[:, :, None]) / base_last[:, :, None]        # (M, I, k)
    dtg = (tgt - base_last) / base_last                                                # (M, I)
    for pos, o in enumerate(idx_all):
        n_res = o - h - WARMUP + 1                    # resolved origins WARMUP..o-h
        if n_res < MIN_ORIGINS:
            out[:, o] = mean_path[:, o]
            continue
        res = slice(0, n_res)
        lo = Y[:, o - 1]
        if fam == "online":
            g = cfg["gamma"] ** (n_res - 1 - np.arange(n_res))
            if cfg["pooled"]:
                L = np.einsum("i,mib->b", g, ell[:, res, :])
                mean_naive = L[ni] / (g.sum() * M)
                w = softmax(-cfg["eta"] * L / max(mean_naive, 1e-18))
                out[:, o] = lo * (1 + dev[:, pos, :] @ w)
            else:
                Lm = np.einsum("i,mib->mb", g, ell[:, res, :])
                for m in range(M):
                    mean_naive = Lm[m, ni] / g.sum()
                    w = softmax(-cfg["eta"] * Lm[m] / max(mean_naive, 1e-18))
                    out[m, o] = lo[m] * (1 + dev[m, pos, :] @ w)
        else:  # shrunk nnls ("nnls") or simplex-constrained least squares ("simplex") on deviations from naive
            lam = cfg["lam"]

            def fit_w(A, b):
                if fam == "simplex":
                    w, _ = nnls(np.vstack([A, SIMPLEX_ROW * np.ones((1, k))]), np.append(b, SIMPLEX_ROW))
                    w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
                else:
                    w, _ = nnls(A, b)
                return (1 - lam) * w + lam / k

            if cfg["pooled"]:
                w = fit_w(dev[:, res, :].reshape(-1, k), dtg[:, res].reshape(-1))
                out[:, o] = lo * (1 + dev[:, pos, :] @ w)
            else:
                for m in range(M):
                    w = fit_w(dev[m, res, :], dtg[m, res])
                    out[m, o] = lo[m] * (1 + dev[m, pos, :] @ w)
    if cfg.get("gate"):
        margin = GATE_MARGIN[cfg["gate"]]
        gated = out.copy()
        for o in range(WARMUP, N - h + 1):
            hi = o - h
            lo_i = max(WARMUP, hi - GATE_WIN + 1)
            if hi < lo_i:
                continue
            ii = np.arange(lo_i, hi + 1)
            t = Y[:, ii + h - 1]
            e_s = np.nanmean(np.abs(t - out[:, ii]) / t)
            e_m = np.nanmean(np.abs(t - mean_path[:, ii]) / t)
            if e_s > e_m * (1 - margin):
                gated[:, o] = mean_path[:, o]
        return gated
    return out


def enumerate_configs():
    cfgs = []
    for b in BASESETS:
        for fam in ("mean", "median"):
            cfgs.append({"bases": b, "family": fam, "pooled": 0, "gate": 0})
        for gate, pooled in itertools.product((0, 1), (0, 1)):
            for eta, gamma in itertools.product((0.5, 1, 2, 4), (1.0, 0.97, 0.9)):
                cfgs.append({"bases": b, "family": "online", "eta": eta, "gamma": gamma, "pooled": pooled, "gate": gate})
            for lam in (0, 0.5, 0.8):
                cfgs.append({"bases": b, "family": "nnls", "lam": lam, "pooled": pooled, "gate": gate})
    for i, c in enumerate(cfgs):
        c["id"] = i
    return cfgs


def cfg_name(c):
    ex = {"online": f"eta{c.get('eta')}g{c.get('gamma')}", "nnls": f"lam{c.get('lam')}"}.get(c["family"], "")
    return f"{c['bases']}|{c['family']}{ex}|{'pooled' if c['pooled'] else 'mkt'}|gate{c['gate']}"


def simplicity_key(c):
    return (len(BASESETS[c["bases"]]), FAMILY_RANK[c["family"]], c["gate"], c["pooled"], c["id"])


def pick(score, cfgs, tol=0.02):
    cand = [c for c, s in zip(cfgs, score) if s <= np.nanmin(score) * (1 + tol)]
    return min(cand, key=simplicity_key)


def val_mape_for_h(h, cfgs, Y, T):
    N = Y.shape[1]
    Fh = build(T, h)
    vT = np.arange(N - 2 * N_EVAL, N - N_EVAL)
    o = vT - h + 1
    res = np.zeros((len(cfgs), M))
    for ci, c in enumerate(cfgs):
        p = stack_path(c, Fh, Y, h)
        res[ci] = np.mean(np.abs(Y[:, vT] - p[:, o]) / Y[:, vT], axis=1) * 100
    naive = np.mean(np.abs(Y[:, vT] - Y[:, vT - h]) / Y[:, vT], axis=1) * 100
    return res, naive


def tune():
    Y, _ = load_Y()
    T = load_forecast_tensors(Y)
    cfgs = enumerate_configs()
    out = Parallel(n_jobs=8)(delayed(val_mape_for_h)(h, cfgs, Y, T) for h in H8)
    TDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, (res, naive) in zip(H8, out):
        for ci, c in enumerate(cfgs):
            for mi, mk in enumerate(MARKETS):
                rows.append({"id": c["id"], "name": cfg_name(c), "h": h, "market": mk,
                             "val_mape": res[ci, mi], "val_naive": naive[mi]})
    pd.DataFrame(rows).to_csv(TDIR / "v3_validation.csv", index=False)
    print("tuned", len(cfgs), "configs")


def select():
    cfgs = enumerate_configs()
    v = pd.read_csv(TDIR / "v3_validation.csv")
    v["rel"] = v["val_mape"] / v["val_naive"]
    comb = np.array([c["family"] in ("mean", "median") for c in cfgs])

    def score(hs):
        return v[v["h"].isin(hs)].groupby("id")["rel"].mean().sort_index().to_numpy()

    chosen = {}
    s_all = score(H8)
    chosen["V3_stack"] = pick(np.where(comb, np.inf, s_all), cfgs)
    chosen["V3_comb"] = pick(np.where(comb, s_all, np.inf), cfgs)
    for name, hs in (("V3_stack_h12", [1, 2]), ("V3_stack_h34", [3, 4]), ("V3_stack_h58", [5, 6, 7, 8])):
        chosen[name] = pick(np.where(comb, np.inf, score(hs)), cfgs)
    rep = pd.DataFrame([{"variant": k, "config": cfg_name(c), "val_rel_mape_vs_naive_all_h": s_all[c["id"]]}
                        for k, c in chosen.items()])
    rep["n_configs_searched"] = len(cfgs)
    rep.to_csv(TDIR / "v3_selected.csv", index=False)
    return chosen, rep


def final_seed(seed, chosen, Y, T, dates):
    N = Y.shape[1]
    tT = np.arange(N - N_EVAL, N)
    rows = []
    for h in H8:
        Fh = build(T, h, seed=seed)
        cache = {}
        for name, c in chosen.items():
            if c["id"] not in cache:
                cache[c["id"]] = stack_path(c, Fh, Y, h)
            p = cache[c["id"]]
            o = tT - h + 1
            for mi, mk in enumerate(MARKETS):
                rows.append(pd.DataFrame({"market": mk, "model": name, "seed": seed, "h": h, "origin": o, "target": tT,
                                          "date": dates[tT], "actual": Y[mi, tT], "forecast": p[mi, o]}))
    return pd.concat(rows, ignore_index=True)


def final():
    Y, dates = load_Y()
    T = load_forecast_tensors(Y)
    chosen, rep = select()
    print(rep.to_string())
    group = {1: "V3_stack_h12", 2: "V3_stack_h12", 3: "V3_stack_h34", 4: "V3_stack_h34"}
    group.update({h: "V3_stack_h58" for h in (5, 6, 7, 8)})
    res = Parallel(n_jobs=5)(delayed(final_seed)(s, chosen, Y, T, dates) for s in range(5))
    df = pd.concat(res, ignore_index=True)
    byh = pd.concat([df[(df["model"] == group[h]) & (df["h"] == h)].assign(model="V3_stack_by_h") for h in H8])
    df = pd.concat([df[~df["model"].isin(["V3_stack_h12", "V3_stack_h34", "V3_stack_h58"])], byh], ignore_index=True)
    df.to_csv(VDIR / "v3_final.csv", index=False)
    print("final written", len(df))


if __name__ == "__main__":
    {"tune": tune, "final": final}[sys.argv[1]]()
