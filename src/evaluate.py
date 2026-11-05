"""Metrics, Diebold-Mariano (HLN) with Holm correction, and Model Confidence Set from saved forecasts only.
Deterministic: everything here is recomputed from results/forecasts/*.csv."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import data
from config import MARKETS, N_TEST, RESULTS

LABEL = {"HS_RF": "Heterogeneous stacking (RF meta)", "HS_NNLS": "Heterogeneous stacking (NNLS meta)",
         "HS_Ridge": "Heterogeneous stacking (ridge meta)"}
BENCH = ["Naive", "SeasonalNaive", "ETS", "Comb_Mean", "Comb_Median"]
STANDALONE = ["ARIMA", "SVR", "RNN", "LSTM", "DT"]
HOMOG = ["RF", "GB", "XGB", "Bagging_SVR", "Bagging_DT"]
STACK = ["HS_RF", "HS_NNLS", "HS_Ridge", "HBoost", "Custom_Boosting"]  # models compared against references


def load_forecasts(market):
    fr = [pd.read_csv(RESULTS / "forecasts" / f"{market}_{k}.csv", parse_dates=["date"])
          for k in ("base", "deep", "ens", "boost")]
    df = pd.concat(fr, ignore_index=True)
    last = df["idx"].max()
    df = df[df["idx"] > last - N_TEST]
    # deterministic models were run with seed 0 only; broadcast to all seeds for uniform handling
    return df


def mase_scale(market, d):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()[: len(s) - N_TEST]
    sn = data.seasonal_naive_index(s)[: len(y)]
    ok = sn >= 0
    return float(np.mean(np.abs(y[ok] - y[sn[ok]])))


def per_seed_metrics(df, scale):
    rows = []
    for (m, sd), g in df.groupby(["model", "seed"]):
        a, f = g["actual"].to_numpy(), g["forecast"].to_numpy()
        e = a - f
        rows.append({"model": m, "seed": sd, "MAPE": np.mean(np.abs(e / a)) * 100, "MAE": np.mean(np.abs(e)),
                     "RMSE": np.sqrt(np.mean(e ** 2)), "MASE": np.mean(np.abs(e)) / scale,
                     "R2": 1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2)})
    return pd.DataFrame(rows)


def seed_avg_loss(df, loss="se"):
    """Return DataFrame (origins x models) of seed-averaged loss."""
    d = df.copy()
    d["loss"] = (d["actual"] - d["forecast"]) ** 2 if loss == "se" else (d["actual"] - d["forecast"]).abs()
    return d.groupby(["idx", "model"])["loss"].mean().unstack("model").sort_index()


def dm_hln(l1, l2):
    """H0: equal accuracy. Positive stat => model 2 has lower loss than model 1. h=1, HLN small-sample correction."""
    dlt = np.asarray(l1) - np.asarray(l2)
    n = len(dlt)
    if np.allclose(dlt.std(ddof=1), 0):
        return 0.0, 1.0
    stat = dlt.mean() / np.sqrt(dlt.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mcs(loss_df, alpha=0.10, B=5000, block=4, seed=123):
    """Hansen-Lunde-Nason MCS, range statistic T_R and T_max, moving-block bootstrap.
    Returns (included set, p-value of each model = largest alpha at which it stays) for both statistics."""
    rng = np.random.default_rng(seed)
    L = loss_df.to_numpy()
    n, m = L.shape
    names = list(loss_df.columns)
    # bootstrap indices (moving blocks)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]

    def run(stat):
        alive = list(range(m))
        pvals = {}
        pmax_prev = 0.0
        while len(alive) > 1:
            sub = L[:, alive]
            k = len(alive)
            dbar = sub.mean(0)[:, None] - sub.mean(0)[None, :]            # d_ij mean
            boot = L[idx][:, :, alive]                                      # B x n x k
            bm = boot.mean(1)                                               # B x k
            bd = bm[:, :, None] - bm[:, None, :]                            # B x k x k
            dev = bd - dbar[None]
            var = np.maximum((dev ** 2).mean(0), 1e-12)   # guard: identical series would give zero variance
            np.fill_diagonal(var, 1.0)
            t_ij = dbar / np.sqrt(var)
            t_bi = dev / np.sqrt(var)[None]
            if stat == "R":
                tobs = np.abs(t_ij).max()
                tb = np.abs(t_bi).reshape(B, -1).max(1)
            else:
                di = dbar.mean(1)
                vi = np.maximum(((bd.mean(2) - di[None]) ** 2).mean(0), 1e-12)
                ti = di / np.sqrt(vi)
                tobs = ti.max()
                tb = (((bd.mean(2) - di[None]) / np.sqrt(vi)[None])).max(1)
            p = float((tb >= tobs).mean())
            pmax_prev = max(pmax_prev, p)
            if stat == "R":
                # Hansen et al. (2011) elimination rule for T_R: argmax_i max_j d_ij / sd(d_ij)
                worst = int(np.argmax(t_ij.max(1)))
            else:
                worst = int(np.argmax(ti))
            pvals[names[alive[worst]]] = pmax_prev
            alive.pop(worst)
        pvals[names[alive[0]]] = 1.0
        return pvals

    out = {}
    for stat in ("R", "max"):
        pv = run(stat)
        out[stat] = pv
    return out


def evaluate_market(market, d, reference_pool=None):
    df = load_forecasts(market)
    scale = mase_scale(market, d)
    ps = per_seed_metrics(df, scale)
    summ = ps.groupby("model").agg(MAPE=("MAPE", "mean"), MAPE_sd=("MAPE", "std"), MAE=("MAE", "mean"),
                                   RMSE=("RMSE", "mean"), MASE=("MASE", "mean"), R2=("R2", "mean")).reset_index()
    summ.insert(0, "market", market)
    return df, ps, summ


if __name__ == "__main__":
    d = data.load_all()
    all_sum, dm_rows, mcs_rows = [], [], []
    for m in (sys.argv[1:] or MARKETS):
        df, ps, summ = evaluate_market(m, d)
        all_sum.append(summ)
        se = seed_avg_loss(df, "se")
        mape = summ.set_index("model")["MAPE"]
        best_sa = mape[[x for x in STANDALONE if x in mape.index]].idxmin()
        best_hom = mape[[x for x in HOMOG if x in mape.index]].idxmin()
        best_bench = mape[[x for x in BENCH if x in mape.index]].idxmin()
        for st in STACK:
            for ref_kind, ref in [("best standalone", best_sa), ("best homogeneous ensemble", best_hom),
                                  ("best benchmark", best_bench)]:
                stat, p = dm_hln(se[ref], se[st])
                dm_rows.append({"market": m, "model": st, "reference": ref, "reference_kind": ref_kind,
                                "MAPE_model": mape[st], "MAPE_ref": mape[ref],
                                "rel_change_pct": 100 * (mape[st] - mape[ref]) / mape[ref],
                                "pp_diff": mape[st] - mape[ref], "DM": stat, "p_raw": p})
        # Custom_Boosting is an exact alias of Comb_Mean (same forecasts): keep it out of the MCS pool
        sub = [c for c in se.columns if c != "Custom_Boosting"]
        for alpha in (0.10,):
            r = mcs(se[sub], alpha=alpha)
            for stat_name, pv in r.items():
                for k, v in pv.items():
                    mcs_rows.append({"market": m, "statistic": stat_name, "model": k, "mcs_p": v,
                                     "in_set_10": v >= 0.10, "in_set_05": v >= 0.05})
        print("done", m, flush=True)
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(all_sum).to_csv(out / "metrics_summary.csv", index=False)
    dm = pd.DataFrame(dm_rows)
    dm["p_holm"] = dm.groupby(["market", "reference_kind"])["p_raw"].transform(lambda x: holm(x.to_numpy()))
    dm.to_csv(out / "dm_tests.csv", index=False)
    pd.DataFrame(mcs_rows).to_csv(out / "mcs.csv", index=False)
