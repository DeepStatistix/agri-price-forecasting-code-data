"""Case study: frozen final design on Azadpur onion, pooled with a five-market supporting panel.
Stages:  python case_study.py classical | global | final
Outputs: results/case_study/ (forecasts for Azadpur only are reported; supporting-market forecasts are internal inputs)."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import cs_data
import models as M_
import v3_stack as S
from config import RESULTS
from evaluate import holm, mcs
from util import md
from mh_evaluate import dm_hln_h
from v3_global import make_model

H_CS = [1, 2, 4, 8]
N_TEST, N_HIST = 104, 156
REFIT, FIRST = 4, 10
SEEDS = list(range(5))
OUT = RESULTS / "case_study"
S.BASESETS["C3"] = ["Naive", "ETS", "ARIMA"]


def panel():
    w = cs_data.load()
    return w.to_numpy().T.astype(float), w.index, list(w.columns)       # (6, N)


def feature_block(Y, wk, j, h):
    """(6, 14): information up to index j-1; calendar fact = ISO week of the target j+h-1."""
    ly = np.log(Y)
    r = {k: ly[:, j - k] - ly[:, j - k - 1] for k in (1, 2, 3, 4)}
    nm = Y.shape[0]
    oth = {k: (r[k].sum() - r[k]) / (nm - 1) for k in (1, 2)}
    dev = ly[:, j - 1] - np.log(Y[:, j - 9:j - 1].mean(1))
    cal = np.full(nm, float(wk[j + h - 1]))
    return np.column_stack([r[1], r[2], r[3], r[4], oth[1], oth[2], dev, cal, np.eye(nm)])


# ------------------------------------------------------------------ base forecasts
def _classical_market(mi, y, S0, order):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    N = len(y)
    out = {"ETS": np.full((8, N), np.nan), "ARIMA": np.full((8, N), np.nan)}
    for o in range(S0, N):
        for k in ("ETS", "ARIMA"):
            try:
                f = (ARIMA(y[:o], order=order).fit().forecast(8) if k == "ARIMA"
                     else ExponentialSmoothing(y[:o], trend="add", damped_trend=True).fit().forecast(8))
                f = np.asarray(f)
                if not np.all(np.isfinite(f)):
                    f = np.full(8, y[o - 1])
            except Exception:
                f = np.full(8, y[o - 1])
            out[k][:, o] = f
    return mi, out


def run_classical():
    Y, idx, names = panel()
    N = Y.shape[1]
    S0 = N - N_TEST - N_HIST
    orders = [M_.select_arima_order(Y[m, :S0]) for m in range(len(names))]
    res = Parallel(n_jobs=6)(delayed(_classical_market)(m, Y[m], S0, orders[m]) for m in range(len(names)))
    arr = {k: np.full((len(names), 8, N), np.nan) for k in ("ETS", "ARIMA")}
    for mi, out in res:
        for k in out:
            arr[k][mi] = out[k]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "classical.npz", **arr)
    (OUT / "arima_orders.json").write_text(json.dumps({n: list(o) for n, o in zip(names, orders)}))
    print("classical done", orders)


def _global_job(name, seed):
    Y, idx, names = panel()
    nm, N = Y.shape
    wk = idx.isocalendar().week.to_numpy()
    S0 = N - N_TEST - N_HIST
    ly = np.log(Y)
    out = {h: np.full((nm, N), np.nan) for h in H_CS}
    for h in H_CS:
        Xc = {j: feature_block(Y, wk, j, h) for j in range(FIRST, N - h + 1)}
        mdl = None
        for o in range(S0, N - h + 1):
            if mdl is None or (o - S0) % REFIT == 0:
                js = range(FIRST, o - h + 1)                           # targets j+h-1 <= o-1 are observed at o
                X = np.vstack([Xc[j] for j in js])
                y = np.concatenate([ly[:, j + h - 1] - ly[:, j - 1] for j in js])
                mdl = make_model(name, seed).fit(X, y)
            out[h][:, o] = Y[:, o - 1] * np.exp(mdl.predict(Xc[o]))
    return name, seed, out


def run_global():
    jobs = [("G_Ridge", 0)] + [(n, s) for n in ("G_RF", "G_XGB", "G_MLP") for s in SEEDS]
    res = Parallel(n_jobs=8)(delayed(_global_job)(n, s) for n, s in jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "global.npz", **{f"{n}|{s}|{h}": o[h] for n, s, o in res for h in H_CS})
    print("global done")


# ------------------------------------------------------------------ stacking and evaluation
CFGS = {"Mean_B2": {"bases": "B2", "family": "mean", "pooled": 0, "gate": 0, "id": 0},
        "Stack_B2": {"bases": "B2", "family": "nnls", "lam": 0.5, "pooled": 0, "gate": 0, "id": 1},
        "Mean_C3": {"bases": "C3", "family": "mean", "pooled": 0, "gate": 0, "id": 2},
        "Stack_C3": {"bases": "C3", "family": "nnls", "lam": 0.5, "pooled": 0, "gate": 0, "id": 3},
        "Mean_B5": {"bases": "B5", "family": "mean", "pooled": 0, "gate": 0, "id": 4}}
TESTED = ["Mean_B2", "Stack_B2"]


def final():
    Y, idx, names = panel()
    nm, N = Y.shape
    off = N - N_TEST - N_HIST - 104            # trimmed index 104 <-> first base origin
    cl = np.load(OUT / "classical.npz")
    gl = np.load(OUT / "global.npz")
    Yt = Y[:, off:]
    T_idx = np.arange(N - N_TEST, N)
    dates = idx.to_numpy()
    rows = []
    for seed in SEEDS:
        for h in H_CS:
            F = np.full((nm, N, len(S.POOL)), np.nan)
            F[:, 1:, S.POOL.index("Naive")] = Y[:, :-1]
            F[:, :, S.POOL.index("ETS")] = cl["ETS"][:, h - 1, :]
            F[:, :, S.POOL.index("ARIMA")] = cl["ARIMA"][:, h - 1, :]
            for g in ("G_Ridge", "G_RF", "G_XGB", "G_MLP"):
                F[:, :, S.POOL.index(g)] = gl[f"{g}|{0 if g == 'G_Ridge' else seed}|{h}"]
            Ft = F[:, off:, :]
            o_full = T_idx - h + 1
            base = {"Naive": Y[0, o_full - 1], "ETS": F[0, o_full, S.POOL.index("ETS")], "ARIMA": F[0, o_full, S.POOL.index("ARIMA")],
                    "SeasonalNaive52": Y[0, T_idx - 52]}
            for g in ("G_Ridge", "G_RF", "G_XGB", "G_MLP"):
                base[g] = F[0, o_full, S.POOL.index(g)]
            for name, c in CFGS.items():
                base[name] = S.stack_path(c, Ft, Yt, h)[0, o_full - off]
            for m, f in base.items():
                rows.append(pd.DataFrame({"model": m, "seed": seed, "h": h, "target": T_idx, "date": dates[T_idx],
                                          "actual": Y[0, T_idx], "forecast": f}))
    df = pd.concat(rows, ignore_index=True)
    (OUT / "forecasts").mkdir(exist_ok=True)
    df.to_csv(OUT / "forecasts" / "azadpur_onion_forecasts.csv", index=False)
    evaluate(df, Y[0], N)


def evaluate(df, y, N):
    sc = np.mean(np.abs(y[52:N - N_TEST] - y[: N - N_TEST - 52]))
    df = df.copy()
    df["e"] = df["actual"] - df["forecast"]
    df["ape"] = df["e"].abs() / df["actual"] * 100
    per = df.groupby(["model", "seed", "h"]).agg(MAPE=("ape", "mean"), MAE=("e", lambda x: np.mean(np.abs(x))),
                                                 RMSE=("e", lambda x: np.sqrt(np.mean(x ** 2)))).reset_index()
    per["MASE"] = per["MAE"] / sc
    s = per.groupby(["model", "h"])[["MAPE", "MAE", "RMSE", "MASE"]].mean().reset_index()
    s.to_csv(OUT / "cs_metrics.csv", index=False)
    df["se"] = df["e"] ** 2
    comps = []
    fams = {"pre-specified vs references": [(m, r) for m in TESTED for r in ("Naive", "ETS", "ARIMA", "Mean_C3", "SeasonalNaive52")],
            "stacker vs mean of its pool": [("Stack_B2", "Mean_B2"), ("Stack_C3", "Mean_C3")],
            "pooled information": [("Mean_B2", "Mean_C3"), ("Mean_B5", "Mean_C3")]}
    mc_rows = []
    for h in H_CS:
        L = df[df["h"] == h].groupby(["target", "model"])["se"].mean().unstack("model").sort_index()
        m = s[s["h"] == h].set_index("model")["MAPE"]
        for fam, pairs in fams.items():
            for ref in sorted({r for _, r in pairs}):
                grp = [(a, b) for a, b in pairs if b == ref]
                rr = []
                for a, b in grp:
                    st, p = dm_hln_h(L[b], L[a], h)
                    rr.append({"family": fam, "h": h, "model": a, "reference": b, "MAPE_model": m[a], "MAPE_ref": m[b],
                               "pp_diff": m[a] - m[b], "rel_change_pct": 100 * (m[a] - m[b]) / m[b], "DM": st, "p_raw": p})
                for r_, pa in zip(rr, holm([r["p_raw"] for r in rr])):
                    r_["p_holm"] = pa
                comps += rr
        for stat, pv in mcs(L, alpha=0.10).items():
            for k, v in pv.items():
                mc_rows.append({"h": h, "statistic": stat, "model": k, "in_set_10": v >= 0.10, "mcs_p": v})
    C = pd.DataFrame(comps)
    C.to_csv(OUT / "cs_dm.csv", index=False)
    MC = pd.DataFrame(mc_rows)
    MC.to_csv(OUT / "cs_mcs.csv", index=False)
    order = ["Naive", "SeasonalNaive52", "ETS", "ARIMA", "G_Ridge", "G_RF", "G_XGB", "G_MLP", "Mean_C3", "Stack_C3", "Mean_B5",
             "Mean_B2", "Stack_B2"]
    out = ["# Case study: Azadpur onion, frozen final design (generated by src/case_study.py)\n",
           "Test = last 104 weekly targets (2023-11 .. 2025-11); MAPE %, mean over 5 seeds; no re-tuning on these data.\n"]
    pm = s.pivot(index="model", columns="h", values="MAPE").reindex(order)
    out += ["## MAPE (%)\n", md(pm) + "\n"]
    out += ["## Relative to Naive (%)\n", md((pm.div(pm.loc["Naive"], axis=1) * 100 - 100).round(1)) + "\n"]
    out += ["## MASE\n", md(s.pivot(index="model", columns="h", values="MASE").reindex(order), 3) + "\n"]
    out += ["## RMSE (Rs/kg)\n", md(s.pivot(index="model", columns="h", values="RMSE").reindex(order)) + "\n"]
    for fam in fams:
        q = C[C["family"] == fam].drop(columns="family")
        out += [f"## DM: {fam} (positive DM = model better; Holm within horizon x reference)\n", md(q.round(3).set_index("h")) + "\n"]
    n_in = MC[(MC.statistic == "R") & MC.in_set_10].groupby("h").size().to_dict()
    out += [f"## MCS (T_R, 90%): models in set per horizon {n_in}; pool = {MC[(MC.statistic == 'R') & (MC.h == 1)].shape[0]} models\n"]
    for m_ in ["Mean_B2", "Stack_B2", "Mean_C3", "Naive", "ETS", "SeasonalNaive52"]:
        out.append(f"- {m_} in set at h = " + ", ".join(f"{h}: {bool(MC[(MC.statistic == 'R') & (MC.h == h) & (MC.model == m_)].in_set_10.iloc[0])}" for h in H_CS))
    top = MC[MC.statistic == "R"].sort_values("mcs_p", ascending=False).groupby("h").head(1).set_index("h")["model"].to_dict()
    out.append(f"\nRank-1 model (T_R): {top}\n")
    (OUT / "CS_SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:12]))


if __name__ == "__main__":
    {"classical": run_classical, "global": run_global, "final": final}[sys.argv[1]]()
