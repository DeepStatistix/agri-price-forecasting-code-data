"""Evaluation: per-horizon metrics, DM (HLN, h-1 lags) vs validation-selected references, tables.
Writes results/tables/mh_*.csv and results/tables/MH_SUMMARY.md"""
import numpy as np
import pandas as pd
from scipy import stats

import data
from config import MARKETS, RESULTS
from evaluate import holm
from util import md
from mh_stack import FDIR, H, N_EVAL, SUF, TDIR, load_base

T = RESULTS / "tables"
STANDALONE = ["ARIMA", "SVR", "RNN", "LSTM", "DT"]
HOMOG = ["RF", "GB", "XGB"]
BENCH = ["Naive", "SeasonalNaive", "ETS", "Comb_Mean", "Comb_Median"]
TESTED = ["HS_tuned_mkt", "HS_tuned_glob", "HS_RF_orig", "Comb_tuned"]
ORDER = ["Naive", "SeasonalNaive", "ETS", "ARIMA", "SVR", "RNN", "LSTM", "DT", "RF", "GB", "XGB", "Comb_Mean",
         "Comb_Median", "Custom_Boosting", "HS_RF_orig", "HS_NNLS_orig", "HS_Ridge_orig", "Comb_tuned",
         "HS_tuned_mkt", "HS_tuned_glob"]


def dm_hln_h(l1, l2, h):
    """Positive => model 2 has lower loss. Truncated long-run variance with h-1 lags, HLN correction, t(n-1)."""
    d = np.asarray(l1) - np.asarray(l2)
    n = len(d)
    dc = d - d.mean()
    g = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 1e-12:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def mase_scale(market, d):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()[: len(s) - N_EVAL]
    sn = data.seasonal_naive_index(s)[: len(y)]
    ok = sn >= 0
    return float(np.mean(np.abs(y[ok] - y[sn[ok]])))


def val_mape_base(df, y):
    """Validation MAPE (mean over h, mean over seeds) of every base model."""
    N = len(y)
    v = df[(df["target"] >= N - 2 * N_EVAL) & (df["target"] < N - N_EVAL)].copy()
    v["ape"] = np.abs(v["actual"] - v["forecast"]) / v["actual"] * 100
    return v.groupby(["model", "seed", "h"])["ape"].mean().groupby("model").mean()


def main():
    d = data.load_all()
    sel = pd.read_csv(TDIR / f"selected_configs{SUF}.csv")
    frozen_refs = pd.read_csv(T / "mh_references_chosen_on_validation.csv").set_index("market") if SUF else None
    summ, dmrows, valrows = [], [], []
    for mk in MARKETS:
        base, y = load_base(mk)
        N = len(y)
        fin = pd.read_csv(FDIR / f"{mk}_mh_final{SUF}.csv")
        test_base = base[base["target"] >= N - N_EVAL]
        df = pd.concat([test_base.drop(columns=[]), fin], ignore_index=True)
        df = df[df["target"] >= N - N_EVAL]
        sc = mase_scale(mk, d)
        df["e"] = df["actual"] - df["forecast"]
        df["ape"] = np.abs(df["e"] / df["actual"]) * 100
        g = df.groupby(["model", "seed", "h"])
        per = g.agg(MAPE=("ape", "mean"), MAE=("e", lambda x: np.mean(np.abs(x))), RMSE=("e", lambda x: np.sqrt(np.mean(x ** 2)))).reset_index()
        per["MASE"] = per["MAE"] / sc
        s = per.groupby(["model", "h"]).agg(MAPE=("MAPE", "mean"), MAPE_sd=("MAPE", "std"), MAE=("MAE", "mean"),
                                            RMSE=("RMSE", "mean"), MASE=("MASE", "mean")).reset_index()
        s.insert(0, "market", mk)
        summ.append(s)
        # references on VALIDATION mape
        if SUF:   # references stay those chosen on the h = 1..4 validation
            refs = frozen_refs.loc[mk].to_dict()
        else:
            vb = val_mape_base(base, y)
            vt = pd.read_csv(TDIR / f"{mk}_stack_validation.csv").set_index("name")["mape_mean"]
            vb["Comb_Mean"], vb["Comb_Median"] = vt["S1|mean|level|expanding"], vt["S1|median|level|expanding"]
            refs = {"best standalone": vb[STANDALONE].idxmin(), "best homogeneous ensemble": vb[HOMOG].idxmin(),
                    "best benchmark": vb[BENCH].idxmin()}
        valrows.append({"market": mk, **refs})
        df["se"] = df["e"] ** 2
        for h in H:
            L = df[df["h"] == h].groupby(["target", "model"])["se"].mean().unstack("model").sort_index()
            m = s[s["h"] == h].set_index("model")["MAPE"]
            for rk, ref in refs.items():
                rows = []
                for mod in TESTED:
                    st, p = dm_hln_h(L[ref], L[mod], h)
                    rows.append({"market": mk, "h": h, "model": mod, "reference_kind": rk, "reference": ref,
                                 "MAPE_model": m[mod], "MAPE_ref": m[ref], "pp_diff": m[mod] - m[ref],
                                 "rel_change_pct": 100 * (m[mod] - m[ref]) / m[ref], "DM": st, "p_raw": p})
                ps = holm([r["p_raw"] for r in rows])
                for r, pa in zip(rows, ps):
                    r["p_holm"] = pa
                dmrows += rows
    S = pd.concat(summ, ignore_index=True)
    S.to_csv(T / f"mh_metrics{SUF}.csv", index=False)
    D = pd.DataFrame(dmrows)
    D.to_csv(T / f"mh_dm{SUF}.csv", index=False)
    if not SUF:
        pd.DataFrame(valrows).to_csv(T / "mh_references_chosen_on_validation.csv", index=False)

    out = ["# Multi-horizon results (generated by src/mh_evaluate.py)\n",
           "Test = last 52 weekly targets per market; forecasts at horizon h are made from data up to h weeks earlier. "
           "MAPE in %, mean over seeds. 'tuned' variants were selected on validation targets only.\n"]
    for h in H:
        p = S[S["h"] == h].pivot(index="model", columns="market", values="MAPE")
        p = p.reindex([x for x in ORDER if x in p.index])[MARKETS]
        p["mean of markets"] = p.mean(axis=1)
        out.append(f"## MAPE, horizon h = {h}\n")
        out.append(md(p) + "\n")
    out.append("## Mean MAPE over the six markets, by horizon\n")
    pm = S.groupby(["model", "h"])["MAPE"].mean().unstack("h")
    pm = pm.reindex([x for x in ORDER if x in pm.index])
    out.append(md(pm) + "\n")
    out.append("## Mean MASE over the six markets, by horizon\n")
    pm2 = S.groupby(["model", "h"])["MASE"].mean().unstack("h")
    out.append(md(pm2.reindex([x for x in ORDER if x in pm2.index]), 3) + "\n")
    out.append("## Lowest-MAPE model per market and horizon\n")
    best = S.loc[S.groupby(["market", "h"])["MAPE"].idxmin()].pivot(index="market", columns="h", values="model")
    out.append(md(best.reindex(MARKETS)) + "\n")
    out.append("## References chosen on VALIDATION MAPE\n")
    out.append(md(pd.DataFrame(valrows).set_index("market")) + "\n")
    out.append("## Selected stacking configurations (selected on h = 1..4 validation); test MAPE = mean over the horizons shown\n")
    tm = S.groupby(["market", "model"])["MAPE"].mean()
    sel["test_mape_mean"] = [tm[(r.market, r.variant)] for r in sel.itertuples()]
    sel["test_minus_val_pp"] = sel["test_mape_mean"] - sel["val_mape_mean"]
    out.append(md(sel[["market", "variant", "config", "val_mape_mean", "test_mape_mean", "test_minus_val_pp"]].set_index("market"), 2) + "\n")
    out.append("## DM (HLN) of validation-tuned and original stacking vs references; positive DM = stacking better\n")
    D2 = D.copy()
    D2["sig_holm_05"] = D2["p_holm"] < 0.05
    out.append(md(D2[D2["reference_kind"] == "best benchmark"][["market", "h", "model", "reference", "MAPE_model", "MAPE_ref", "pp_diff", "DM", "p_holm", "sig_holm_05"]].set_index("market"), 3) + "\n")
    sig = D2[D2["sig_holm_05"]]
    out.append("## All Holm-significant (p<0.05) comparisons, any reference\n")
    out.append(md(sig[["market", "h", "model", "reference_kind", "reference", "pp_diff", "DM", "p_holm"]].set_index("market"), 3) if len(sig) else "None.\n")
    (T / f"MH_SUMMARY{SUF}.md").write_text("\n".join(out), encoding="utf-8")
    print("written MH_SUMMARY.md")


if __name__ == "__main__":
    main()
