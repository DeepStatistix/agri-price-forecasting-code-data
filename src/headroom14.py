"""Headroom analysis (validation weeks only) repeated with all 14 base models incl. pooled ones."""
import numpy as np
import pandas as pd

import v3_stack as S
from config import MARKETS, RESULTS
from headroom import mape, simplex_fit
from util import md

T_OUT = RESULTS / "tables"


def main():
    Y, _ = S.load_Y()
    T = S.load_forecast_tensors(Y)
    N = Y.shape[1]
    vT = np.arange(N - 2 * S.N_EVAL, N - S.N_EVAL)
    scale = Y[:, : N - 2 * S.N_EVAL].mean(1)
    rows, corr = [], []
    sets = {"per-market 10": S.POOL10, "all 14": S.POOL, "pooled only (4)": ["G_Ridge", "G_RF", "G_XGB", "G_MLP"]}
    for h in S.H8:
        F = S.build(T, h)
        data = {}
        for mi, mk in enumerate(MARKETS):
            data[mk] = (F[mi, vT - h + 1, :] / scale[mi], Y[mi, vT] / scale[mi], Y[mi, vT - h] / scale[mi])
        for name, pool in sets.items():
            cols = [S.POOL.index(b) for b in pool]
            for mi, mk in enumerate(MARKETS):
                X, a, last = data[mk]
                X = X[:, cols]
                half = len(a) // 2
                cf = np.empty_like(a)
                for tr, te in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
                    cf[te] = X[te] @ simplex_fit(X[tr], a[tr])
                Xp = np.vstack([data[o][0][:, cols] for o in MARKETS if o != mk])
                ap = np.concatenate([data[o][1] for o in MARKETS if o != mk])
                w_in = simplex_fit(X, a)
                rows.append({"pool": name, "h": h, "market": mk, "naive": mape(a, last),
                             "oracle_in": mape(a, X @ w_in), "crossfit": mape(a, cf),
                             "pooled_lomo": mape(a, X @ simplex_fit(Xp, ap))})
            E = np.stack([data[mk][1][:, None] - data[mk][0][:, cols] for mk in MARKETS]).reshape(-1, len(cols), order="C") if False else None
        # mean pairwise error correlation, per-market 10 vs pooled-4 vs cross
        for mi, mk in enumerate(MARKETS):
            X, a, _ = data[mk]
            E = a[:, None] - X
            c = np.corrcoef(E.T)
            i10, i4 = list(range(10)), list(range(10, 14))
            corr.append({"h": h, "market": mk, "corr_within_per_market10": c[np.ix_(i10, i10)][np.triu_indices(10, 1)].mean(),
                         "corr_within_pooled4": c[np.ix_(i4, i4)][np.triu_indices(4, 1)].mean(),
                         "corr_between": c[np.ix_(i10, i4)].mean()})
    R = pd.DataFrame(rows)
    for k in ("oracle_in", "crossfit", "pooled_lomo"):
        R[f"gain_{k}_pct"] = 100 * (R["naive"] - R[k]) / R["naive"]
    R.to_csv(T_OUT / "headroom14.csv", index=False)
    C = pd.DataFrame(corr)
    C.to_csv(T_OUT / "headroom14_error_correlation.csv", index=False)
    g = R.groupby(["pool", "h"])[["gain_oracle_in_pct", "gain_crossfit_pct", "gain_pooled_lomo_pct"]].mean().round(1).unstack("h")
    out = ["# Headroom with pooled models (validation weeks only; relative MAPE gain vs naive, %, mean over six markets)\n",
           "oracle_in = weights fitted on the same weeks (upper bound); crossfit = fit on half / test on other half; pooled_lomo = weights from the other five markets.\n"]
    for k in ("gain_oracle_in_pct", "gain_crossfit_pct", "gain_pooled_lomo_pct"):
        out += [f"## {k}\n", md(g[k].reset_index().rename(columns={"pool": "base pool"})) + "\n"]
    out += ["## Error correlation within and between model groups (validation)\n",
            md(C.groupby("h")[["corr_within_per_market10", "corr_within_pooled4", "corr_between"]].mean().round(3).reset_index()) + "\n"]
    (T_OUT / "HEADROOM14.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
