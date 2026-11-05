"""Robustness: test performance of the configurations re-selected on unflagged validation weeks (reselect_clean.py),
compared with the original selections, on all test weeks and on unflagged test weeks. Not used for any design decision."""
import numpy as np
import pandas as pd

import data
import v3_stack as S
import v31_stack as V
from config import MARKETS, RESULTS
from data_quality import flags
from util import md


def cfg(base, fam, lam=None):
    c = {"bases": base, "family": fam, "pooled": 0, "gate": 0}
    if lam is not None:
        c["lam"] = lam
    return c


def main():
    Y, dates = S.load_Y()
    T = S.load_forecast_tensors(Y)
    d = data.load_all()
    clean = np.array([~flags(data.series(d, m)["MODAL"].astype(float).to_numpy(), data.series(d, m)["date"]) for m in MARKETS])
    named = {"stack_original_selection (B2 nnls lam 0.5)": cfg("B2", "nnls", 0.5), "stack_reselected (B2 nnls lam 0)": cfg("B2", "nnls", 0.0),
             "comb_original_selection (B2 mean)": cfg("B2", "mean"), "comb_reselected (B4 mean)": cfg("B4", "mean")}
    for i, c in enumerate(named.values()):
        c["id"] = i
    N = Y.shape[1]
    tT = np.arange(N - S.N_EVAL, N)
    rows = []
    for h in S.H8:
        Fh = S.build(T, h)
        o = tT - h + 1
        for name, c in named.items():
            p = S.stack_path(c, Fh, Y, h)
            for mi, mk in enumerate(MARKETS):
                a, f = Y[mi, tT], p[mi, o]
                for scope, m in (("all", np.ones(len(tT), bool)), ("unflagged", clean[mi, tT])):
                    if m.sum() >= 8:
                        rows.append({"model": name, "h": h, "market": mk, "scope": scope, "MAPE": np.mean(np.abs(a[m] - f[m]) / a[m]) * 100})
        for mi, mk in enumerate(MARKETS):
            a, f = Y[mi, tT], Y[mi, tT - h]
            for scope, m in (("all", np.ones(len(tT), bool)), ("unflagged", clean[mi, tT])):
                if m.sum() >= 8:
                    rows.append({"model": "Naive", "h": h, "market": mk, "scope": scope, "MAPE": np.mean(np.abs(a[m] - f[m]) / a[m]) * 100})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "tables" / "reselect_clean_test.csv", index=False)
    out = ["# Test performance of re-selected configurations (robustness; relative MAPE vs Naive, %, mean over markets)\n"]
    for scope in ("all", "unflagged"):
        p = R[R.scope == scope].pivot_table(index=["market", "h"], columns="model", values="MAPE")
        rel = (p.drop(columns="Naive").div(p["Naive"], axis=0) - 1) * 100
        out += [f"## {scope} test weeks\n", md(rel.groupby("h").mean().T.round(1).reset_index()) + "\n"]
    (RESULTS / "tables" / "RESELECT_CLEAN_TEST.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
