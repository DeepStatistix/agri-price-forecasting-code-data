"""Pooled Diebold-Mariano comparisons on all test weeks and on unflagged test weeks only (data_quality.flags).
For each market and week the squared-error loss is scaled by the squared mean price; the weekly loss is averaged over the markets
that have the week available (unflagged weeks only in the second scope). Holm adjustment within scope, horizon and reference
across the heterogeneous mean and the tuned stacking model; stacker-versus-mean comparisons are adjusted across the four pools.
Writes results/tables/pooled_dm_unflagged.csv"""
import numpy as np
import pandas as pd

import data
from config import MARKETS, RESULTS
from data_quality import flags
from evaluate import holm
from mh_evaluate import dm_hln_h
from v31_evaluate import load

POOL = [("V31_stack", "V31_comb"), ("ABL_B3_stack", "ABL_B3_mean"), ("ABL_B5_stack", "ABL_B5_mean"), ("ABL_B1_stack", "ABL_B1_mean")]
HET = ["V31_comb", "V31_stack"]
REFS = ["Naive", "ETS"]


def main():
    d = data.load_all()
    ok, sc, L = {}, {}, {}
    for mk in MARKETS:
        s = data.series(d, mk)
        ok[mk] = ~flags(s["MODAL"].astype(float).to_numpy(), s["date"])
        sc[mk] = float(s["MODAL"].astype(float).mean()) ** 2
        df = load(mk)
        df = df[df.model.isin(set(HET + REFS + [m for p in POOL for m in p]))].copy()
        df["se"] = (df.actual - df.forecast) ** 2 / sc[mk]
        df["clean"] = ok[mk][df.target.to_numpy()]
        L[mk] = df
    rows = []
    for scope in ("all", "unflagged"):
        for h in range(1, 9):
            loss = {}
            for mk in MARKETS:
                q = L[mk][L[mk].h == h]
                if scope == "unflagged":
                    q = q[q.clean]
                loss[mk] = q.groupby(["target", "model"])["se"].mean().unstack("model")

            def weekly(m):
                return pd.DataFrame({mk: loss[mk][m] for mk in MARKETS}).mean(axis=1).dropna()

            for ref in REFS:
                fam = []
                for m in HET:
                    a, b = weekly(m), weekly(ref)
                    idx = a.index.intersection(b.index)
                    st, p = dm_hln_h(b[idx], a[idx], h)
                    fam.append((m, ref, 100 * (a[idx].mean() - b[idx].mean()) / b[idx].mean(), p))
                for (m, r, ch, p), pa in zip(fam, holm([f[3] for f in fam])):
                    rows.append({"scope": scope, "h": h, "model": m, "reference": r, "loss_change_pct": ch, "p_raw": p, "p_holm": pa})
            fam = []
            for a_m, b_m in POOL:
                a, b = weekly(a_m), weekly(b_m)
                idx = a.index.intersection(b.index)
                st, p = dm_hln_h(b[idx], a[idx], h)
                fam.append((a_m, b_m, 100 * (a[idx].mean() - b[idx].mean()) / b[idx].mean(), p))
            for (m, r, ch, p), pa in zip(fam, holm([f[3] for f in fam])):
                rows.append({"scope": scope, "h": h, "model": m, "reference": r, "loss_change_pct": ch, "p_raw": p, "p_holm": pa})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "tables" / "pooled_dm_unflagged.csv", index=False)
    for scope in ("all", "unflagged"):
        for ref in ("Naive", "ETS"):
            q = R[(R.scope == scope) & (R.reference == ref)]
            print(f"\n{scope}, reference {ref}: loss change % (Holm p)")
            print(q.assign(v=lambda x: x.loss_change_pct.round(1).astype(str) + " (" + x.p_holm.round(3).astype(str) + ")")
                  .pivot(index="model", columns="h", values="v").to_string())


if __name__ == "__main__":
    main()
