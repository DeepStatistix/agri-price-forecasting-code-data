"""Sensitivity analysis: evaluate only on test targets that are NOT on interpolated/constant stretches (data_quality.flags).
Writes results/tables/sensitivity_clean_targets.csv and SENSITIVITY_CLEAN.md. Forecast files are unchanged; only the evaluated
targets are restricted (models still see the full history)."""
import numpy as np
import pandas as pd

import data
from config import MARKETS, RESULTS
from data_quality import flags
from util import md
from v3_evaluate import load_market

KEY = ["Naive", "ETS", "HS_RF_orig", "V3_comb", "V3_stack"]
T = RESULTS / "tables"


def main():
    d = data.load_all()
    rows = []
    for mk in MARKETS:
        s = data.series(d, mk)
        f = flags(s["MODAL"].astype(float).to_numpy(), s["date"])
        df = load_market(mk)
        df = df[df["model"].isin(KEY)].copy()
        df["clean"] = ~f[df["target"].to_numpy()]
        for h in (1, 2, 4, 8):
            sub = df[df["h"] == h]
            for scope, q in (("all targets", sub), ("unflagged targets", sub[sub["clean"]])):
                if q["target"].nunique() < 8:
                    continue
                ape = (q["actual"] - q["forecast"]).abs() / q["actual"] * 100
                for m, g in ape.groupby(q["model"]):
                    rows.append({"market": mk, "h": h, "scope": scope, "model": m, "MAPE": g.mean(), "n_targets": q["target"].nunique()})
    R = pd.DataFrame(rows)
    R.to_csv(T / "sensitivity_clean_targets.csv", index=False)
    out = ["# Sensitivity: evaluation restricted to unflagged test targets\n"]
    n = R[(R.h == 1) & (R.model == "Naive")].pivot(index="market", columns="scope", values="n_targets")
    out += ["## Number of test targets evaluated (h = 1)\n", md(n.reindex(MARKETS)) + "\n"]
    for scope in ("all targets", "unflagged targets"):
        q = R[R.scope == scope]
        p = q.pivot_table(index=["market", "h"], columns="model", values="MAPE")
        rel = (p[KEY[1:]].div(p["Naive"], axis=0) - 1) * 100
        agg = rel.groupby("h").mean().T.round(1)
        out += [f"## Relative MAPE vs Naive (%), mean over markets with >= 8 evaluated targets: {scope}\n", md(agg.reset_index().rename(columns={"model": "model"})) + "\n"]
        full = p.groupby("h").mean().T.round(2)
        out += [f"## Mean MAPE (%): {scope}\n", md(full.reset_index()) + "\n"]
    cl = R[(R.scope == "unflagged targets") & R.market.isin(["Parimpore", "Binny Mill", "Sultanpur"])]
    p = cl.pivot_table(index=["market", "h"], columns="model", values="MAPE")
    rel = (p[KEY[1:]].div(p["Naive"], axis=0) - 1) * 100
    out += ["## Three markets with clean test windows only (Parimpore, Binny Mill, Sultanpur): relative MAPE vs Naive (%)\n",
            md(rel.groupby("h").mean().T.round(1).reset_index()) + "\n"]
    (T / "SENSITIVITY_CLEAN.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
