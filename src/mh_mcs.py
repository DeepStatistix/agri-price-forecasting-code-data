"""Model Confidence Set (alpha=0.10; T_R and T_max, block bootstrap B=5000, block 4) at every horizon, test targets,
seed-averaged squared-error loss. Custom_Boosting (alias of Comb_Mean) excluded. Writes results/tables/mh_mcs.csv + MH_MCS.md"""
import numpy as np
import pandas as pd

from config import MARKETS, RESULTS
from evaluate import mcs
from util import md
from mh_stack import FDIR, H, N_EVAL, SUF, load_base

T = RESULTS / "tables"


def main():
    rows = []
    for mk in MARKETS:
        base, y = load_base(mk)
        N = len(y)
        fin = pd.read_csv(FDIR / f"{mk}_mh_final{SUF}.csv")
        df = pd.concat([base, fin], ignore_index=True)
        df = df[(df["target"] >= N - N_EVAL) & (df["model"] != "Custom_Boosting")]
        df["se"] = (df["actual"] - df["forecast"]) ** 2
        for h in H:
            L = df[df["h"] == h].groupby(["target", "model"])["se"].mean().unstack("model").sort_index()
            for stat, pv in mcs(L, alpha=0.10).items():
                for k, v in pv.items():
                    rows.append({"market": mk, "h": h, "statistic": stat, "model": k, "mcs_p": v, "in_set_10": v >= 0.10})
        print("done", mk, flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(T / f"mh_mcs{SUF}.csv", index=False)
    inn = M[M.in_set_10].groupby(["market", "h", "statistic"])["model"].nunique().unstack("statistic").reset_index()
    tot = M.groupby(["market", "h", "statistic"])["model"].nunique().iloc[0]
    top = M.sort_values("mcs_p", ascending=False).groupby(["market", "h", "statistic"]).head(1)
    top = top[top.statistic == "R"].pivot(index="market", columns="h", values="model").reindex(MARKETS)
    out = ["# Model Confidence Set, all horizons (alpha = 0.10; pool = %d models)\n" % tot,
           "## Number of models in the set (T_R, T_max)\n", md(inn.rename(columns={"R": "n_in_set_TR", "max": "n_in_set_Tmax"}).set_index("market")) + "\n",
           "## Rank-1 model under T_R\n", md(top) + "\n"]
    for stk in ("HS_tuned_mkt", "HS_tuned_glob", "Comb_tuned", "HS_RF_orig", "HS_NNLS_orig", "Naive", "ETS"):
        s = M[(M.model == stk) & (M.statistic == "R")].pivot(index="market", columns="h", values="in_set_10").reindex(MARKETS)
        out += [f"## Is {stk} in the 90% MCS (T_R)?\n", md(s.astype(str)) + "\n"]
    (T / f"MH_MCS{SUF}.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
