"""Residual diagnostics of one-week forecasts (test weeks): mean error, t-test, Ljung-Box autocorrelation (5 lags), for key models.
Interpretation is cautious: a non-significant test does not establish unbiasedness or absence of structure."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

import data
from config import MARKETS, RESULTS, ROOT
from v3_evaluate import load_market

KEYS = {"Naive": "Naive", "ETS": "ETS", "V3_comb": "Heterogeneous mean", "V3_stack": "Heterogeneous stacking (tuned)",
        "HS_RF_orig": "Heterogeneous stacking (original design)"}


def main():
    rows = []
    for mk in MARKETS:
        df = load_market(mk)
        df = df[(df.h == 1) & df.model.isin(KEYS)]
        g = df.groupby(["model", "target"]).agg(f=("forecast", "mean"), a=("actual", "first")).reset_index()
        for mo, q in g.groupby("model"):
            q = q.sort_values("target")
            e = (q.a - q.f).to_numpy()
            pe = e / q.a.to_numpy() * 100
            t = stats.ttest_1samp(e, 0.0)
            lb = acorr_ljungbox(e, lags=[5], return_df=True)
            rows.append({"market": mk, "model": KEYS[mo], "mean error (% of price)": pe.mean(), "bias t-test p": t.pvalue,
                         "Ljung-Box p (5 lags)": float(lb["lb_pvalue"].iloc[0]), "residual SD (Rs)": e.std(ddof=1)})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "tables" / "residual_diagnostics.csv", index=False)
    print(R.round(3).to_string(index=False))
    s = R.groupby("model").agg(bias_sig=("bias t-test p", lambda x: int((x < 0.05).sum())), lb_sig=("Ljung-Box p (5 lags)", lambda x: int((x < 0.05).sum())))
    print(s)


if __name__ == "__main__":
    main()
