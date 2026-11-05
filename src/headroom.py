"""How much can ANY combination of the base models gain over naive? Validation targets only
(indices N-104..N-53); the test set is not touched. Writes results/tables/headroom*.csv and HEADROOM.md.
  oracle_in  : best convex weights fitted ON the validation weeks themselves (cheating upper bound)
  crossfit   : weights fitted on one half of the validation weeks, evaluated on the other half (and vice versa)
  lomo       : pooled weights fitted on the other five markets' validation weeks (scaled), evaluated on this market"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import MARKETS, RESULTS
from util import md
from mh_stack import DET, N_EVAL, POOL, build_F, load_base

T = RESULTS / "tables"
HS = [1, 2, 3, 4]


def simplex_fit(X, y):
    k = X.shape[1]
    f = lambda w: np.mean((y - X @ w) ** 2)
    r = minimize(f, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 300})
    return r.x


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


def main():
    rows, corr_rows, data = [], [], {}
    for mk in MARKETS:
        df, y = load_base(mk)
        N = len(y)
        F = build_F(df, N)
        val_T = np.arange(N - 2 * N_EVAL, N - N_EVAL)
        scale = y[: N - 2 * N_EVAL].mean()
        for h in HS:
            X = F[h][val_T - h + 1]               # (52, 10) forecasts of y[T] made h weeks earlier
            a = y[val_T]
            last = y[val_T - h]
            data[(mk, h)] = (X / scale, a / scale, last / scale)
            E = a[:, None] - X
            en = a - last                          # naive error
            cm = np.corrcoef(E.T)
            offd = cm[np.triu_indices(len(POOL), 1)]
            # correlation of each model's error with the naive error (diversity relative to the anchor)
            cn = [np.corrcoef(E[:, j], en)[0, 1] for j in range(len(POOL))]
            corr_rows.append({"market": mk, "h": h, "mean_pairwise_err_corr": offd.mean(), "min_pairwise": offd.min(),
                              "mean_corr_with_naive_err": np.nanmean([c for c, p in zip(cn, POOL) if p != "Naive"])})
    for mk in MARKETS:
        for h in HS:
            X, a, last = data[(mk, h)]
            m_naive, m_best = mape(a, last), min(mape(a, X[:, j]) for j in range(len(POOL)))
            w = simplex_fit(X, a)
            m_in = mape(a, X @ w)
            half = len(a) // 2
            cf = np.empty_like(a)
            for tr, te in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
                cf[te] = X[te] @ simplex_fit(X[tr], a[tr])
            m_cf = mape(a, cf)
            Xp = np.vstack([data[(o, h)][0] for o in MARKETS if o != mk])
            ap = np.concatenate([data[(o, h)][1] for o in MARKETS if o != mk])
            wl = simplex_fit(Xp, ap)
            m_lo = mape(a, X @ wl)
            m_mean = mape(a, X.mean(1))
            rows.append({"market": mk, "h": h, "naive": m_naive, "best_single_in_hindsight": m_best, "equal_mean": m_mean,
                         "oracle_convex_in_sample": m_in, "crossfit_convex": m_cf, "pooled_lomo_convex": m_lo,
                         "gain_oracle_pct": 100 * (m_naive - m_in) / m_naive,
                         "gain_crossfit_pct": 100 * (m_naive - m_cf) / m_naive,
                         "gain_pooled_lomo_pct": 100 * (m_naive - m_lo) / m_naive,
                         "w_naive_oracle": w[POOL.index("Naive")]})
    R = pd.DataFrame(rows)
    C = pd.DataFrame(corr_rows)
    R.to_csv(T / "headroom.csv", index=False)
    C.to_csv(T / "headroom_error_correlation.csv", index=False)
    g = R.groupby("h")[["naive", "best_single_in_hindsight", "equal_mean", "oracle_convex_in_sample", "crossfit_convex",
                        "pooled_lomo_convex", "gain_oracle_pct", "gain_crossfit_pct", "gain_pooled_lomo_pct"]].mean()
    out = ["# Step 1: headroom analysis (validation weeks only; MAPE %, mean over six markets)\n",
           "oracle = weights fitted on the same weeks (upper bound, not attainable); crossfit = fit on half, test on the other half; "
           "pooled_lomo = one set of weights fitted on the other five markets. gain = relative MAPE reduction vs naive (%).\n",
           md(g.round(2)) + "\n", "## By market\n", md(R.round(2).set_index("market")) + "\n",
           "## Error correlation between the ten base models (validation)\n",
           md(C.round(3).set_index("market")) + "\n"]
    (T / "HEADROOM.md").write_text("\n".join(out), encoding="utf-8")
    print(md(g.round(2)))
    print(C.groupby("h")[["mean_pairwise_err_corr", "min_pairwise", "mean_corr_with_naive_err"]].mean().round(3))


if __name__ == "__main__":
    main()
