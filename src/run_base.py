"""Rolling one-step forecasts for the non-deep models (naive, seasonal naive, ARIMA, ETS, SVR, DT, RF, GB, XGB,
bagging variants). Forecasts exist for every index t >= WARMUP so the same series serves as (i) inner
out-of-fold training data for the stacking meta-learner and (ii) the outer evaluation forecasts
(the last N_TEST indices). Output: results/forecasts/<market>_base.csv (long format)."""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
import models as M
from config import MARKETS, RESULTS, SEEDS, WARMUP

DETERMINISTIC = ["SVR", "DT"]
STOCHASTIC = ["RF", "GB", "XGB", "Bagging_SVR", "Bagging_DT"]


def one(market, y, idx, sn, name, seed, tune):
    if name == "Naive":
        f = M.naive(y, idx)
    elif name == "SeasonalNaive":
        f = M.seasonal_naive(y, idx, sn)
    elif name == "ARIMA":
        f = M.rolling_arima(y, tuple(tune["ARIMA"]["order"]), idx)
    elif name == "ETS":
        f = M.rolling_ets(y, idx)
    else:
        p = tune[name]
        f = M.rolling_tabular(y, name, p, p["lookback"], idx, seed=seed)
    return name, seed, f


def run_market(market, d, n_jobs=-1):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    idx = np.arange(WARMUP, len(y))
    sn = data.seasonal_naive_index(s)
    tune = json.loads((RESULTS / "tuning" / f"{market}.json").read_text())
    jobs = [(n, 0) for n in ["Naive", "SeasonalNaive", "ARIMA", "ETS"] + DETERMINISTIC]
    jobs += [(n, sd) for n in STOCHASTIC for sd in SEEDS]
    res = Parallel(n_jobs=n_jobs)(delayed(one)(market, y, idx, sn, n, sd, tune) for n, sd in jobs)
    rows = []
    for name, seed, f in res:
        rows.append(pd.DataFrame({"market": market, "model": name, "seed": seed, "idx": idx,
                                  "date": s["date"].to_numpy()[idx], "actual": y[idx], "forecast": f}))
    out = RESULTS / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out / f"{market}_base.csv", index=False)


if __name__ == "__main__":
    d = data.load_all()
    for m in (sys.argv[1:] or MARKETS):
        run_market(m, d)
        print("done", m, flush=True)
