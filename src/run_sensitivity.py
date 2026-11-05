"""Sensitivity: lag-based models whose input windows never cross the Mar-Aug off-season gap.
Fixed lookback L=4 (the shortest tuned lookback, so most origins are covered); windows must lie inside one trading
season. For the first L weeks of a season (not enough in-season history) the forecast falls back to the last observed
price. Tuned hyperparameters are reused from the primary run. Output: results/forecasts/<market>_sens.csv (seed 0)."""
import json
import sys

import numpy as np
import pandas as pd

import data
import models as M
from config import MARKETS, N_TEST, RESULTS

L = 4
NAMES = ["SVR", "DT", "RF", "GB", "XGB"]


def within_season_windows(y, season, t, L):
    X, Y = [], []
    for i in range(L, t):
        if season[i - L] == season[i]:  # window y[i-L:i] and target y[i] all in one season (seasons are contiguous)
            X.append(y[i - L:i]); Y.append(y[i])
    return np.array(X), np.array(Y)


def run_market(market, d):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    season = s["season"].to_numpy()
    tune = json.loads((RESULTS / "tuning" / f"{market}.json").read_text())
    idx = np.arange(len(y) - N_TEST, len(y))
    rows = []
    for name in NAMES:
        p = tune[name]
        fc = []
        for t in idx:
            if season[t - L] != season[t]:
                fc.append(y[t - 1]); continue
            X, Y = within_season_windows(y, season, t, L)
            fc.append(float(M.FACTORIES[name](p, 0).fit(X, Y).predict(y[t - L:t][None])[0]))
        rows.append(pd.DataFrame({"market": market, "model": name + "_ws", "seed": 0, "idx": idx,
                                  "date": s["date"].to_numpy()[idx], "actual": y[idx], "forecast": fc}))
    pd.concat(rows).to_csv(RESULTS / "forecasts" / f"{market}_sens.csv", index=False)


if __name__ == "__main__":
    d = data.load_all()
    for m in (sys.argv[1:] or MARKETS):
        run_market(m, d)
        print("done", m, flush=True)
