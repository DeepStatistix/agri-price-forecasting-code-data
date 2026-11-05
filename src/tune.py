"""Hyperparameter selection on pre-origin data only (first n - N_TEST rows), expanding-window CV.
Writes results/tuning/<market>.json and the full grid log."""
import itertools
import json
import sys

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import data
from config import LOOKBACKS, MARKETS, N_TEST, RESULTS
from models import BAG_BASE, FACTORIES, GRIDS, select_arima_order, windows


def tune_market(market, d):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()[: len(s) - N_TEST]  # tuning window: pre-origin only
    tscv = TimeSeriesSplit(n_splits=4, test_size=12)
    best, log = {}, []
    for name, grid in GRIDS.items():
        keys = list(grid)
        top = (np.inf, None)
        for L in LOOKBACKS:
            X, Y = windows(y, L)
            for vals in itertools.product(*[grid[k] for k in keys]):
                p = dict(zip(keys, vals))
                errs = []
                for tr, te in tscv.split(X):
                    m = FACTORIES[name](p, 0).fit(X[tr], Y[tr])
                    errs.append(np.mean(np.abs((Y[te] - m.predict(X[te])) / Y[te])) * 100)
                score = float(np.mean(errs))
                log.append({"model": name, "lookback": L, **p, "cv_mape": score})
                if score < top[0]:
                    top = (score, {"lookback": L, **p})
        best[name] = top[1]
        best[name]["cv_mape"] = top[0]
    for b, base in BAG_BASE.items():
        best[b] = dict(best[base])
    best["ARIMA"] = {"order": list(select_arima_order(y))}
    return best, log


if __name__ == "__main__":
    d = data.load_all()
    out = RESULTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    for m in (sys.argv[1:] or MARKETS):
        best, log = tune_market(m, d)
        (out / f"{m}.json").write_text(json.dumps(best, indent=2))
        import pandas as pd
        pd.DataFrame(log).to_csv(out / f"{m}_grid_log.csv", index=False)
        print(m, {k: v.get("lookback", v) for k, v in best.items()}, flush=True)
