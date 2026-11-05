"""Bagged-LSTM for the one-week horizon.
At every 4th test origin, 10 LSTMs are trained on bootstrap resamples of the training windows (lookback = the market's
tuned LSTM lookback) and averaged. 3 seeds (different bootstrap draws and initialisations). Test origins only
(last 52); every forecast for index t uses y[:t]. Output: results/forecasts/<market>_baglstm.csv"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
from config import MARKETS, N_TEST, RESULTS
from run_deep import BATCH, EPOCHS, _build

N_BAG, REFIT, SEEDS = 10, 4, [0, 1, 2]


def fit_bag(ytr, L, seed):
    mu, sd = ytr.mean(), ytr.std()
    z = (ytr - mu) / sd
    X = np.lib.stride_tricks.sliding_window_view(z, L)[:-1][:, :, None]
    Y = z[L:]
    members = []
    for b in range(N_BAG):
        rng = np.random.default_rng(seed * 1000 + b)
        i = rng.integers(0, len(X), len(X))
        m = _build("LSTM", L, seed * 1000 + b)
        m.fit(X[i], Y[i], epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=False)
        members.append(m)
    return members, mu, sd


def predict(state, hist, L):
    members, mu, sd = state
    z = ((hist[-L:] - mu) / sd)[None, :, None]
    return float(np.mean([m(z, training=False).numpy()[0, 0] for m in members]) * sd + mu)


def run(market, seed):
    d = data.load_all()
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    L = json.loads((RESULTS / "tuning" / f"{market}_deep.json").read_text())["LSTM"]["lookback"]
    idx = np.arange(len(y) - N_TEST, len(y))
    out, st = [], None
    for k, t in enumerate(idx):
        if st is None or k % REFIT == 0:
            st = fit_bag(y[:t], L, seed)
        out.append(predict(st, y[:t], L))
    return pd.DataFrame({"market": market, "model": "Bagging_LSTM", "seed": seed, "idx": idx,
                         "date": s["date"].to_numpy()[idx], "actual": y[idx], "forecast": out})


if __name__ == "__main__":
    for m in (sys.argv[1:] or MARKETS):
        df = pd.concat(Parallel(n_jobs=3)(delayed(run)(m, sd) for sd in SEEDS))
        df.to_csv(RESULTS / "forecasts" / f"{m}_baglstm.csv", index=False)
        print("done", m, round(float((abs(df.actual - df.forecast) / df.actual).mean() * 100), 2), flush=True)
