"""LSTM and RNN rolling one-step forecasts.
Fixed architecture: one recurrent layer (32 units, tanh) + Dense(1), Adam, 60 epochs, batch 16, inputs/targets
standardised on the current training window. Lookback chosen once on pre-origin data (expanding-window CV).
Refit every 4 forecast origins; 10 seeds. Output: results/forecasts/<market>_deep.csv (same long format as base)."""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
from config import MARKETS, N_TEST, RESULTS, SEEDS, WARMUP

DEEP_LOOKBACKS = [8, 20]
REFIT_EVERY = 4
EPOCHS, UNITS, BATCH = 60, 32, 16


def _build(kind, L, seed):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.keras.utils.set_random_seed(seed)
    rec = layers.LSTM if kind == "LSTM" else layers.SimpleRNN
    m = models.Sequential([layers.Input((L, 1)), rec(UNITS, activation="tanh"), layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m


def _fit(kind, ytr, L, seed):
    mu, sd = ytr.mean(), ytr.std()
    z = (ytr - mu) / sd
    X = np.lib.stride_tricks.sliding_window_view(z, L)[:-1][:, :, None]
    m = _build(kind, L, seed)
    m.fit(X, z[L:], epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=False)
    return m, mu, sd


def _predict(m, mu, sd, hist, L):
    z = ((hist[-L:] - mu) / sd)[None, :, None]
    return float(m(z, training=False).numpy()[0, 0] * sd + mu)


def choose_lookback(kind, y_pre):
    """4 expanding-window folds of 12 steps on pre-origin data, seed 0."""
    best, sc = None, np.inf
    for L in DEEP_LOOKBACKS:
        errs = []
        n = len(y_pre)
        for k in range(4):
            cut = n - 12 * (4 - k)
            m, mu, sd = _fit(kind, y_pre[:cut], L, 0)
            pr = np.array([_predict(m, mu, sd, y_pre[:t], L) for t in range(cut, cut + 12)])
            errs.append(np.mean(np.abs((y_pre[cut:cut + 12] - pr) / y_pre[cut:cut + 12])) * 100)
        if np.mean(errs) < sc:
            best, sc = L, float(np.mean(errs))
    return best, sc


def rolling(kind, y, idx, L, seed):
    out, state = [], None
    for k, t in enumerate(idx):
        if state is None or k % REFIT_EVERY == 0:
            state = _fit(kind, y[:t], L, seed)
        out.append(_predict(*state, y[:t], L))
    return np.array(out)


def run_market(market, d, kinds=("LSTM", "RNN"), seeds=SEEDS, n_jobs=5):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    idx = np.arange(WARMUP, len(y))
    y_pre = y[: len(y) - N_TEST]
    rows, lbs = [], {}
    for kind in kinds:
        L, cv = choose_lookback(kind, y_pre)
        lbs[kind] = {"lookback": L, "cv_mape": cv}
        res = Parallel(n_jobs=n_jobs)(delayed(rolling)(kind, y, idx, L, sd) for sd in seeds)
        for sd, f in zip(seeds, res):
            rows.append(pd.DataFrame({"market": market, "model": kind, "seed": sd, "idx": idx,
                                      "date": s["date"].to_numpy()[idx], "actual": y[idx], "forecast": f}))
    out = RESULTS / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out / f"{market}_deep.csv", index=False)
    import json
    (RESULTS / "tuning" / f"{market}_deep.json").write_text(json.dumps(lbs, indent=2))


if __name__ == "__main__":
    d = data.load_all()
    args = sys.argv[1:]
    quick = "--quick" in args
    args = [a for a in args if a != "--quick"]
    for m in (args or MARKETS):
        run_market(m, d, seeds=SEEDS[:1] if quick else SEEDS)
        print("done", m, flush=True)
