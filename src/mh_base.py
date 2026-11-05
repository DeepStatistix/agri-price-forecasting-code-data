"""Multi-horizon (h = 1..4) base forecasts. Every forecast for target index o+h-1 is made from y[:o].
Output: results/mh/forecasts/<market>_mhbase.csv  (long: market, model, seed, h, origin, target, date, actual, forecast)
Usage: python mh_base.py classical|tabular|deep [market ...]"""
import json
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
import models as M
from config import MARKETS, RESULTS, WARMUP

H = [int(x) for x in os.environ.get("MH_H", "1,2,3,4").split(",")]
HMAX = max(H)
SUF = os.environ.get("MH_SUF", "")
SEEDS_MH = list(range(5))
OUT = RESULTS / "mh" / "forecasts"


def _frame(market, model, seed, y, dates, fc):
    """fc: dict h -> dict origin -> forecast"""
    rows = []
    for h in H:
        for o, f in fc[h].items():
            T = o + h - 1
            rows.append((market, model, seed, h, o, T, dates[T], y[T], f))
    return pd.DataFrame(rows, columns=["market", "model", "seed", "h", "origin", "target", "date", "actual", "forecast"])


def _origins(N, h):
    return range(WARMUP, N - h + 1)


def run_classical(market, d):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    dates = s["date"].to_numpy()
    N = len(y)
    sn = data.seasonal_naive_index(s)
    order = tuple(json.loads((RESULTS / "tuning" / f"{market}.json").read_text())["ARIMA"]["order"])
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fc = {k: {h: {} for h in H} for k in ("Naive", "SeasonalNaive", "ETS", "ARIMA")}
    for o in range(WARMUP, N):
        hs = [h for h in H if o + h - 1 <= N - 1]
        for k in ("ETS", "ARIMA"):
            try:
                if k == "ARIMA":
                    f = np.asarray(ARIMA(y[:o], order=order).fit().forecast(HMAX))
                else:
                    f = np.asarray(ExponentialSmoothing(y[:o], trend="add", damped_trend=True).fit().forecast(HMAX))
                if not np.all(np.isfinite(f)):
                    f = np.full(HMAX, y[o - 1])
            except Exception:
                f = np.full(HMAX, y[o - 1])
            for h in hs:
                fc[k][h][o] = float(f[h - 1])
        for h in hs:
            T = o + h - 1
            fc["Naive"][h][o] = y[o - 1]
            fc["SeasonalNaive"][h][o] = y[sn[T]] if 0 <= sn[T] < o else y[o - 1]
    return pd.concat([_frame(market, k, 0, y, dates, fc[k]) for k in fc], ignore_index=True)


def _direct_windows(y, L, h):
    W = np.lib.stride_tricks.sliding_window_view(y, L)
    return W[: len(W) - h], y[L + h - 1:]


def run_tabular_one(market, name, seed, refit_every):
    d = data.load_all()
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    dates = s["date"].to_numpy()
    N = len(y)
    p = json.loads((RESULTS / "tuning" / f"{market}.json").read_text())[name]
    L = p["lookback"]
    fc = {h: {} for h in H}
    mdl = {}
    for h in H:
        for k, o in enumerate(_origins(N, h)):
            if h not in mdl or k % refit_every == 0:
                X, Y = _direct_windows(y[:o], L, h)
                mdl[h] = M.FACTORIES[name](p, seed).fit(X, Y)
            fc[h][o] = float(mdl[h].predict(y[o - L:o][None, :])[0])
    return _frame(market, name, seed, y, dates, fc)


# ---------------- deep: one network, 4 outputs ----------------
EPOCHS, UNITS, BATCH, REFIT_DEEP = 60, 32, 16, 4


def _deep_fit(kind, ytr, L, seed):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    tf.keras.utils.set_random_seed(seed)
    mu, sd = ytr.mean(), ytr.std()
    z = (ytr - mu) / sd
    W = np.lib.stride_tricks.sliding_window_view(z, L)
    n = len(W) - HMAX
    X, Y = W[:n][:, :, None], np.stack([z[L + j: L + j + HMAX] for j in range(n)])
    rec = layers.LSTM if kind == "LSTM" else layers.SimpleRNN
    m = models.Sequential([layers.Input((L, 1)), rec(UNITS, activation="tanh"), layers.Dense(HMAX)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(X, Y, epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=False)
    return m, mu, sd


def run_deep_one(market, kind, seed):
    d = data.load_all()
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    dates = s["date"].to_numpy()
    N = len(y)
    L = json.loads((RESULTS / "tuning" / f"{market}_deep.json").read_text())[kind]["lookback"]
    fc = {h: {} for h in H}
    st = None
    for k, o in enumerate(range(WARMUP, N)):
        if st is None or k % REFIT_DEEP == 0:
            st = _deep_fit(kind, y[:o], L, seed)
        m, mu, sd = st
        out = m(((y[o - L:o] - mu) / sd)[None, :, None], training=False).numpy()[0] * sd + mu
        for h in H:
            if o + h - 1 <= N - 1:
                fc[h][o] = float(out[h - 1])
    return _frame(market, kind, seed, y, dates, fc)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mode, mk = sys.argv[1], (sys.argv[2:] or MARKETS)
    d = data.load_all()
    for m in mk:
        if mode == "classical":
            df = run_classical(m, d)
        elif mode == "tabular":
            jobs = [("SVR", 0, 1), ("DT", 0, 1)] + [(n, sd, 2) for n in ("RF", "GB", "XGB") for sd in SEEDS_MH]
            df = pd.concat(Parallel(n_jobs=8)(delayed(run_tabular_one)(m, n, sd, r) for n, sd, r in jobs))
        elif mode == "deep":
            jobs = [(k, sd) for k in ("LSTM", "RNN") for sd in SEEDS_MH]
            df = pd.concat(Parallel(n_jobs=3)(delayed(run_deep_one)(m, k, sd) for k, sd in jobs))
        df.to_csv(OUT / f"{m}_mh_{mode}{SUF}.csv", index=False)
        print(mode, "done", m, len(df), flush=True)
