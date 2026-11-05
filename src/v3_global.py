"""Pooled ("global") base models G_Ridge, G_RF, G_XGB, G_MLP for horizons 1..8.
One model per horizon fitted on the six markets together; every input at row origin j uses information up to j-1 only.
Output: results/v3/forecasts/global.csv (market, model, seed, h, origin, target, date, actual, forecast)."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import data
from config import MARKETS, RESULTS, WARMUP

H8 = list(range(1, 9))
REFIT = 4
FIRST = 10          # first usable row origin (needs 9 past weeks)
SEEDS = list(range(5))
OUT = RESULTS / "v3" / "forecasts"


def load_panel():
    d = data.load_all()
    S = [data.series(d, m) for m in MARKETS]
    assert all((s["date"].to_numpy() == S[0]["date"].to_numpy()).all() for s in S)
    Y = np.array([s["MODAL"].astype(float).to_numpy() for s in S])
    A = np.log1p(np.clip(np.array([s["Arrival"].astype(float).to_numpy() for s in S]), 0, None))
    return Y, A, S[0]["season_week"].to_numpy(), S[0]["season"].to_numpy(), S[0]["date"].to_numpy()


def feature_block(Y, A, sw, season, j, h):
    """(6, n_features) for origin j and horizon h: information up to index j-1; calendar facts for target j+h-1."""
    M = Y.shape[0]
    ly = np.log(Y)
    r = {k: ly[:, j - k] - ly[:, j - k - 1] for k in (1, 2, 3, 4)}
    oth = {k: (r[k].sum() - r[k]) / (M - 1) for k in (1, 2)}
    da = A[:, j - 1] - A[:, j - 2]
    ra = A[:, j - 1] - A[:, j - 9:j - 1].mean(1)
    dev = ly[:, j - 1] - np.log(Y[:, j - 9:j - 1].mean(1))
    T = j + h - 1
    cal = np.column_stack([np.full(M, sw[T], float), np.full(M, float(season[T] != season[j - 1]))])
    return np.column_stack([r[1], r[2], r[3], r[4], oth[1], oth[2], da, ra, dev, cal, np.eye(M)])


def make_model(name, seed):
    if name == "G_Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "G_RF":
        return RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=seed, n_jobs=1)
    if name == "G_XGB":
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed, n_jobs=1, verbosity=0)
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=300, random_state=seed))
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


def run_one(name, seed):
    Y, A, sw, season, dates = load_panel()
    M, N = Y.shape
    ly = np.log(Y)
    rows = []
    for h in H8:
        mdl = None
        for o in range(WARMUP, N - h + 1):
            if mdl is None or (o - WARMUP) % REFIT == 0:
                Xs, ys = [], []
                for j in range(FIRST, o - h + 1):               # target j+h-1 <= o-1: already observed at o
                    Xs.append(feature_block(Y, A, sw, season, j, h))
                    ys.append(ly[:, j + h - 1] - ly[:, j - 1])
                mdl = make_model(name, seed).fit(np.vstack(Xs), np.concatenate(ys))
            pred = mdl.predict(feature_block(Y, A, sw, season, o, h))
            T = o + h - 1
            for m in range(M):
                rows.append((MARKETS[m], name, seed, h, o, T, dates[T], Y[m, T], float(Y[m, o - 1] * np.exp(pred[m]))))
    return pd.DataFrame(rows, columns=["market", "model", "seed", "h", "origin", "target", "date", "actual", "forecast"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("G_Ridge", 0)] + [(n, s) for n in ("G_RF", "G_XGB", "G_MLP") for s in SEEDS]
    if "--quick" in sys.argv:
        jobs = [("G_Ridge", 0), ("G_XGB", 0)]
    res = Parallel(n_jobs=8)(delayed(run_one)(n, s) for n, s in jobs)
    df = pd.concat(res, ignore_index=True)
    df.to_csv(OUT / ("global_quick.csv" if "--quick" in sys.argv else "global.csv"), index=False)
    print("done", len(df))
