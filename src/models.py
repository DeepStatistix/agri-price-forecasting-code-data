"""One-step-ahead rolling forecasters. Every forecast for index t uses y[:t] only."""
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def windows(y, L):
    X = np.lib.stride_tricks.sliding_window_view(y, L)[:-1]
    return X, y[L:]


# ---------- factories for lag-based regressors (params are tuned in tune.py) ----------
def make_svr(p, seed=0):
    base = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma="scale"))
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


def make_dt(p, seed=0):
    return DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=seed)


def make_rf(p, seed=0):
    return RandomForestRegressor(n_estimators=100, max_depth=p["max_depth"],
                                 min_samples_leaf=p["min_samples_leaf"], random_state=seed, n_jobs=1)


def make_gb(p, seed=0):
    return GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                     max_depth=p["max_depth"], random_state=seed)


def make_xgb(p, seed=0):
    return XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                        max_depth=p["max_depth"], random_state=seed, n_jobs=1, verbosity=0)


def make_bag_svr(p, seed=0):
    return BaggingRegressor(estimator=make_svr(p), n_estimators=30, random_state=seed, n_jobs=1)


def make_bag_dt(p, seed=0):
    return BaggingRegressor(estimator=make_dt(p, seed), n_estimators=30, random_state=seed, n_jobs=1)


FACTORIES = {"SVR": make_svr, "DT": make_dt, "RF": make_rf, "GB": make_gb, "XGB": make_xgb,
             "Bagging_SVR": make_bag_svr, "Bagging_DT": make_bag_dt}

# search spaces (complete grids are written to results/tuning/ and the supplement)
GRIDS = {
    "SVR": {"C": [1, 10, 100], "epsilon": [0.01, 0.03, 0.1]},
    "DT": {"max_depth": [3, 5, 8], "min_samples_leaf": [2, 4, 6]},
    "RF": {"max_depth": [None, 5], "min_samples_leaf": [2, 4]},
    "GB": {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
    "XGB": {"n_estimators": [200], "learning_rate": [0.03, 0.1], "max_depth": [2, 4]},
}
# bagging variants reuse the tuned parameters of their base learner
BAG_BASE = {"Bagging_SVR": "SVR", "Bagging_DT": "DT"}


def rolling_tabular(y, name, params, L, idx, seed=0, refit_every=1):
    """Forecast y[t] for t in idx with a lag-L regressor refit on y[:t] (expanding window)."""
    fac = FACTORIES[name]
    out, mdl = [], None
    for k, t in enumerate(idx):
        if mdl is None or k % refit_every == 0:
            X, Y = windows(y[:t], L)
            mdl = fac(params, seed).fit(X, Y)
        out.append(float(mdl.predict(y[t - L:t][None, :])[0]))
    return np.array(out)


def rolling_arima(y, order, idx):
    from statsmodels.tsa.arima.model import ARIMA
    out = []
    for t in idx:
        try:
            f = ARIMA(y[:t], order=order).fit().forecast(1)[0]
            if not np.isfinite(f):
                f = y[t - 1]
        except Exception:
            f = y[t - 1]
        out.append(float(f))
    return np.array(out)


def rolling_ets(y, idx):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    out = []
    for t in idx:
        try:
            f = ExponentialSmoothing(y[:t], trend="add", damped_trend=True).fit().forecast(1)[0]
            if not np.isfinite(f):
                f = y[t - 1]
        except Exception:
            f = y[t - 1]
        out.append(float(f))
    return np.array(out)


def naive(y, idx):
    return np.array([y[t - 1] for t in idx], dtype=float)


def seasonal_naive(y, idx, sn_index):
    return np.array([y[sn_index[t]] if sn_index[t] >= 0 else y[t - 1] for t in idx], dtype=float)


def select_arima_order(y_train):
    from statsmodels.tsa.arima.model import ARIMA
    best, best_aic = (1, 1, 1), np.inf
    for p in range(4):
        for d in (0, 1):
            for q in range(3):
                try:
                    a = ARIMA(y_train, order=(p, d, q)).fit().aic
                except Exception:
                    continue
                if np.isfinite(a) and a < best_aic:
                    best, best_aic = (p, d, q), a
    return best
