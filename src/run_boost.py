"""Heterogeneous residual boosting ("HBoost"): a genuine boosting counterpart of the original equal-weight
"Custom Boosting".

At each test origin t, using y[:t] only:
  stage 0: ARIMA (tuned order) -> in-sample one-step fitted values; residuals r0 = y - fitted
  stage 1: SVR (tuned params/lookback) fitted to r0 from price lags; r1 = r0 - NU * SVR(in-sample)
  stage 2: DT  (tuned params/lookback) fitted to r1 from price lags
  forecast = ARIMA_forecast + NU * SVR_pred + NU * DT_pred          (NU = 0.5, fixed, not tuned)
Deterministic given the data (ARIMA, SVR, DT have no random component), so only seed 0 is produced."""
import json
import sys

import numpy as np
import pandas as pd

import data
import models as M
from config import MARKETS, N_TEST, RESULTS

NU = 0.5


def hboost_forecast(ytr, order, svr_p, dt_p):
    from statsmodels.tsa.arima.model import ARIMA
    res = ARIMA(ytr, order=order).fit()
    f0 = float(res.forecast(1)[0])
    r = ytr - res.fittedvalues  # in-sample one-step residuals (training window only)
    pred = f0
    for name, p in (("SVR", svr_p), ("DT", dt_p)):
        L = p["lookback"]
        X, _ = M.windows(ytr, L)
        target = r[L:]
        mdl = M.FACTORIES[name](p, 0).fit(X, target)
        step = float(mdl.predict(ytr[-L:][None, :])[0])
        r = r.copy()
        r[L:] = r[L:] - NU * mdl.predict(X)
        pred += NU * step
    return pred


def run_market(market, d):
    s = data.series(d, market)
    y = s["MODAL"].astype(float).to_numpy()
    tune = json.loads((RESULTS / "tuning" / f"{market}.json").read_text())
    order = tuple(tune["ARIMA"]["order"])
    idx = np.arange(len(y) - N_TEST, len(y))
    out = []
    for t in idx:
        try:
            f = hboost_forecast(y[:t], order, tune["SVR"], tune["DT"])
            if not np.isfinite(f):
                f = y[t - 1]
        except Exception:
            f = y[t - 1]
        out.append(f)
    df = pd.DataFrame({"market": market, "model": "HBoost", "seed": 0, "idx": idx,
                       "date": s["date"].to_numpy()[idx], "actual": y[idx], "forecast": out})
    df.to_csv(RESULTS / "forecasts" / f"{market}_boost.csv", index=False)
    return df


if __name__ == "__main__":
    d = data.load_all()
    for m in (sys.argv[1:] or MARKETS):
        df = run_market(m, d)
        mape = (abs(df.actual - df.forecast) / df.actual).mean() * 100
        print(m, "HBoost MAPE", round(mape, 2), flush=True)
