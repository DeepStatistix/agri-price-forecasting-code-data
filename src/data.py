"""Data loading, season indexing and the objective volatility table."""
import numpy as np
import pandas as pd

from config import DATA, MARKET_NAMES, MARKETS, N_TEST


def load_all() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["date"] = pd.to_datetime(d["date_arrival"], format="%d-%m-%Y")
    d["market"] = d["market_name"].replace(MARKET_NAMES)
    d = d.sort_values(["market", "date"]).reset_index(drop=True)
    # trading season runs Sep -> Feb of the next year
    d["season"] = np.where(d["date"].dt.month >= 9, d["date"].dt.year, d["date"].dt.year - 1)
    d["season_week"] = d.groupby(["market", "season"]).cumcount()
    return d


def series(d: pd.DataFrame, market: str) -> pd.DataFrame:
    return d[d["market"] == market].reset_index(drop=True)


def seasonal_naive_index(s: pd.DataFrame) -> np.ndarray:
    """For each row, index of the same season-week in the previous season (-1 if absent)."""
    out = np.full(len(s), -1)
    key = {(r.season, r.season_week): i for i, r in enumerate(s.itertuples())}
    for i, r in enumerate(s.itertuples()):
        out[i] = key.get((r.season - 1, r.season_week), -1)
    return out


def volatility_table(d: pd.DataFrame) -> pd.DataFrame:
    """Volatility from the pre-origin (training) data only; terciles over the six markets."""
    rows = []
    for m in MARKETS:
        s = series(d, m)
        tr = s.iloc[: len(s) - N_TEST]
        p = tr["MODAL"].astype(float).to_numpy()
        lr = np.diff(np.log(p))
        # within-season log returns only (do not difference across the Mar-Aug gap)
        same = (tr["season"].to_numpy()[1:] == tr["season"].to_numpy()[:-1])
        rows.append({
            "market": m, "n_total": len(s), "n_train": len(tr), "n_test": N_TEST,
            "train_start": tr["date"].min().date(), "train_end": tr["date"].max().date(),
            "test_start": s["date"].iloc[-N_TEST].date(), "test_end": s["date"].iloc[-1].date(),
            "mean_price": p.mean(), "cv": p.std(ddof=1) / p.mean(),
            "logret_sd": lr[same].std(ddof=1),
        })
    v = pd.DataFrame(rows)
    rk = v["cv"].rank() + v["logret_sd"].rank()
    v["vol_rank_score"] = rk
    order = rk.rank(method="first")
    v["regime"] = pd.cut(order, [0, 2, 4, 6], labels=["Low", "Medium", "High"])
    return v
