"""Data-quality audit: share of weekly values lying on exactly linear (constant-slope) or constant stretches, which indicate
interpolation or carry-forward in the source file. A point is 'suspect' if it belongs to a run of >= 4 consecutive points
with identical first differences (within each trading block; runs do not cross the Mar-Aug gap).
Writes results/tables/data_quality.csv and data_quality_flags.csv (per-week flags, used for the sensitivity analysis)."""
import numpy as np
import pandas as pd

import data
from config import MARKETS, N_TEST, RESULTS


def flags(y, dt, min_pts=4, tol=1.0):
    """Flag points on exactly linear or constant stretches (interpolation / carry-forward). A point is flagged when it lies in
    a run of >= min_pts consecutive weeks (inside one trading block) whose second differences are all within +-tol
    (prices are integers, so tol = 1 allows for rounding of an interpolated line)."""
    y = np.asarray(y, float)
    n = len(y)
    brk = np.r_[False, (pd.to_datetime(dt).diff().dt.days.to_numpy()[1:] > 60)]
    block = np.cumsum(brk)
    ok = np.zeros(n, bool)                       # interior point i is 'collinear' with its neighbours
    for i in range(1, n - 1):
        ok[i] = block[i - 1] == block[i] == block[i + 1] and abs(y[i + 1] - 2 * y[i] + y[i - 1]) <= tol
    f = np.zeros(n, bool)
    i = 1
    while i < n - 1:
        if ok[i]:
            j = i
            while j + 1 < n - 1 and ok[j + 1]:
                j += 1
            if j - i + 3 >= min_pts:             # interior points i..j  ->  points i-1..j+1
                f[i - 1:j + 2] = True
            i = j + 1
        else:
            i += 1
    return f


def main():
    d = data.load_all()
    rows, fl = [], []
    for mk in MARKETS:
        s = data.series(d, mk)
        y = s["MODAL"].astype(float).to_numpy()
        f = flags(y, s["date"])
        n = len(y)
        flat = np.r_[False, np.diff(y) == 0]
        win = {"all": slice(0, n), "validation": slice(n - 2 * N_TEST, n - N_TEST), "test": slice(n - N_TEST, n)}
        r = {"market": mk, "n_weeks": n}
        for k, sl in win.items():
            r[f"suspect_share_{k}_%"] = 100 * f[sl].mean()
        r["repeated_value_share_all_%"] = 100 * flat.mean()
        rows.append(r)
        fl.append(pd.DataFrame({"market": mk, "idx": np.arange(n), "date": s["date"], "suspect": f}))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tables" / "data_quality.csv", index=False)
    pd.concat(fl).to_csv(RESULTS / "tables" / "data_quality_flags.csv", index=False)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
