"""Case study data: weekly onion panel (Azadpur + five supporting markets) built from the daily FAQ price files in data/case_study/daily.
Writes data/case_study/onion_weekly_panel.csv and data/case_study/SOURCES.txt (SHA-256 of every source file)."""
import hashlib
import os
from pathlib import Path

import pandas as pd

from config import ROOT

SRC = Path(os.environ.get("ONION_DAILY_DIR", str(ROOT / "data" / "case_study" / "daily")))
PANEL = {"Azadpur": "Azadpur", "Ahmedabad": "Ahmedabad_Chimanbhai_Patal_Market_Vasana", "Kangra": "Kangra",
         "Narwal_Jammu": "Narwal_Jammu", "Kathua": "Kathua", "Panipat": "Panipat"}
OUT = ROOT / "data" / "case_study"


def build():
    cols, lines = {}, []
    for name, folder in PANEL.items():
        f = SRC / folder / "Onion_FAQ_dataset.csv"
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {folder}/Onion_FAQ_dataset.csv")
        d = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date")
        assert (d["Mask"] >= 0).all() and d["Mask"].mean() >= 0.99
        cols[name] = d.set_index("Date")["Avg Price (per kg)"].resample("W").mean()
    w = pd.DataFrame(cols)
    start = max(c.first_valid_index() for c in cols.values())
    end = min(c.last_valid_index() for c in cols.values())
    w = w.loc[start:end]
    assert w.notna().all().all(), "missing weeks in the common window"
    OUT.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT / "onion_weekly_panel.csv")
    (OUT / "SOURCES.txt").write_text("\n".join(lines) + f"\nweekly window {w.index[0].date()} .. {w.index[-1].date()} ({len(w)} weeks)\n")
    return w


def load():
    w = pd.read_csv(OUT / "onion_weekly_panel.csv", index_col=0, parse_dates=True)
    return w


if __name__ == "__main__":
    w = build()
    print(w.shape, w.index[0].date(), w.index[-1].date())
    print(w.describe().round(1).T)
