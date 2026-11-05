"""Shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Apple Data.csv"
RESULTS = ROOT / "results"

MARKET_NAMES = {"Binny Mill (F&V), Bangalore": "Binny Mill"}
MARKETS = ["Parimpore", "Solan", "Azadpur", "Binny Mill", "Mechua", "Sultanpur"]

N_TEST = 52                 # forecast origins = last 52 observations of each market
WARMUP = 104                # first index for which a base-model forecast is produced (inner OOF start)
LOOKBACKS = [4, 8, 12, 20]  # lookback candidates (20 = original paper)
SEEDS = list(range(10))
MIN_STACK_TRAIN = 30        # minimum OOF rows before a meta-learner is fitted (else mean of bases)

for _sub in ("forecasts", "tuning", "tables", "mh/forecasts", "mh/tuning", "v3/forecasts", "v3/tuning", "v31/forecasts",
             "v31/tuning", "case_study/forecasts"):
    (RESULTS / _sub).mkdir(parents=True, exist_ok=True)
