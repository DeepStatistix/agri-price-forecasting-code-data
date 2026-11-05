# Code and data

Code and data accompanying the paper *Heterogeneous ensembles for agricultural price forecasting: a multi-horizon evaluation across markets with different volatility*.

## Contents

* `data/raw/Apple Data.csv` (checksum in `data/CHECKSUMS.txt`)
* `data/case_study/daily/`: daily onion prices of the six markets used in the case study
* `data/case_study/onion_weekly_panel.csv`: weekly panel built from them by `src/cs_data.py` (checksums in `data/case_study/SOURCES.txt`)
* `src/`: Python code
* `requirements.txt`: package versions (Python 3.10)

## Usage

Install the requirements and run the scripts from `src/`; outputs are written to `results/`.

```
python tune.py ; python run_base.py ; python run_deep.py ; python run_ensembles.py ; python run_boost.py
python run_sensitivity.py ; python evaluate.py ; python robustness.py ; python run_bagged_lstm.py
python mh_base.py classical ; python mh_base.py tabular ; python mh_base.py deep
MH_H=5,6,7,8 MH_SUF=_ext python mh_base.py classical          # likewise tabular and deep
python mh_stack.py tune ; python mh_stack.py final ; python mh_evaluate.py ; python mh_mcs.py
python v3_global.py ; python v3_stack.py tune ; python v3_stack.py final ; python v3_evaluate.py
python v31_stack.py tune ; python v31_stack.py final ; python v31_evaluate.py
python data_quality.py ; python pooled_dm_unflagged.py ; python sensitivity_clean.py ; python reselect_clean.py ; python reselect_clean_test.py
python headroom.py ; python headroom14.py ; python residual_diagnostics.py
python case_study.py classical ; python case_study.py global ; python case_study.py final
```

## Licence

Code: MIT (`LICENSE`).
