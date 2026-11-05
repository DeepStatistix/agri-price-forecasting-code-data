"""Small helper: markdown table writer used by the evaluation scripts."""
import pandas as pd

ORDER = ["Naive", "SeasonalNaive", "ETS", "ARIMA", "SVR", "RNN", "LSTM", "DT", "RF", "GB", "XGB",
         "Bagging_SVR", "Bagging_DT", "Comb_Mean", "Comb_Median", "Custom_Boosting", "HBoost",
         "HS_RF", "HS_NNLS", "HS_Ridge"]


def md(df, floatfmt=2):
    """Minimal markdown table writer (avoids the optional 'tabulate' dependency)."""
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df

    def f(v):
        if isinstance(v, float):
            return "" if v != v else f"{v:.{floatfmt}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return chr(10).join([head, sep] + body)
