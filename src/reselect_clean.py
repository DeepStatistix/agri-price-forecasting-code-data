"""Robustness check: repeat the validation-only selection of the combiner using ONLY unflagged validation targets
(data_quality.flags). Same 550 configurations, same score and parsimony rule. If the same configuration is selected the
design does not depend on interpolated weeks. Writes results/v31/tuning/v31_reselect_clean.csv and RESELECT_CLEAN.md"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import data
import v3_stack as S
import v31_stack as V
from config import MARKETS, RESULTS
from data_quality import flags
from util import md


def val_masked(h, cfgs, Y, T, clean):
    N = Y.shape[1]
    Fh = S.build(T, h)
    vT = np.arange(N - 2 * S.N_EVAL, N - S.N_EVAL)
    o = vT - h + 1
    res = np.full((len(cfgs), S.M), np.nan)
    for ci, c in enumerate(cfgs):
        p = S.stack_path(c, Fh, Y, h)
        for mi in range(S.M):
            m = clean[mi, vT]
            if m.sum() >= 8:
                res[ci, mi] = np.mean(np.abs(Y[mi, vT][m] - p[mi, o][m]) / Y[mi, vT][m]) * 100
    nv = np.full(S.M, np.nan)
    for mi in range(S.M):
        m = clean[mi, vT]
        if m.sum() >= 8:
            nv[mi] = np.mean(np.abs(Y[mi, vT][m] - Y[mi, vT - h][m]) / Y[mi, vT][m]) * 100
    return res, nv


def main():
    d = data.load_all()
    Y, _ = S.load_Y()
    clean = np.array([~flags(data.series(d, m)["MODAL"].astype(float).to_numpy(), data.series(d, m)["date"]) for m in MARKETS])
    T = S.load_forecast_tensors(Y)
    cfgs = V.enumerate_configs()
    out = Parallel(n_jobs=8)(delayed(val_masked)(h, cfgs, Y, T, clean) for h in S.H8)
    rel = np.stack([res / nv[None, :] for res, nv in out])               # (H, cfg, M)
    score = np.nanmean(rel, axis=(0, 2))                                  # same score: mean over markets and horizons
    comb = np.array([c["family"] in ("mean", "median") for c in cfgs])
    in_set = lambda sets: np.array([c["bases"] in sets for c in cfgs])
    chosen = {"V31_stack": S.pick(np.where(comb, np.inf, score), cfgs), "V31_comb": S.pick(np.where(comb, score, np.inf), cfgs)}
    for b in ("B3", "B5", "B1"):
        chosen[f"ABL_{b}_stack"] = S.pick(np.where(comb | ~in_set({b}), np.inf, score), cfgs)
    orig = pd.read_csv(RESULTS / "v31" / "tuning" / "v31_selected.csv").set_index("variant")["config"].to_dict()
    rows = [{"variant": k, "original selection (all validation weeks)": orig[k], "re-selection (unflagged validation weeks)": V.cfg_name(c),
             "same?": orig[k] == V.cfg_name(c), "score_clean": float(score[c["id"]])} for k, c in chosen.items()]
    st, cb = chosen["V31_stack"], chosen["V31_comb"]
    rows.append({"variant": "validation gate (stacker used only if >= 2% better than combination)",
                 "original selection (all validation weeks)": "combination used (gate failed)",
                 "re-selection (unflagged validation weeks)": "stacker used" if score[st["id"]] <= 0.98 * score[cb["id"]] else "combination used (gate failed)",
                 "same?": not (score[st["id"]] <= 0.98 * score[cb["id"]]), "score_clean": float(score[st["id"]] / score[cb["id"]])})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "v31" / "tuning" / "v31_reselect_clean.csv", index=False)
    top = np.argsort(np.where(comb, np.inf, score))[:5]
    txt = ["# Re-selection using only unflagged validation weeks\n", md(R) + "\n",
           "## Five best non-combination configurations on unflagged validation weeks (relative MAPE vs naive)\n",
           md(pd.DataFrame({"config": [V.cfg_name(cfgs[i]) for i in top], "score": score[top].round(4)})) + "\n",
           f"Validation weeks used per market: " + ", ".join(f"{m} {int(clean[i, S.Y_VAL].sum()) if hasattr(S,'Y_VAL') else int(clean[i, Y.shape[1]-104:Y.shape[1]-52].sum())}" for i, m in enumerate(MARKETS)) + "\n"]
    (RESULTS / "tables" / "RESELECT_CLEAN.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
