"""550 configurations (adds simplex family, margin gate, base set B5), validation gate, ablation.
Stages: tune (validation only) -> final (test, once).   Usage: python v31_stack.py tune | final"""
import itertools
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import v3_stack as S
from config import MARKETS, RESULTS

H8 = S.H8
OUT = RESULTS / "v31"
TDIR = OUT / "tuning"
FDIR = OUT / "forecasts"


def enumerate_configs():
    cfgs = []
    for b in S.BASESETS:
        for fam in ("mean", "median"):
            cfgs.append({"bases": b, "family": fam, "pooled": 0, "gate": 0})
        for gate, pooled in itertools.product((0, 1, 2), (0, 1)):
            for eta, gamma in itertools.product((0.5, 1, 2, 4), (1.0, 0.97, 0.9)):
                cfgs.append({"bases": b, "family": "online", "eta": eta, "gamma": gamma, "pooled": pooled, "gate": gate})
            for fam in ("nnls", "simplex"):
                for lam in (0, 0.5, 0.8):
                    cfgs.append({"bases": b, "family": fam, "lam": lam, "pooled": pooled, "gate": gate})
    for i, c in enumerate(cfgs):
        c["id"] = i
    return cfgs


def cfg_name(c):
    ex = {"online": f"eta{c.get('eta')}g{c.get('gamma')}", "nnls": f"lam{c.get('lam')}", "simplex": f"lam{c.get('lam')}"}.get(c["family"], "")
    return f"{c['bases']}|{c['family']}{ex}|{'pooled' if c['pooled'] else 'mkt'}|gate{c['gate']}"


def tune():
    Y, _ = S.load_Y()
    T = S.load_forecast_tensors(Y)
    cfgs = enumerate_configs()
    out = Parallel(n_jobs=8)(delayed(S.val_mape_for_h)(h, cfgs, Y, T) for h in H8)
    TDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, (res, naive) in zip(H8, out):
        for ci, c in enumerate(cfgs):
            for mi, mk in enumerate(MARKETS):
                rows.append({"id": c["id"], "name": cfg_name(c), "h": h, "market": mk, "val_mape": res[ci, mi], "val_naive": naive[mi]})
    pd.DataFrame(rows).to_csv(TDIR / "v31_validation.csv", index=False)
    print("tuned", len(cfgs), "configs")


def select():
    cfgs = enumerate_configs()
    v = pd.read_csv(TDIR / "v31_validation.csv")
    v["rel"] = v["val_mape"] / v["val_naive"]
    s_all = v.groupby("id")["rel"].mean().sort_index().to_numpy()
    comb = np.array([c["family"] in ("mean", "median") for c in cfgs])
    in_set = lambda sets: np.array([c["bases"] in sets for c in cfgs])
    inf = np.inf
    chosen = {}
    chosen["V31_stack"] = S.pick(np.where(comb, inf, s_all), cfgs)
    chosen["V31_comb"] = S.pick(np.where(comb, s_all, inf), cfgs)
    # validation gate: stacker only if >= 2% (relative) better than the equal-effort combination control
    st, cb = chosen["V31_stack"], chosen["V31_comb"]
    gated = st if s_all[st["id"]] <= 0.98 * s_all[cb["id"]] else cb
    chosen["V31_stack_vgated"] = gated
    for b in ("B3", "B5", "B1"):
        chosen[f"ABL_{b}_stack"] = S.pick(np.where(comb | ~in_set({b}), inf, s_all), cfgs)
        chosen[f"ABL_{b}_mean"] = next(c for c in cfgs if c["bases"] == b and c["family"] == "mean")
    rep = pd.DataFrame([{"variant": k, "config": cfg_name(c), "val_rel_mape_vs_naive_all_h": s_all[c["id"]]} for k, c in chosen.items()])
    rep["v_gate_used_stacker"] = [(s_all[st["id"]] <= 0.98 * s_all[cb["id"]]) if k == "V31_stack_vgated" else "" for k in chosen]
    rep["n_configs_searched"] = len(cfgs)
    rep.to_csv(TDIR / "v31_selected.csv", index=False)
    return chosen, rep


def final():
    Y, dates = S.load_Y()
    T = S.load_forecast_tensors(Y)
    chosen, rep = select()
    print(rep.to_string())
    res = Parallel(n_jobs=5)(delayed(S.final_seed)(s, chosen, Y, T, dates) for s in range(5))
    FDIR.mkdir(parents=True, exist_ok=True)
    pd.concat(res, ignore_index=True).to_csv(FDIR / "v31_final.csv", index=False)
    print("final written")


if __name__ == "__main__":
    {"tune": tune, "final": final}[sys.argv[1]]()
