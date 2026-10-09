# -*- coding: utf-8 -*-
"""
文件名: p2_5_crosscheck.py
功能: 【论文二 P2-5 · G4 双路复核】**不 import 主模块**，独立重算并逐项比对。

独立性保证:
  1. **不 import** `robustness.py`（不共用任何函数、常量、口径代码）。
  2. **重新读原始来源**：架构级直接读 `p31a_units_n14.csv`；
     配置级**直接读 D 盘 results.json + E 盘 decomp_ablation_etis.csv**，
     **不读** robustness.json，也**不读** confound.json。
  3. **换实现**：秩用 scipy.stats.rankdata；相关用 scipy.stats.spearmanr；
     置信用同一 seed 的等价 RNG 流程重新实现（同口径 ⇒ 期望逐位一致）。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/p2_5_crosscheck.py
退出码: 0 = 全部 MATCH；1 = 有 MISMATCH
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

E = "E:/paper2_ablation_reliability"
STATS = f"{E}/03_results/stats"
D_PROJ = "D:/medical_segmentation"
ABLATION_ROOT = f"{D_PROJ}/experiments/ablation"
ABLATION_ORDER = ["baseline", "dpa_only", "egm_only", "msfa_only",
                  "egm_dpa", "egm_msfa", "dpa_msfa"]
NEAR_TIE = 0.01
SEED = 20260914
BOOT_B = 10000
TOL = 1e-9

rows = []


def cmp(name, got, exp, tol=TOL):
    if isinstance(got, bool) or isinstance(exp, bool):
        ok = bool(got) == bool(exp)
    elif got is None or exp is None or not np.isfinite(got) or not np.isfinite(exp):
        ok = (got is None and exp is None)
    else:
        ok = abs(float(got) - float(exp)) <= tol
    rows.append((name, ok, got, exp))
    return ok


def load_arch():
    df = pd.read_csv(f"{STATS}/p31a_units_n14.csv")
    return df[["arch", "id_dice", "id_del", "ood_dice", "ood_det_subst"]].copy()


def load_cfg():
    dec = pd.read_csv(f"{STATS}/decomp_ablation_etis.csv").set_index("config")
    out = []
    for c in ABLATION_ORDER:
        with open(os.path.join(ABLATION_ROOT, c, "results.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        out.append(dict(arch=c, id_dice=float(m["test_metrics"]["Dice"]),
                        id_del=float(m["test_metrics"]["Dice"]),
                        ood_dice=float(dec.loc[c, "mean_dice"]),
                        ood_det_subst=float(dec.loc[c, "detection"])))
    return pd.DataFrame(out)


# --- 独立实现：平均秩（scipy）+ 定向成对清单 ---
def pair_list(df, s_col, t_col, q_col, d_col):
    s = df[s_col].to_numpy(float)
    t = df[t_col].to_numpy(float)
    q = df[q_col].to_numpy(float)
    d = df[d_col].to_numpy(float)
    out = []
    for a in range(len(df)):
        for b in range(a + 1, len(df)):
            if s[a] == s[b]:
                continue
            i, j = (a, b) if s[a] > s[b] else (b, a)
            out.append(dict(dlogQ=math.log(q[i] / q[j]),
                            dlogD=math.log(d[j] / d[i]),
                            delta=math.log(t[j] / t[i])))
    return out


def _mean(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()) if a.size else float("nan")


def summ(df, s_col, t_col, q_col, d_col):
    P = pair_list(df, s_col, t_col, q_col, d_col)
    s = df[s_col].to_numpy(float)
    t = df[t_col].to_numpy(float)
    rs = sps.rankdata(s)
    rt = sps.rankdata(t)
    conc = disc = 0
    for a in range(len(df)):
        for b in range(a + 1, len(df)):
            dx, dy = rs[a] - rs[b], rt[a] - rt[b]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    n_pairs = len(df) * (len(df) - 1) // 2
    deltas = np.array([p["delta"] for p in P], dtype=float)
    rev = deltas[deltas > 0]
    near = np.array([p["delta"] for p in P if p["dlogQ"] <= NEAR_TIE], dtype=float)
    return dict(n_pairs=n_pairs, conc=conc, disc=disc,
                tau_a=(conc - disc) / n_pairs,
                rho=float(sps.spearmanr(s, t).statistic),
                mean_rev=_mean(rev), signed_dev=_mean(deltas),
                near_mean=_mean(near), n_near=int(len(near)),
                res_max=float(max(abs(p["delta"] - (p["dlogD"] - p["dlogQ"])) for p in P)),
                deltas=deltas, near=near)


def boot_ci(df, s_col, t_col, q_col, d_col, which):
    rng = np.random.default_rng(SEED)
    n = len(df)
    vals = []
    for _ in range(BOOT_B):
        idx = np.unique(rng.integers(0, n, n))
        if len(idx) < 3:
            continue
        sub = df.iloc[idx].reset_index(drop=True)
        S = summ(sub, s_col, t_col, q_col, d_col)
        vals.append({"mean_rev": S["mean_rev"], "signed_dev": S["signed_dev"],
                     "near_mean": S["near_mean"]}[which])
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def pair_boot(vals):
    """独立实现的对级 bootstrap（同 seed、同口径 ⇒ 期望逐位一致）。"""
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(BOOT_B, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def loo_signs(df, s_col, t_col, q_col, d_col):
    base = summ(df, s_col, t_col, q_col, d_col)
    fl = {"tau_a": 0, "mean_rev": 0, "signed_dev": 0, "near_mean": 0}
    for k in range(len(df)):
        S = summ(df.drop(index=df.index[k]).reset_index(drop=True),
                 s_col, t_col, q_col, d_col)
        for key in fl:
            if np.sign(S[key]) != np.sign(base[key]):
                fl[key] += 1
    return fl


def main():
    ref = json.load(open(f"{STATS}/robustness.json", encoding="utf-8"))
    arch, cfg = load_arch(), load_cfg()
    specs = {
        "arch14__dice": (arch, "id_dice", "ood_dice", "id_del", "ood_det_subst"),
        "arch14__align": (arch, "id_del", "ood_det_subst", "id_del", "ood_det_subst"),
        "config7__dice": (cfg, "id_dice", "ood_dice", "id_dice", "ood_det_subst"),
    }
    for key, (df, s, t, q, d) in specs.items():
        got = summ(df, s, t, q, d)
        R = ref["pairings"][key]
        cmp(f"[{key}] n_pairs", got["n_pairs"], R["n_pairs"])
        cmp(f"[{key}] 一致对", got["conc"], R["n_concordant"])
        cmp(f"[{key}] 反转对", got["disc"], R["n_discordant"])
        cmp(f"[{key}] τ_a", got["tau_a"], R["kendall_tau_a"], 1e-12)
        cmp(f"[{key}] ρ(Spearman, scipy)", got["rho"], R["rho_spearman"], 1e-12)
        cmp(f"[{key}] 反转幅度均值", got["mean_rev"], R["rev_mag"]["mean"])
        cmp(f"[{key}] 全对定向 Δ 均值", got["signed_dev"], R["signed_dev"]["mean"])
        cmp(f"[{key}] 近并列对数", got["n_near"], R["near_tie"]["n"])
        cmp(f"[{key}] 近并列 Δ 均值", got["near_mean"], R["near_tie"]["mean_delta"])
        cmp(f"[{key}] 反转幅度 min", float(got["deltas"][got["deltas"] > 0].min()),
            R["rev_mag"]["min_"])
        cmp(f"[{key}] 反转幅度 max", float(got["deltas"][got["deltas"] > 0].max()),
            R["rev_mag"]["max_"])
        # bootstrap 端点（同 seed、同流程 ⇒ 期望逐位一致）
        for which, path in [("mean_rev", "mean_rev_mag"),
                            ("signed_dev", "signed_dev"),
                            ("near_mean", "near_tie_mean_delta")]:
            lo, hi = boot_ci(df, s, t, q, d, which)
            bc = ref["bootstrap"][key]["ci"][path]
            cmp(f"[{key}] boot {which} lo", lo, bc["lo"], 1e-12)
            cmp(f"[{key}] boot {which} hi", hi, bc["hi"], 1e-12)
        # 对级 bootstrap 端点（同 seed、同流程 ⇒ 期望逐位一致）
        P = pair_list(df, s, t, q, d)
        deltas = np.array([p["delta"] for p in P], dtype=float)
        near = np.array([p["delta"] for p in P if p["dlogQ"] <= NEAR_TIE], dtype=float)
        for vals, path in [(deltas[deltas > 0], "rev_mag"), (deltas, "signed_dev"),
                           (near, "near_tie_mean_delta")]:
            lo, hi = pair_boot(vals)
            bc = ref["pair_bootstrap"][key][path]
            cmp(f"[{key}] 对级 boot {path} lo", lo, bc["lo"], 1e-12)
            cmp(f"[{key}] 对级 boot {path} hi", hi, bc["hi"], 1e-12)
        # 留一变号次数（primary + aux）
        fl = loo_signs(df, s, t, q, d)
        LOO = ref["leave_one_out"][key]["checks"]
        for k2, ck in [("tau_a", "kendall_tau_a"), ("mean_rev", "mean_rev_mag"),
                       ("signed_dev", "signed_dev"), ("near_mean", "near_tie_mean_delta")]:
            cmp(f"[{key}] 留一变号次数 {k2}", fl[k2], LOO[ck]["n_flips"])

    # 与 P2-3 冻结产物对账（独立读 alignment.json）
    al = json.load(open(f"{STATS}/alignment.json", encoding="utf-8"))["rank_reversal"]
    cmp("P2-3 alignment.json 反转对", al["n_discordant"], ref["pairings"]["arch14__align"]["n_discordant"])
    cmp("P2-3 alignment.json τ_a", al["kendall_tau_a"], ref["pairings"]["arch14__align"]["kendall_tau_a"], 1e-12)

    W = 96
    print("=" * W)
    print("P2-5 G4 双路复核（独立重算，不 import 主模块）")
    print("=" * W)
    bad = 0
    for name, ok, got, exp in rows:
        if isinstance(got, (int, np.integer)) and not isinstance(got, bool):
            gs, es = f"{int(got)}", f"{int(exp)}"
        elif isinstance(got, bool):
            gs, es = str(got), str(exp)
        else:
            gs, es = f"{float(got):.10g}", f"{float(exp):.10g}"
        print(f"  [{'  MATCH  ' if ok else ' MISMATCH'}] {name:<44} got={gs:<22} exp={es}")
        bad += (not ok)
    print("-" * W)
    print(f"共 {len(rows)} 项：MATCH {len(rows)-bad} ｜ MISMATCH {bad}")
    print("结论:", "**全部 MATCH** —— P2-5 关键数字双路复核通过 ✅" if bad == 0
          else "❌ 存在 MISMATCH，必须排查")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
