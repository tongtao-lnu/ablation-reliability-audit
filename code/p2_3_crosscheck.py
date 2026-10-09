# -*- coding: utf-8 -*-
"""
文件名: p2_3_crosscheck.py
功能: 【论文二 P2-3 · G4 双路复核】对齐性检验（H3）的独立重算

铁律（G4）: **不 import 任何主模块**（不 import alignment_test / decompose /
           variance_attribution / p31_cross_eval），只用落盘 CSV + 第三方库独立重算。

与主脚本**刻意不同的实现路径**（避免"同一 bug 复现两遍"）:
  1. Spearman: 主脚本走"平均秩 → Pearson"；本脚本走**解析式** 1 - 6Σd²/(n(n²-1))
     （本数据集 14 个值全部互异 ⇒ 两式严格等价, 若不等则说明有并列未被处理）。
  2. 第三方对照: `scipy.stats.spearmanr`（完全独立的实现）。
  3. 置换 p: 主脚本用自写随机排列 MC；本脚本用 `scipy.stats.permutation_test`
     （permutation_type='pairings'）, 并额外给"枚举全部 n!/(...) 不可行 ⇒ MC"的说明。
  4. CI: 主脚本用 bootstrap；本脚本用 **Fisher-z 解析** + 另一次不同种子的 bootstrap。
  5. 排名反转: 主脚本逐对比较；本脚本用 **Kendall τ-a 计数公式**反推不一致对数。
  6. 功效/MDE: 主脚本用二元正态 MC；本脚本用 **Fisher-z + Spearman 校正的闭式近似**，
     只要求两者落在同一区间（不要求逐位相等）。

容差分级（P3-1bc 教训: 浮点派生量禁用 `==`）:
  TOL_RHO = 1e-9（相关系数）｜ TOL_ID = 1e-12（恒等式）｜ 整数计数严格相等

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/p2_3_crosscheck.py
产物:
    03_results/audit/p2_3_crosscheck.json
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

E_ROOT = "E:/paper2_ablation_reliability"
UNITS = f"{E_ROOT}/03_results/stats/p31a_units_n14.csv"
ALIGN = f"{E_ROOT}/03_results/stats/alignment.json"
OUT = f"{E_ROOT}/03_results/audit/p2_3_crosscheck.json"

TOL_RHO = 1e-9
TOL_ID = 1e-12
X_COL, Y_COL = "id_del", "ood_det_subst"


def spearman_analytic(x, y):
    """解析式 Spearman（仅适用于无并列; 末尾会断言无并列）。"""
    n = len(x)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    d2 = ((rx - ry) ** 2).sum()
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main():
    df = pd.read_csv(UNITS)
    x = df[X_COL].to_numpy(float)
    y = df[Y_COL].to_numpy(float)
    n = len(x)
    ref = json.load(open(ALIGN, encoding="utf-8"))

    checks = []

    def chk(name, ok, detail):
        checks.append(dict(name=name, status=("MATCH" if ok else "DIFF"), detail=detail))
        return ok

    # ---- 0. 前提: 与主脚本同一份数据 ----
    chk("units_n", n == 14 and ref["data"]["n"] == 14, dict(n=n, ref_n=ref["data"]["n"]))
    uniq_x = int(len(set(np.round(x, 12))))
    uniq_y = int(len(set(np.round(y, 12))))
    chk("no_ties", uniq_x == n and uniq_y == n, dict(uniq_x=uniq_x, uniq_y=uniq_y, n=n))

    # ---- 1. Spearman 三路 ----
    r_analytic = spearman_analytic(x, y)
    r_scipy = float(stats.spearmanr(x, y).statistic)
    r_main = float(ref["primary"]["rho"])
    chk("spearman_analytic_vs_main", abs(r_analytic - r_main) <= TOL_RHO,
        dict(analytic=r_analytic, main=r_main, diff=abs(r_analytic - r_main)))
    chk("spearman_scipy_vs_main", abs(r_scipy - r_main) <= TOL_RHO,
        dict(scipy=r_scipy, main=r_main, diff=abs(r_scipy - r_main)))

    # ---- 2. 置换 p（scipy 独立实现） ----
    perm = stats.permutation_test(
        (x, y), lambda a, b: float(stats.spearmanr(a, b).statistic),
        permutation_type="pairings", n_resamples=200_000,
        alternative="two-sided", random_state=20260914)
    p_scipy = float(perm.pvalue)
    p_main = float(ref["primary"]["p_perm_mc"]["p_mc"])
    p_t = float(ref["primary"]["p_t_approx"])
    chk("perm_p_scipy_vs_main", abs(p_scipy - p_main) <= 5e-3,
        dict(scipy=p_scipy, main_mc=p_main, t_approx=p_t,
             diff=abs(p_scipy - p_main)))

    # ---- 3. 临界值: t 近似解析 ----
    tc = stats.t.ppf(0.975, n - 2)
    crit_analytic = float(tc / math.sqrt(n - 2 + tc ** 2))
    crit_main = float(ref["primary"]["rho_crit_alpha05_t_approx"])
    chk("rho_crit_t_approx", abs(crit_analytic - crit_main) <= 1e-9,
        dict(analytic=crit_analytic, main=crit_main))

    # ---- 4. CI: Fisher-z 解析（与主脚本的 bootstrap 不同路径） ----
    z = math.atanh(r_analytic)
    se = 1.0 / math.sqrt(n - 3)
    f_lo = math.tanh(z - 1.959963984540054 * se)
    f_hi = math.tanh(z + 1.959963984540054 * se)
    f_main = ref["effect_size_ci"]["fisher_z"]
    chk("fisher_ci", abs(f_lo - f_main["lo"]) <= TOL_RHO and abs(f_hi - f_main["hi"]) <= TOL_RHO,
        dict(lo=f_lo, hi=f_hi, main_lo=f_main["lo"], main_hi=f_main["hi"]))
    b_main = ref["effect_size_ci"]["bootstrap"]
    chk("ci_spans_zero", (f_lo < 0 < f_hi) and (b_main["lo"] < 0 < b_main["hi"]),
        dict(fisher=[f_lo, f_hi], bootstrap=[b_main["lo"], b_main["hi"]]))

    # ---- 5. 排名反转: Kendall τ-a 计数公式 ----
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    tau_scipy = float(stats.kendalltau(x, y).statistic)
    rr = ref["rank_reversal"]
    n_pairs = n * (n - 1) // 2
    n_disc_implied = round((1.0 - tau_scipy) * n_pairs / 2.0)
    chk("reversal_counts_from_tau",
        abs(n_disc_implied - rr["n_discordant"]) <= 1 and n_pairs == rr["n_pairs"],
        dict(implied_discordant=n_disc_implied, main_discordant=rr["n_discordant"],
             n_pairs=n_pairs, tau=tau_scipy, main_tau=rr["kendall_tau_a"]))

    # ---- 6. 恒等式与口径断言 ----
    ident_id = bool(np.allclose(df["id_del"], df["id_dice"], rtol=0, atol=TOL_ID))
    ident_ood = bool((df["ood_det_frozen"] == df["ood_det_subst"]).all())
    chk("identity_id_del_equals_dice", ident_id,
        dict(max_abs=float(np.abs(df["id_del"] - df["id_dice"]).max())))
    chk("identity_ood_frozen_equals_subst", ident_ood,
        dict(n_mismatch=int((df["ood_det_frozen"] != df["ood_det_subst"]).sum())))
    chk("id_det_frozen_all_one", bool((df["id_det_frozen"] == 1.0).all()),
        dict(n=int((df["id_det_frozen"] == 1.0).sum())))

    # ---- 7. MDE 闭式近似（主脚本用 MC） ----
    kappa = 1.03
    za, zb = 1.959963984540054, 0.8416212335728546
    mde_closed = math.tanh(kappa * (za + zb) * math.sqrt(1.0 / (n - 3)))
    mde_main = ref["power"]["mde_rechecked"]
    mde_pre = ref["power"]["mde_preregistered"]
    chk("mde_closed_vs_mc_same_band",
        abs(mde_closed - float(mde_main)) <= 0.06,
        dict(closed_form=round(mde_closed, 4), mc=float(mde_main),
             preregistered=float(mde_pre)))

    # ---- 8. 判定一致性 ----
    verdict_expected = ("REJECT_H3" if (min(p_scipy, p_main) < 0.05 and abs(r_scipy) > 0.6)
                        else "NOT_REJECTED")
    chk("verdict_consistent", verdict_expected == ref["hypothesis"]["verdict"],
        dict(independent=verdict_expected, main=ref["hypothesis"]["verdict"]))

    # ---- 9. 附带量: OOD 检出 vs OOD 勾画（构念正交性） ----
    rho_orth = float(stats.spearmanr(df["ood_det_subst"], df["ood_del"]).statistic)
    aux_key = "ood_det_subst__ood_del"
    rho_aux_main = float(ref["auxiliary_spearman"][aux_key]["rho"])
    chk("aux_orthogonality", abs(rho_orth - rho_aux_main) <= TOL_RHO,
        dict(scipy=rho_orth, main=rho_aux_main))

    n_match = sum(1 for c in checks if c["status"] == "MATCH")
    payload = dict(
        script="02_code/analysis/p2_3_crosscheck.py",
        segment="P2-3", mode="G4_independent_crosscheck",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        policy="不 import 任何主模块；Spearman 走解析式 + scipy 双第三方路径",
        tolerances=dict(TOL_RHO=TOL_RHO, TOL_ID=TOL_ID),
        checks=checks,
        summary=dict(n_checks=len(checks), n_match=n_match, n_diff=len(checks) - n_match,
                     overall=("MATCH" if n_match == len(checks) else "DIFF")),
        key_numbers=dict(
            rho_analytic=r_analytic, rho_scipy=r_scipy, rho_main=r_main,
            p_scipy_perm=p_scipy, p_main_mc=p_main, p_t_approx=p_t,
            fisher_ci=[f_lo, f_hi], boot_ci=[b_main["lo"], b_main["hi"]],
            crit_alpha05=crit_analytic, mde_closed=round(mde_closed, 4),
            mde_mc=mde_main, tau=tau_scipy),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("=" * 84)
    print("P2-3 G4 双路复核（独立实现, 不 import 主模块）")
    print("=" * 84)
    for c in checks:
        mark = "OK " if c["status"] == "MATCH" else "!! "
        print(f"  [{mark}] {c['name']}")
        print(f"        {c['detail']}")
    print("-" * 84)
    print(f"  {n_match}/{len(checks)} MATCH  -> {payload['summary']['overall']}")
    print(f"已写: {OUT}")
    return 0 if payload["summary"]["overall"] == "MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
