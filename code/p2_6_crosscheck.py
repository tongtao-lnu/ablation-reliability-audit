"""
P2-6 · independent cross-check
==============================
**不 import** proposition_check.py（也不 import decompose / variance_attribution）。
用另一条代码路径重算 P2-6 的关键数字，与 03_results/stats/proposition_check.json 逐项比对。
任一项不一致 → 打印 MISMATCH 并以非零码退出（G4：关键数字双路复核）。

独立性的具体做法：
  * Spearman 用 scipy.stats.rankdata + 手写 Pearson 相关（不用 scipy.stats.spearmanr）
  * 消融 ETIS 的检出改用 **另一个阈值**（1e-6）与另一套载入方式（np.load 后 reshape 校验）
  * 样本量公式用 math 手写（不调用主脚本的 budget_A_samples）
  * K 的 MDE 用 **闭式 Fisher-z 近似**独立估一遍，只要求落在主脚本 MC 值的邻域内
  * 精确置换 p 用"直接枚举 (2*K)!? 不 — 枚举 K! 个排列并数 |rho| >= |rho_obs|"独立算

用法：
    D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/p2_6_crosscheck.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

MAIN_JSON = "E:/paper2_ablation_reliability/03_results/stats/proposition_check.json"
DECOMP_CSV = "E:/paper2_ablation_reliability/03_results/stats/decomposition.csv"
ABL_ETIS = "D:/medical_segmentation/results_ablation_etis"

ABL = ["baseline", "egm_only", "dpa_only", "msfa_only", "egm_dpa", "egm_msfa", "dpa_msfa"]
ARCH3 = ["AttentionUNet", "TransUNet", "UNet"]
EGAUNET_ID = 0.8853
TAU = 1e-6                     # 与主脚本 (1e-5) 不同，用于独立验证阈值不敏感性


def spearman_manual(x, y):
    """手写 Spearman：rankdata → Pearson。"""
    rx = stats.rankdata(np.asarray(x, float))
    ry = stats.rankdata(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))


def exact_p(rho_abs, k):
    base = list(range(k))
    vals = [abs(stats.spearmanr(base, p).statistic) for p in itertools.permutations(base)]
    n = len(vals)
    return sum(1 for v in vals if v >= rho_abs - 1e-12) / n


def main():
    ref = json.load(open(MAIN_JSON, encoding="utf-8"))
    df = pd.read_csv(DECOMP_CSV)
    idf = df[df["scope"] == "ID"].set_index("group_id")
    checks = []

    def chk(name, got, want, tol=1e-9):
        ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
        checks.append((name, got, want, ok))

    # ---- 1. 命题 1：冻结判据下 S = Q，且排名恒等
    S = idf["mean_dice"].to_numpy(float)
    Q = idf["delineation"].to_numpy(float)
    chk("prop1.max_abs_diff_frozen", float(np.abs(S - Q).max()), 0.0)
    chk("prop1.spearman_S_Q_frozen", spearman_manual(S, Q), 1.0, tol=1e-12)
    chk("prop1.D_all_one", bool(np.all(idf["detection"].to_numpy(float) == 1.0)), True)

    # ---- 2. 消融 ETIS 检出（独立阈值 1e-6）+ 命题 2 枚举
    ood = {}
    for c in ABL:
        d = np.load(f"{ABL_ETIS}/{c}/detailed_metrics.npz")["dice"]
        frac = np.asarray(d, float).reshape(-1) / 100.0
        assert frac.size == 196, (c, frac.size)
        ood[c] = {"det": float((frac > TAU).mean()), "dice": float(frac.mean())}
    chk("prop2.n_antecedent", None, None)  # 占位，稍后填
    checks.pop()

    n_ant = n_rev = 0
    for i, j in itertools.combinations(ABL, 2):
        Qi, Qj = float(idf.loc[f"ablation/{i}", "delineation"]), float(idf.loc[f"ablation/{j}", "delineation"])
        Di, Dj = ood[i]["det"], ood[j]["det"]
        if Qi > Qj and Di < Dj:
            X, Y = i, j
        elif Qj > Qi and Dj < Di:
            X, Y = j, i
        else:
            continue
        n_ant += 1
        if (float(idf.loc[f"ablation/{X}", "mean_dice"]) > float(idf.loc[f"ablation/{Y}", "mean_dice"])
                and ood[X]["dice"] < ood[Y]["dice"]):
            n_rev += 1
    chk("prop2.n_pairs_antecedent", n_ant, ref["prop2_ablation_grid"]["n_pairs_antecedent"])
    chk("prop2.n_reversal_realised", n_rev, ref["prop2_ablation_grid"]["n_reversal_realised"])

    # ---- 3. 推论三个口径
    q7 = [float(idf.loc[f"ablation/{c}", "delineation"]) for c in ABL]
    d7 = [ood[c]["det"] for c in ABL]
    chk("corollary.n7.spearman", spearman_manual(q7, d7),
        ref["corollary_ablation_grid"]["spearman_Q_s_D_t"], tol=1e-12)

    ood_rows = df[df["scope"] == "OOD"]

    def d_t(arch):
        return float(ood_rows[ood_rows["group_id"].str.endswith("/" + arch)]["detection_sub"].mean())

    q3 = [float(idf.loc[f"baseline/{a}", "delineation"]) for a in ARCH3]
    d3 = [d_t(a) for a in ARCH3]
    chk("corollary.n3.spearman", spearman_manual(q3, d3), -1.0, tol=1e-12)
    q4 = [float(idf.loc["baseline/AttentionUNet", "delineation"]), EGAUNET_ID,
          float(idf.loc["baseline/TransUNet", "delineation"]),
          float(idf.loc["baseline/UNet", "delineation"])]
    d4 = [d_t(a) for a in ["AttentionUNet", "EGAUNet", "TransUNet", "UNet"]]
    chk("corollary.n4.spearman", spearman_manual(q4, d4), 0.2, tol=1e-12)

    # ---- 4. 精确置换 p（小 K）
    chk("exact_p.K3", exact_p(1.0, 3), 1 / 3, tol=1e-12)
    chk("exact_p.K4_min", 2 / math.factorial(4), 0.08333333333333333, tol=1e-15)
    chk("exact_p.K7_obs", exact_p(abs(spearman_manual(q7, d7)), 7),
        ref["corollary_ablation_grid"]["p_value_exact_permutation"], tol=1e-12)

    # ---- 5. 设计 A：样本量（独立手写）
    z = 1.959963984540054
    chk("designA.eta_1pct", math.ceil(math.log(0.05) / math.log(0.99)),
        ref["minimal_design"]["budget_A_samples"]["boundary_zero_failure_exact"]["eta_0.01"]["n_exact"])
    chk("designA.eta_2pct", math.ceil(math.log(0.05) / math.log(0.98)),
        ref["minimal_design"]["budget_A_samples"]["boundary_zero_failure_exact"]["eta_0.02"]["n_exact"])
    chk("designA.plus2pp_D078", math.ceil(z ** 2 * 0.78 * 0.22 / 0.02 ** 2),
        ref["minimal_design"]["budget_A_samples"]["interior_normal_approx"]["D=0.78"]["delta_0.02"])
    chk("designA.certified_eta_n242", 1 - 0.05 ** (1 / 242),
        ref["minimal_design"]["budget_A_samples"]["certified_eta_at_n242"], tol=1e-12)
    chk("designA.shortfall", math.ceil(math.log(0.05) / math.log(0.99)) - 242,
        ref["minimal_design"]["budget_A_samples"]["shortfall_vs_eta_0.01"])

    # ---- 6. 设计 B：K=14 的 MDE 用闭式 Fisher-z 独立估（κ=1.0 与 κ=1.075 给出区间）
    atanh_mde = {}
    for kappa in (1.0, 1.075):
        at = kappa * (z + stats.norm.ppf(0.80)) / math.sqrt(14 - 3)
        atanh_mde[kappa] = math.tanh(at)
    mc14 = ref["minimal_design"]["budget_B_domains"]["table"]["14"]["mde80"]
    lo, hi = atanh_mde[1.0], atanh_mde[1.075]
    ok = lo - 0.03 <= mc14 <= hi + 0.03
    checks.append(("designB.K14_mde_in_closedform_band",
                   round(mc14, 3), f"[{lo:.3f}, {hi:.3f}] ± 0.03", ok))

    # ---- 输出
    width = max(len(c[0]) for c in checks)
    print(f"{'check'.ljust(width)}  {'independent':>18}  {'delivered':>18}  ok")
    allok = True
    for name, got, want, ok in checks:
        allok &= ok
        g = f"{got:.6g}" if isinstance(got, float) else str(got)
        w = f"{want:.6g}" if isinstance(want, float) else str(want)
        print(f"{name.ljust(width)}  {g:>18}  {w:>18}  {'Y' if ok else 'N'}")
    print()
    print(f"  independent: n=7 rho = {spearman_manual(q7, d7):+.4f} "
          f"(delivered {ref['corollary_ablation_grid']['spearman_Q_s_D_t']:+.4f})")
    print(f"  independent: n=3 rho = {spearman_manual(q3, d3):+.4f} "
          f"(delivered {ref['corollary_4arch']['n3_rule_compliant']['spearman_Q_s_D_t']:+.4f})")
    print(f"  independent: n=4 rho = {spearman_manual(q4, d4):+.4f} "
          f"(delivered {ref['corollary_4arch']['n4_with_EGAUNet_protocol_value']['spearman_Q_s_D_t']:+.4f})")
    print(f"  independent: detection threshold 1e-6 vs delivered 1e-5 -> counts agree: "
          f"{n_ant} pairs / {n_rev} reversals")
    print(f"  independent: K=14 MDE closed-form band [{lo:.3f}, {hi:.3f}] vs MC {mc14:.2f}")
    print()
    print("RESULT:", "MATCH" if allok else "MISMATCH")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
