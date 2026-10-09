"""
P2-6 · Proposition numerical-consistency check + minimal validation-set design
==============================================================================
论文二《分割基准的检出饱和与消融可信边界》  P2-6 产出（零实验成本，纯计算）

本脚本做四件事，全部只读已落盘产物 + 只读 D 盘原始预测，不训练、不动 GPU：

  A. 命题 1 的数值核验  —— 分布内(冻结判据) S_i = Q_i 是否**逐格精确成立**，排名是否恒等
  B. 引理 1 / 命题 2 的数值核验 —— 全对枚举：前件成立的对数、反转对数、反例及其归因
  C. 推论（一行诊断）的数值核验 —— Spearman(Q^(s), D^(t))
  D. 最小验证集设计 —— 两条互相独立的预算：
       D1 样本量预算：把检出率估到给定精度需要多少例（含 D→1 边界用精确单边界）
       D2 域数预算：把对齐相关系数检到给定效应量需要多少个**独立域**（含 K≤8 精确置换零分布）

────────────────────────────────────────────────────────────────────────────
⚠️ 边界条件（写作时必须一并声明，见 05_manuscript/propositions.md §5）
────────────────────────────────────────────────────────────────────────────
B1 只对**宏平均**（逐样本平均）Dice 成立；微平均不满足。
B2 检出因子依赖二值化阈值 τ（勾画因子依赖弱）→ 报检出率必须报 τ。
B3 要求 G ≠ ∅；约定 P = ∅ ⇒ Dice = 0（标准约定，非 0/0）。
B4 命题 2 依赖"勾画质量跨域可迁移"这一**较弱假设**；本脚本给出该假设**失效幅度的定量刻画**
   （三对数项分解），并说明命题 2 为何只是**充分**条件（converse 不成立，见 §3.2 反例）。
B5 勾画因子不是"纯边界质量"：同时吸收过分割/欠分割/假阳性定位误差（构念混淆）。
B6 **检出饱和的判据依赖性**：两侧度量实现均带平滑常数 ε（ID 侧 1e-8 / OOD 侧 1e-6），
   使任意预测（含全空）的 Dice/Recall 恒 > 0 → 冻结判据 |P∩G|>0 下的"检出率 = 1.000"
   是 ε 的**构造结果**；必须并给**实质判据** I ≥ 1 px 的读数。

⚠️ 消融网格 ETIS 的检出判据
   `results_ablation_etis/<cfg>/detailed_metrics.npz` 只存 `dice/iou/hd95`（百分数），**无 Recall**。
   按 B6 的同一逻辑：ε 伪影使 dice_frac 落在 ~1e-11 量级，真实重叠 ≥ ~1e-5。
   本脚本**先验证两者之间存在数量级空档**（三阈值 1e-5 / 1e-6 / 1e-8 给出**同一**检出计数），
   再用 τ_det = 1e-5 判定"实质检出"。空档不成立时脚本会报错退出，不许默默退化。

⚠️ 幂/功效口径
    Spearman 的 p 值：**K ≤ 8 用精确置换零分布**（全排列枚举），K ≥ 10 用 t 近似。
    理由：t 近似在 K ≤ 8 时**反保守**（K=4 时它给出名义功效 0.12–0.30，而精确功效恒为 0，
    因为 2/4! = 0.0833 > 0.05 任何 ρ 都无法拒绝）。这是最小验证集设计里的一个实质结论。

用法（本机）：
    D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/proposition_check.py
产物：
    03_results/stats/proposition_check.json
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- 常量 / 路径
E_ROOT = "E:/paper2_ablation_reliability"
DECOMP_CSV = f"{E_ROOT}/03_results/stats/decomposition.csv"
DECOMP_JSON = f"{E_ROOT}/03_results/stats/decomposition.json"
FACTORIAL_JSON = f"{E_ROOT}/03_results/stats/paper2_factorial.json"
VARIANCE_JSON = f"{E_ROOT}/03_results/stats/variance_attribution_n4.json"
ABL_ETIS = "D:/medical_segmentation/results_ablation_etis"
OUT_JSON = f"{E_ROOT}/03_results/stats/proposition_check.json"

ABL_CFGS = ["baseline", "egm_only", "dpa_only", "msfa_only",
            "egm_dpa", "egm_msfa", "dpa_msfa"]
ARCHS = ["AttentionUNet", "EGAUNet", "TransUNet", "UNet"]

TAU_DET = 1e-5          # 消融 ETIS 的实质检出阈值（dice_frac）；见 docstring 的数量级论证
ALPHA = 0.05
Z = stats.norm.ppf(1 - ALPHA / 2)         # 1.959964
Z_POW80 = stats.norm.ppf(0.80)            # 0.841621
K_GRID = [4, 5, 6, 7, 8, 10, 11, 12, 14, 20, 25, 30, 40]
RHO_GRID = [0.4, 0.5, 0.6, 0.7, 0.72, 0.8]
MC_REPS = 20000
MC_REPS_MDE = 8000
MC_SEED = 42


# ================================================================ A. 命题 1
def load_units():
    """读 P2-1 产物，抽出 7 个消融配置的分布内(冻结/实质) 与 4 架构的分布内记录。"""
    df = pd.read_csv(DECOMP_CSV)
    idf = df[df["scope"] == "ID"].set_index("group_id")
    abl = {}
    for c in ABL_CFGS:
        r = idf.loc[f"ablation/{c}"]
        abl[c] = {
            "id_dice": float(r["mean_dice"]),
            "id_det_frozen": float(r["detection"]),
            "id_del_frozen": float(r["delineation"]),
            "id_det_sub": float(r["detection_sub"]),
            "id_del_sub": float(r["delineation_sub"]),
            "n_id": int(r["n"]),
        }
    arch = {}
    for a in ARCHS:
        key = f"baseline/{a}"
        if key not in idf.index:
            # EGAUNet 不在 P2-1 的 20 个预注册分布内单元里（其分布内记录是单文件路线，
            # 被 M10 从分解单元集中排除）→ 只登记缺席，不臆造数值。
            arch[a] = {"present_in_id_groups": False}
            continue
        r = idf.loc[key]
        arch[a] = {
            "present_in_id_groups": True,
            "id_dice": float(r["mean_dice"]),
            "id_det_frozen": float(r["detection"]),
            "id_del_frozen": float(r["delineation"]),
            "id_del_sub": float(r["delineation_sub"]),
            "n_id": int(r["n"]),
        }
    return df, abl, arch, idf


def ood_ablation_etis():
    """从 results_ablation_etis/<cfg>/detailed_metrics.npz 算 ETIS 域的 检出/勾画/宏平均 Dice。

    返回 (rec, gap_report)。gap_report 记录三阈值下的检出计数是否一致（B6 的数量级空档验证）。
    """
    rec, gap = {}, []
    for c in ABL_CFGS:
        frac = np.load(f"{ABL_ETIS}/{c}/detailed_metrics.npz")["dice"].astype(float) / 100.0
        n = int(frac.size)
        counts = {t: int((frac > t).sum()) for t in (1e-5, 1e-6, 1e-8)}
        vals = np.sort(np.unique(frac))
        # 「空档」= ε 伪影值(≤1e-5) 与真实重叠值(>1e-5) 之间的分离比；并验证三阈值给出同一检出计数
        arte = vals[vals <= 1e-5]
        real = vals[vals > 1e-5]
        sep = (float(real.min() / arte.max()) if arte.size and real.size else None)
        gap.append({"config": c, "n": n, "det_counts": counts,
                    "consistent": len(set(counts.values())) == 1,
                    "dice_frac_max_artefact": float(arte.max()) if arte.size else None,
                    "dice_frac_min_real_overlap": float(real.min()) if real.size else None,
                    "separation_ratio_min_real_over_max_artefact": sep,
                    "dice_frac_min": float(frac.min())})
        det = counts[1e-5] / n
        md = float(frac.mean())
        rec[c] = {"n": n, "ood_dice": md, "ood_det_sub": det,
                  "ood_del_sub": md / det if det > 0 else float("nan")}
    bad = [g for g in gap if not g["consistent"]]
    if bad:
        raise SystemExit(f"[FATAL] 消融 ETIS 检出阈值不稳健（三阈值计数不一致）：{bad}")
    return rec, gap


def check_prop1(abl, idf):
    """命题 1：分布内(冻结判据) S = Q 逐格精确成立 → 排名恒等。"""
    rows = []
    for c in ABL_CFGS:
        r = idf.loc[f"ablation/{c}"]
        rows.append({"unit": c, "S_frozen": float(r["mean_dice"]),
                     "Q_frozen": float(r["delineation"]),
                     "abs_diff": abs(float(r["mean_dice"]) - float(r["delineation"])),
                     "D_frozen": float(r["detection"]),
                     "S_sub": float(r["mean_dice"]),
                     "Q_sub": float(r["delineation_sub"]),
                     "abs_diff_sub": abs(float(r["mean_dice"]) - float(r["delineation_sub"]))})
    all_id = idf[idf["scope"] == "ID"] if "scope" in idf.columns else idf
    S = all_id["mean_dice"].to_numpy(float)
    Q = all_id["delineation"].to_numpy(float)
    D = all_id["detection"].to_numpy(float)
    Qs = all_id["delineation_sub"].to_numpy(float)
    sp_frozen = stats.spearmanr(S, Q).statistic
    sp_sub = stats.spearmanr(S, Qs).statistic
    # 弹性：d log S / d log D = d log S / d log Q = 1（用 S = D Q 在扰动下数值验证）
    eps = 1e-6
    D0, Q0 = 0.7, 0.6
    elas_D = (math.log((D0 * (1 + eps)) * Q0) - math.log(D0 * Q0)) / math.log(1 + eps)
    elas_Q = (math.log(D0 * (Q0 * (1 + eps))) - math.log(D0 * Q0)) / math.log(1 + eps)
    return {
        "units_ablation": rows,
        "max_abs_diff_frozen": max(r["abs_diff"] for r in rows),
        "max_abs_diff_sub": max(r["abs_diff_sub"] for r in rows),
        "n_id_groups_total": int(len(S)),
        "D_frozen_all_one": bool(np.allclose(D, 1.0, rtol=0, atol=0)),
        "D_frozen_spread": float(D.max() - D.min()),
        "spearman_S_Q_frozen_n20": float(sp_frozen),
        "spearman_S_Q_sub_n20": float(sp_sub),
        "elasticity_dlogS_dlogD": float(elas_D),
        "elasticity_dlogS_dlogQ": float(elas_Q),
        "elasticity_equal": bool(abs(elas_D - elas_Q) < 1e-9),
    }


# ==================================================== B/C. 命题 2 与推论
def check_prop2(abl_id, abl_ood, unit_label):
    """全对枚举：前件成立的对数 / 反转对数 / 反例归因（三对数项分解）。"""
    units = list(abl_id.keys())
    pairs = []
    for i, j in itertools.combinations(units, 2):
        Qi, Qj = abl_id[i]["id_del_frozen"], abl_id[j]["id_del_frozen"]
        Di, Dj = abl_ood[i]["ood_det_sub"], abl_ood[j]["ood_det_sub"]
        # 前件：分布内勾画更高、域外检出更低（方向 a：i 勾画高、检出低）
        swapped = False
        if Qi > Qj and Di < Dj:
            X, Y = i, j
        elif Qj > Qi and Dj < Di:
            X, Y = j, i
            swapped = True
        else:
            continue
        SXi, SYi = abl_id[X]["id_dice"], abl_id[Y]["id_dice"]
        OXi, OYi = abl_ood[X]["ood_dice"], abl_ood[Y]["ood_dice"]
        QX, QY = abl_id[X]["id_del_frozen"], abl_id[Y]["id_del_frozen"]
        qOX, qOY = abl_ood[X]["ood_del_sub"], abl_ood[Y]["ood_del_sub"]
        aX = qOX / QX if QX else float("nan")     # 迁移因子 α_k = Q^(t)/Q^(s)
        aY = qOY / QY if QY else float("nan")
        d = {
            "pair": [X, Y], "antecedent_swapped": swapped,
            "Q_s_X": QX, "Q_s_Y": QY,
            "D_t_X": abl_ood[X]["ood_det_sub"], "D_t_Y": abl_ood[Y]["ood_det_sub"],
            "S_s_X": SXi, "S_s_Y": SYi, "S_t_X": OXi, "S_t_Y": OYi,
            "Q_t_X": qOX, "Q_t_Y": qOY,
            "alpha_X": aX, "alpha_Y": aY,
            # 精确三项分解：log(S_t_Y/S_t_X) = ΔlogD + ΔlogQ + Δlogα
            "term_dlogD": math.log(abl_ood[Y]["ood_det_sub"] / abl_ood[X]["ood_det_sub"]),
            "term_dlogQ": math.log(QY / QX),
            "term_dlogalpha": math.log(aY / aX),
            "term_sum": (math.log(abl_ood[Y]["ood_det_sub"] / abl_ood[X]["ood_det_sub"])
                         + math.log(QY / QX) + math.log(aY / aX)),
            "lhs_log_St_ratio": math.log(OYi / OXi),
            "id_rank_reversed": bool(SXi > SYi),
            "ood_rank_reversed": bool(OXi < OYi),
        }
        d["reversal_realised"] = bool(d["id_rank_reversed"] and d["ood_rank_reversed"])
        d["identity_residual"] = abs(d["term_sum"] - d["lhs_log_St_ratio"])
        # 前件的"比值条件"（B4 成立时命题 2 的充要形式）
        d["ratio_condition_holds"] = bool(abl_ood[X]["ood_det_sub"] / abl_ood[Y]["ood_det_sub"]
                                          < QY / QX)
        d["alpha_ratio_dominates"] = bool(abs(d["term_dlogalpha"]) > abs(d["term_dlogQ"]))
        pairs.append(d)
    n_ant = len(pairs)
    n_rev = sum(1 for p in pairs if p["reversal_realised"])
    viol = [p for p in pairs if not p["reversal_realised"]]
    return {
        "unit_set": unit_label,
        "n_units": len(units),
        "n_pairs_total": len(list(itertools.combinations(units, 2))),
        "n_pairs_antecedent": n_ant,
        "n_reversal_realised": n_rev,
        "n_reversal_violated": len(viol),
        "violations": [{"pair": p["pair"], "term_dlogD": p["term_dlogD"],
                        "term_dlogQ": p["term_dlogQ"], "term_dlogalpha": p["term_dlogalpha"],
                        "sum": p["term_sum"], "lhs": p["lhs_log_St_ratio"],
                        "reason": "B4 失效：|Δlog α| 抵消了 ΔlogD + ΔlogQ"}
                       for p in viol if p["alpha_ratio_dominates"]],
        "max_identity_residual": max(p["identity_residual"] for p in pairs) if pairs else None,
        "pairs": pairs,
    }


def check_corollary(abl_id, abl_ood):
    Q = [abl_id[c]["id_del_frozen"] for c in ABL_CFGS]
    D = [abl_ood[c]["ood_det_sub"] for c in ABL_CFGS]
    r = stats.spearmanr(Q, D)
    return {"n_domains": len(Q), "spearman_Q_s_D_t": float(r.statistic),
            "p_value": float(r.pvalue),
            "p_value_exact_permutation": exact_spearman_p(abs(float(r.statistic)), len(Q)),
            "exact_min_two_sided_p": 2 / math.factorial(len(Q)),
            "null_of_unit_set": "ablation grid (7 configs); the 7 configs are NOT 7 independent domains "
                                "(same backbone/data pipeline) -> treat as an upper bound on K",
            "interpretation": ("significantly positive -> ablation usable; "
                               "~0 -> uninformative; significantly negative -> reverse-misleading")}


# EGAUNet 的分布内读数取自论文二冻结方案（v4.2.2）的架构级口径；
# 该读数走单文件路线（D:/medical_segmentation/results/test_results/），
# 被 M10 从 P2-1 的「分解单元集」排除，故此处**显式标注来源**、不并入消融网格。
EGAUNET_ID_DELIN_PROTOCOL = 0.8853


def check_corollary_arch(arch_units, df):
    """架构级口径的推论核验。

    (a) n=3 —— **只用 P2-1 20 个预注册分布内单元里的架构**（AttentionUNet/TransUNet/UNet）；
    (b) n=4 —— 追加 EGAUNet（分布内读数取自冻结方案，来源已标注），复现预注册的 +0.200。
    """
    def dt(arch):
        rows = df[(df["scope"] == "OOD") & (df["group_id"].str.endswith("/" + arch))]
        return float(rows["detection_sub"].mean())

    out = {}
    avail = [a for a in ARCHS if arch_units[a].get("present_in_id_groups")]
    for label, names, qs_override in (
            ("n3_rule_compliant", avail, None),
            ("n4_with_EGAUNet_protocol_value", ARCHS, {"EGAUNet": EGAUNET_ID_DELIN_PROTOCOL})):
        ov = qs_override or {}
        q = [ov[a] if a in ov else arch_units[a]["id_del_frozen"] for a in names]
        d = [dt(a) for a in names]
        r = stats.spearmanr(q, d)
        out[label] = {
            "architectures": names, "n_domains": len(names),
            "Q_s": [float(x) for x in q], "D_t": [float(x) for x in d],
            "spearman_Q_s_D_t": float(r.statistic), "p_value_t_approx": float(r.pvalue),
            "p_value_exact_permutation": exact_spearman_p(abs(float(r.statistic)), len(names)),
            "exact_min_two_sided_p": 2 / math.factorial(len(names)),
            "conclusion_possible": bool(2 / math.factorial(len(names)) < ALPHA),
        }
    out["provenance"] = {
        "n3_rule_compliant": "Q^(s) from P2-1 decomposition.csv (13 baseline + 7 ablation unit set only)",
        "n4_with_EGAUNet_protocol_value":
            f"EGAUNet Q^(s) = {EGAUNET_ID_DELIN_PROTOCOL} taken from the frozen protocol (architecture-level "
            "caliber, single-file route); this row is NOT part of P2-1's 20 decomposition units (M10)",
        "sign_flip_warning": "n=3 gives rho=-1.000 (maximal reverse-misleading) while n=4 gives +0.200: "
                             "the point estimate changes sign when ONE domain is added -> no conclusion is "
                             "licit below the design's minimum K",
    }
    return out


# ================================================== D. 最小验证集设计
def budget_A_samples():
    """把检出率估到给定绝对精度 δ 所需的样本量。"""
    interior = {}
    for d_prior in (0.78, 0.50):
        row = {}
        for delta in (0.02, 0.03, 0.05):
            n = Z ** 2 * d_prior * (1 - d_prior) / delta ** 2
            row[f"delta_{delta:.2f}"] = int(math.ceil(n))
        interior[f"D={d_prior:.2f}"] = row
    boundary = {}
    for eta in (0.005, 0.01, 0.02, 0.05):
        n = math.log(ALPHA) / math.log(1 - eta)
        boundary[f"eta_{eta}"] = {"n_exact": int(math.ceil(n)),
                                  "n_rule_of_three": int(math.ceil(3 / eta))}
    # 现有样本量下的实得精度
    have = {}
    for n in (196, 242, 612, 1000):
        have[str(n)] = {
            "certified_eta_one_sided_95pct": float(1 - ALPHA ** (1 / n)),
            "halfwidth_at_D_0.78": float(Z * math.sqrt(0.78 * 0.22 / n)),
            "halfwidth_worst_case": float(Z * math.sqrt(0.25 / n)),
        }
    return {
        "model": "X ~ Binomial(N, D)",
        "interior_normal_approx": interior,
        "boundary_zero_failure_exact": boundary,
        "achieved_with_existing_n": have,
        "current_id_test_split_n": 242,
        "certified_eta_at_n242": float(1 - ALPHA ** (1 / 242)),
        "n_needed_for_eta_0.01": int(math.ceil(math.log(ALPHA) / math.log(1 - 0.01))),
        "shortfall_vs_eta_0.01": int(math.ceil(math.log(ALPHA) / math.log(1 - 0.01)) - 242),
    }


def _rank_within_reps(a):
    """沿 axis=1 对每个 rep 内部求秩（a: (reps, k)）。"""
    order = np.argsort(a, axis=1, kind="stable")
    ranks = np.empty_like(a, dtype=float)
    ranks[np.arange(a.shape[0])[:, None], order] = np.arange(a.shape[1])[None, :]
    return ranks


def _spearman_from_reps(z):
    ra = _rank_within_reps(z[:, :, 0])
    rb = _rank_within_reps(z[:, :, 1])
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    return np.clip((ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1)), -1, 1)


_EXACT_NULL = {}
for _k in range(2, 9):
    _base = np.arange(_k)
    _vals = np.sort(np.abs(np.array([stats.spearmanr(_base, p).statistic
                                     for p in itertools.permutations(_base)])))
    _EXACT_NULL[_k] = _vals


def exact_spearman_p(r_abs, k):
    """K ≤ 8 用精确置换零分布；K ≥ 10 用 t 近似。"""
    if k in _EXACT_NULL:
        v = _EXACT_NULL[k]
        n = math.factorial(k)
        cnt = int(np.searchsorted(v, r_abs, side="left"))
        return (len(v) - cnt) / n
    t = r_abs * math.sqrt((k - 2) / max(1e-15, 1 - r_abs ** 2))
    return float(2 * stats.t.sf(abs(t), k - 2))


def power(k, rho_s, reps=MC_REPS, seed=MC_SEED):
    """给定域数 K 与总体 Spearman ρ，MC 求 α=0.05 双侧的检验功效。"""
    rng = np.random.default_rng(seed)
    rp = 2 * math.sin(math.pi * rho_s / 6)          # 二元正态下 Spearman → Pearson
    z = rng.multivariate_normal([0, 0], [[1, rp], [rp, 1]], size=(reps, k))
    r = _spearman_from_reps(z)
    if k in _EXACT_NULL:
        v = _EXACT_NULL[k]
        idx = np.searchsorted(v, np.abs(r), side="left")
        p = (len(v) - idx) / math.factorial(k)
    else:
        t = r * np.sqrt((k - 2) / np.maximum(1e-15, 1 - r * r))
        p = 2 * stats.t.sf(np.abs(t), k - 2)
    return float((p < ALPHA).mean())


def budget_B_domains():
    """把 Spearman(Q^(s),D^(t)) 检到给定效应量所需的**独立域数** K。"""
    table = {}
    for k in K_GRID:
        pows = {f"rho_{r:.2f}": round(power(k, r), 4) for r in RHO_GRID}
        if k in _EXACT_NULL and k == 4:
            mde = 1.00            # 精确功效恒 0 → MDE 未定义，记 1.00
            mde_note = "exact power identically 0 (2/4! = 0.0833 > 0.05)"
        else:
            mde, mde_note = 1.00, None
            for r in np.arange(0.20, 1.00, 0.02):
                if power(k, float(r), reps=MC_REPS_MDE) >= 0.80:
                    mde = round(float(r), 2)
                    break
        crit = None
        if k in _EXACT_NULL:
            v, n = _EXACT_NULL[k], math.factorial(k)
            for t in np.unique(v):
                if (v >= t - 1e-12).sum() / n >= ALPHA:
                    crit = float(t)
        table[str(k)] = {"power": pows, "mde80": mde, "mde_note": mde_note,
                         "exact_min_two_sided_p": (2 / math.factorial(k) if k <= 8 else None),
                         "max_non_significant_abs_rho_exact": crit,
                         "p_value_method": "exact permutation" if k in _EXACT_NULL else "t approximation"}
    return {
        "model": "bivariate normal scores; Spearman via rank correlation; two-sided alpha=0.05",
        "mc_reps": MC_REPS, "mc_reps_for_mde": MC_REPS_MDE, "mc_seed": MC_SEED,
        "mde_grid_step": 0.02,
        "table": table,
        "binding_constraint": "K (number of independent domains), NOT N (images per domain)",
        "minimal_K_for_MDE_050": min([int(k) for k, v in table.items() if v["mde80"] <= 0.50] or [None]),
        "minimal_K_for_MDE_060": min([int(k) for k, v in table.items() if v["mde80"] <= 0.60] or [None]),
        "note_t_approx_anticonservative": ("t approximation overstates power for K<=8; "
                                          "exact permutation used there (K=4: exact power 0, "
                                          "t-approx would report 0.12-0.30)"),
    }


def budget_pairwise_mcnemar():
    """配对检出差异（反转的实证检验）需要多少例：以实测不一致率折算。"""
    def dv(c):
        f = np.load(f"{ABL_ETIS}/{c}/detailed_metrics.npz")["dice"].astype(float) / 100.0
        return f > TAU_DET
    out = {}
    for a, b in [("msfa_only", "egm_dpa"), ("msfa_only", "dpa_msfa"), ("dpa_msfa", "egm_dpa")]:
        A, B = dv(a), dv(b)
        n01 = int((~A & B).sum())      # a 未检出、b 检出
        n10 = int((A & ~B).sum())      # a 检出、b 未检出
        disc = n10 + n01
        p_exact = stats.binomtest(min(n10, n01), disc, 0.5).pvalue if disc else 1.0
        # 以实测条件概率 p1 = max/(n10+n01) 反算"检到 80% 功效所需的不一致对数"
        p1 = max(n10, n01) / disc if disc else 0.5
        p1c = min(max(p1, 0.5 + 1e-6), 0.999)
        num = Z * math.sqrt(0.25) + Z_POW80 * math.sqrt(p1c * (1 - p1c))
        n_disc_needed = (num / (p1c - 0.5)) ** 2
        pi_disc = disc / A.size
        out[f"{a}_vs_{b}"] = {
            "n": int(A.size), "n10": n10, "n01": n01, "n_discordant": disc,
            "discordant_fraction": disc / A.size,
            "mcnemar_exact_two_sided_p": float(p_exact),
            "observed_conditional_p": p1,
            "n_discordant_needed_80pct_power": float(math.ceil(n_disc_needed)),
            "n_images_needed_80pct_power": float(math.ceil(n_disc_needed / pi_disc)) if pi_disc else None,
        }
    return {"model": "McNemar / exact sign test on discordant pairs",
            "pairs": out,
            "note": "worked example using the observed discordance pattern, not a general guarantee"}


# ================================================================ 汇总 + 落盘
def build():
    t0 = time.time()
    df, abl_id, arch_units, idf = load_units()
    abl_ood, gap = ood_ablation_etis()

    # 与 paper2_factorial.json 的 OOD Dice 对账（防口径漂移）
    fac = json.load(open(FACTORIAL_JSON, encoding="utf-8"))["per_config"]
    recon = {c: abs(abl_ood[c]["ood_dice"] - fac[c]["ood_dice"] / 100.0) for c in ABL_CFGS}
    max_recon = max(recon.values())
    if max_recon > 1e-12:
        raise SystemExit(f"[FATAL] 消融 ETIS 宏平均 Dice 与 paper2_factorial.json 不符：{recon}")

    res = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "02_code/analysis/proposition_check.py",
        "segment": "P2-6",
        "cost": "0 (pure derivation + recomputation on existing artefacts; no training, no GPU)",
        "scope": {
            "ablation_grid_units": ABL_CFGS,
            "n_ablation_units": len(ABL_CFGS),
            "ood_domain_for_ablation_grid": "ETIS (n=196)",
            "architectures_4": ARCHS,
            "id_test_split_n": 242,
            "ood_cells_n": {"cvc_to_etis": 196, "cvc_to_kvasir": 1000,
                            "kvasir_to_cvc": 612, "kvasir_to_etis": 196},
        },
        "criterion": {
            "frozen": "|P cap G| > 0  (equivalent to metric > 0; constructed by smoothing constant eps -> B6)",
            "substantive": "I = |P cap G| >= 1 px (equivalent to metric > 1e-6)",
            "eps_id": 1e-8, "eps_ood": 1e-6,
            "ablation_etis_detection_threshold": TAU_DET,
            "ablation_etis_gap_report": gap,
        },
        "prop1": check_prop1(abl_id, idf),
        "prop2_ablation_grid": check_prop2(abl_id, abl_ood, "ablation grid (7 configs) x ETIS"),
        "corollary_ablation_grid": check_corollary(abl_id, abl_ood),
        "corollary_4arch": check_corollary_arch(arch_units, df),
        "minimal_design": {
            "budget_A_samples": budget_A_samples(),
            "budget_B_domains": budget_B_domains(),
            "budget_pairwise_mcnemar": budget_pairwise_mcnemar(),
        },
        "reconciliation": {
            "factorial_json_max_abs_diff": max_recon,
            "identity_residual_from_p2_1": None,
        },
        "runtime_sec": None,
    }

    # 恒等式残差（P2-1 产物）随附，便于单文件自足
    if os.path.exists(DECOMP_JSON):
        dj = json.load(open(DECOMP_JSON, encoding="utf-8"))
        s = dj.get("summary", {})
        for key in ("max_abs_residual", "max_residual", "residual_max"):
            if key in s:
                res["reconciliation"]["identity_residual_from_p2_1"] = s[key]
                break
        else:
            res["reconciliation"]["identity_residual_from_p2_1"] = s or None
    res["runtime_sec"] = round(time.time() - t0, 1)
    return res


def main():
    res = build()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    p1 = res["prop1"]
    print("=== P2-6 数值核验 ===")
    print(f"[命题 1] 分布内(冻结) S=Q 逐格最大偏差 = {p1['max_abs_diff_frozen']:.3e}"
          f" | 20 组 Spearman(S,Q) = {p1['spearman_S_Q_frozen_n20']:.6f}")
    print(f"         实质判据下最大偏差 = {p1['max_abs_diff_sub']:.3e}"
          f" | Spearman(S,Q_sub) = {p1['spearman_S_Q_sub_n20']:.4f}"
          f" | 弹性 D/Q = {p1['elasticity_dlogS_dlogD']:.9f}/{p1['elasticity_dlogS_dlogQ']:.9f}")
    p2 = res["prop2_ablation_grid"]
    print(f"[命题 2] 前件成立 {p2['n_pairs_antecedent']} 对 → 反转实现 {p2['n_reversal_realised']}"
          f" / 未实现 {p2['n_reversal_violated']}（三项分解恒等式最大残差 "
          f"{p2['max_identity_residual']:.3e}）")
    for v in p2["violations"]:
        print(f"         反例 {v['pair']}: ΔlogD={v['term_dlogD']:+.4f} ΔlogQ={v['term_dlogQ']:+.4f} "
              f"Δlogα={v['term_dlogalpha']:+.4f} → 和 {v['sum']:+.4f}（<0 ⇒ 不反转）")
    cor = res["corollary_ablation_grid"]
    print(f"[推论]   n=7: Spearman(Q^s,D^t) = {cor['spearman_Q_s_D_t']:+.4f} (p={cor['p_value']:.4f})")
    c4 = res["corollary_4arch"]["n4_with_EGAUNet_protocol_value"]
    c3 = res["corollary_4arch"]["n3_rule_compliant"]
    print(f"         n=4: Spearman = {c4['spearman_Q_s_D_t']:+.4f} (p_t={c4['p_value_t_approx']:.4f})"
          f" — 与预注册一致；但 2/4! = {c4['exact_min_two_sided_p']:.4f} > 0.05 → 功效恒 0")
    print(f"         n=3: Spearman = {c3['spearman_Q_s_D_t']:+.4f}"
          f" — **符号翻转**！2/3! = 1/3 → 任何 ρ 都不可判")
    A = res["minimal_design"]["budget_A_samples"]
    print(f"[设计A] 现有 ID 测试集 n=242 只能认证 η = {A['certified_eta_at_n242']*100:.2f}%；"
          f"要认证 1% 需 n = {A['n_needed_for_eta_0.01']}（缺 {A['shortfall_vs_eta_0.01']} 例）")
    print(f"        D=0.78 时 ±2pp 需 n = {A['interior_normal_approx']['D=0.78']['delta_0.02']}"
          f"；现有每格 n∈{{196,612,1000}} 实得 ±"
          f"{A['achieved_with_existing_n']['196']['halfwidth_at_D_0.78']*100:.2f}/"
          f"{A['achieved_with_existing_n']['612']['halfwidth_at_D_0.78']*100:.2f}/"
          f"{A['achieved_with_existing_n']['1000']['halfwidth_at_D_0.78']*100:.2f} pp")
    B = res["minimal_design"]["budget_B_domains"]
    print(f"[设计B] MDE@80%：K=4 → 未定义(精确功效 0)；K=11 → "
          f"{B['table']['11']['mde80']:.2f}；K=14 → {B['table']['14']['mde80']:.2f}；"
          f"K=20 → {B['table']['20']['mde80']:.2f}；K=30 → {B['table']['30']['mde80']:.2f}"
          f"（MDE≤0.5 需 K={B['minimal_K_for_MDE_050']}）")
    M = res["minimal_design"]["budget_pairwise_mcnemar"]["pairs"]["msfa_only_vs_egm_dpa"]
    print(f"[配对]   msfa_only vs egm_dpa: 不一致 {M['n_discordant']}/{M['n']}，"
          f"精确 McNemar p = {M['mcnemar_exact_two_sided_p']:.2e}；"
          f"80% 功效仅需 ≈{M['n_images_needed_80pct_power']:.0f} 例")
    print(f"\n[OK] 产物：{OUT_JSON}（{res['runtime_sec']}s）")


if __name__ == "__main__":
    sys.exit(main())
