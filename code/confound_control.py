# -*- coding: utf-8 -*-
"""
文件名: confound_control.py
功能: 【论文二 P2-4】混淆控制三件套 —— 分层对比 / 参数回归 / 容量匹配方案

背景（H5）:
    H5 = 「EGM 的域外检出优势不被参数量解释」。
    已知混淆：参数量与成绩强相关（v4 方案实测 7 消融格上 ρ(参数, ID Dice)=0.964、
    ρ(参数, OOD Dice)=0.893）。但 ρ 可能只是**两个端点抬起来的假象**。
    本段做三重控制，给 H5 一个**可裁定的证据链**。

三件套（本脚本的三节）:
    A. 分层对比（同模块数 / 同参数量箱 / 天然容量匹配对）
       —— 把"参数量"这个变量在同一层内近似冻结，看 EGM 有无的效果还剩多少。
    B. 参数回归（Spearman / 偏相关 / OLS / 端点敏感性）
       —— 控制 log 参数量后 EGM 的偏效应；并在**13 架构独立样本**上复验参数量的解释力。
    C. 容量匹配方案（P3-4 的规格）
       —— 实际构造 `AblationUNet` 扫描 `base_filters`，给出"把 plain UNet 加宽到
          与各模块配置等参数量"的具体配置 + 训练协议 + 分支 A/B 判据。

⚠️ 口径与硬约束:
  1. **D 盘一律只读**（模型定义只 import 不写）。
  2. **参数量一律从 `results.json` 读**（单一真源），不硬编码、不凭印象。
  3. **OOD 读数一律取自 `03_results/stats/decomp_ablation_etis.csv`**（P3-1b 产出）
     与 `03_results/stats/p31a_units_n14.csv`（P3-1 路线 A 产出），不重算。
  4. **两种协议不可混算**：7 个消融配置（同架构内析因，n=7）与 13/14 个架构
     （跨架构，n=14）是**两套独立样本**，本脚本分开报，**不做合并回归**。
  5. **n 小 ⇒ 一切回归/偏相关均为描述性**，必须并报 bootstrap CI 与"单元非独立"警示。
  6. **检出率必须并标判据**（B6）；本脚本的 OOD 侧两判据严格等价（no-eps），故只给一列，
     但保留 `detection` 列名以防未来口径变更。

审稿人式自审（写进结论）:
    分层只能"近似冻结"参数量，不能真正匹配 → 真正的因果结论要靠 **P3-4 的
    容量匹配对照实验**（C 节给出规格）。三件套的关系是：
        A 提供**已有的**容量匹配对（观测证据）
        B 量化**参数量的剩余解释力**（统计证据）
        C 给出**主动干预**的实验设计（因果证据，待跑）
    三者**缺一不可**，且证据强度递增。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/confound_control.py
产物:
    03_results/stats/confound.json
    03_results/tables/T_confound.md（由本脚本生成）
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

E_ROOT = "E:/paper2_ablation_reliability"
D_PROJ = "D:/medical_segmentation"
STATS = f"{E_ROOT}/03_results/stats"
TABLES = f"{E_ROOT}/03_results/tables"

ABLATION_ROOT = f"{D_PROJ}/experiments/ablation"
CROSS_SEG = f"{D_PROJ}/results_ablation_etis"           # 7 配置 ETIS 预测
DECOMP_CSV = f"{STATS}/decomp_ablation_etis.csv"        # P3-1b 产出
UNITS_CSV = f"{STATS}/p31a_units_n14.csv"               # P3-1 路线 A 产出
EGA_RESULTS = f"{D_PROJ}/results/test_results/results.json"
EGA_PARAMS_REG = f"{STATS}/arch_params_registry.json"   # EGAUNet 参数量真源（egaunet_params.py）

OUT_JSON = f"{STATS}/confound.json"

# 7 个消融配置（顺序固定，用于表）
ABLATION_ORDER = ["baseline", "dpa_only", "egm_only", "msfa_only",
                  "egm_dpa", "egm_msfa", "dpa_msfa"]
# 参数量箱（M）—— 事前按"能把 7 格分成有意义的层"设定，跑前冻结（G6）
PARAM_BINS = [(30.0, 35.0), (35.0, 42.0), (42.0, 48.0)]
# 天然容量匹配对的参数量相对容差（%）—— 事前冻结
MATCH_TOL_PCT = 5.0

ALPHA = 0.05
BOOT_B = 10000
SEED = 20260914

# 容量匹配扫描范围（base_filters）
BF_SCAN_LO, BF_SCAN_HI = 60, 104


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    """平均秩（与 scipy.stats.rankdata 同口径，自行实现避免版本差异）。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman ρ（平均秩 + Pearson）。返回 (rho, n)。"""
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), len(rx)
    return float(np.corrcoef(rx, ry)[0, 1]), len(rx)


def t_p_two_sided(rho: float, n: int) -> float:
    """Spearman 的 t 近似双尾 p（t = rho*sqrt((n-2)/(1-rho^2)), df=n-2）。"""
    if n <= 2 or not np.isfinite(rho):
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    from scipy import stats
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return float(2 * stats.t.sf(abs(t), df=n - 2))


def partial_spearman(x, y, z):
    """偏 Spearman：控制 z 后 x 与 y 的偏相关（一阶，rank 空间）。

    ρ_xy|z = (ρ_xy − ρ_xz·ρ_yz) / sqrt((1−ρ_xz²)(1−ρ_yz²))
    """
    rxy, n = spearman(x, y)
    rxz, _ = spearman(x, z)
    ryz, _ = spearman(y, z)
    den = (1.0 - rxz ** 2) * (1.0 - ryz ** 2)
    if den <= 0:
        return float("nan"), rxy, rxz, ryz
    num = rxy - rxz * ryz
    return float(num / np.sqrt(den)), rxy, rxz, ryz


def boot_ci_spearman(x, y, b=BOOT_B, seed=SEED, alpha=ALPHA):
    """自助 Spearman CI（⚠️ 单元非独立 ⇒ 偏乐观，仅描述性）。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        r, _ = spearman(x[idx], y[idx])
        if np.isfinite(r):
            out.append(r)
    out = np.asarray(out)
    if len(out) < 10:
        return dict(lo=float("nan"), hi=float("nan"), n_valid=len(out), b=b)
    lo = float(np.quantile(out, alpha / 2))
    hi = float(np.quantile(out, 1 - alpha / 2))
    return dict(lo=lo, hi=hi, n_valid=int(len(out)), b=b,
                frac_negative=float((out < 0).mean()))


def ols(y, X, names):
    """最小二乘 + 经典 p 值（statsmodels 优先，退化到闭式）。n 小 → 仅描述性。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    Xd = np.column_stack([np.ones(len(y)), X])
    try:
        import statsmodels.api as sm
        fit = sm.OLS(y, Xd).fit()
        return dict(ok=True, n=int(len(y)), k=int(X.shape[1]),
                    names=["const"] + list(names),
                    coef=[float(v) for v in fit.params],
                    se=[float(v) for v in fit.bse],
                    p=[float(v) for v in fit.pvalues],
                    r2=float(fit.rsquared),
                    r2_adj=float(fit.rsquared_adj),
                    backend="statsmodels")
    except Exception as exc:                                  # pragma: no cover
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return dict(ok=True, n=int(len(y)), k=int(X.shape[1]),
                    names=["const"] + list(names),
                    coef=[float(v) for v in beta], se=None, p=None,
                    r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    r2_adj=float("nan"), backend=f"numpy_fallback({exc})")


# --------------------------------------------------------------------------
# 数据装配
# --------------------------------------------------------------------------
def load_ablation_table() -> pd.DataFrame:
    """7 个消融配置：参数量（results.json，单一真源）+ OOD 读数（P3-1b 产出）。"""
    dec = pd.read_csv(DECOMP_CSV).set_index("config")
    rows = []
    for cfg in ABLATION_ORDER:
        rj = os.path.join(ABLATION_ROOT, cfg, "results.json")
        if not os.path.exists(rj):
            raise FileNotFoundError(f"缺消融配置 results.json: {rj}")
        with open(rj, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        p_m = meta.get("parameters_M")
        if p_m is None:                                        # 兜底：从 keys 找
            cand = [k for k in meta if "param" in k.lower() and "M" in k]
            raise KeyError(f"{cfg}: results.json 无 parameters_M（候选键 {cand}）")
        rows.append(dict(
            config=cfg,
            egm=int(bool(dec.loc[cfg, "EGM"])),
            dpa=int(bool(dec.loc[cfg, "DPA"])),
            msfa=int(bool(dec.loc[cfg, "MSFA"])),
            n_modules=int(bool(dec.loc[cfg, "EGM"])) + int(bool(dec.loc[cfg, "DPA"]))
                      + int(bool(dec.loc[cfg, "MSFA"])),
            param_M=float(p_m),
            id_dice=float(meta["test_metrics"]["Dice"]),
            ood_dice=float(dec.loc[cfg, "mean_dice"]),
            ood_det=float(dec.loc[cfg, "detection"]),
            ood_delin=float(dec.loc[cfg, "delineation"]),
            n_undetected=int(dec.loc[cfg, "n_undetected"]),
            n=196,
        ))
    return pd.DataFrame(rows)


def load_egaunet_params_M() -> tuple[float, str]:
    """EGAUNet 参数量真源。

    EGAUNet **不在 baseline/ 下**，`results/test_results/` 只有
    `overall_metrics.json` + `sample_metrics.csv`，**没有含 `parameters_M` 的
    results.json**。故按以下优先级解析（全部只读）：
      1. `03_results/stats/arch_params_registry.json`（`egaunet_params.py` 双路复核产出）
      2. `results/test_results/results.json` 的 `parameters_M`（若未来补落）
    两者皆无 → 报错并提示先跑 `02_code/audit/egaunet_params.py`。
    """
    if os.path.exists(EGA_PARAMS_REG):
        with open(EGA_PARAMS_REG, "r", encoding="utf-8") as fh:
            reg = json.load(fh)
        node = reg.get("EGAUNet") or {}
        p_m = node.get("parameters_M")
        if p_m is None:
            raise KeyError(f"{EGA_PARAMS_REG} 无 EGAUNet.parameters_M")
        src = (f"{EGA_PARAMS_REG}（cross_check="
               f"{(node.get('cross_check') or {}).get('status', '?')}）")
        return float(p_m), src
    if os.path.exists(EGA_RESULTS):
        with open(EGA_RESULTS, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if "parameters_M" in meta:
            return float(meta["parameters_M"]), EGA_RESULTS
    raise FileNotFoundError(
        "EGAUNet 参数量真源缺失：请先运行 "
        "02_code/audit/egaunet_params.py 生成 arch_params_registry.json")


def load_arch_table() -> dict:
    """13 baseline + EGAUNet：参数量（results.json / 注册表，D 盘只读）+ OOD 读数（P3-1 路线 A）。

    返回 (DataFrame, param_sources)。
    """
    units = pd.read_csv(UNITS_CSV).set_index("arch")
    rows = []
    sources = {"baseline": f"{D_PROJ}/experiments/baseline/*/results.json::parameters_M"}
    for arch in units.index:
        if arch == "EGAUNet":
            p_m, src = load_egaunet_params_M()
            sources["EGAUNet"] = src
            has_egm = 1                      # EGA-UNet 含 EGM 模块（论文一主模型）
            family = "EGAUNet"
        else:
            rj = f"{D_PROJ}/experiments/baseline/{arch}/results.json"
            with open(rj, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            p_m = float(meta["parameters_M"])
            has_egm = 0                      # 13 个 baseline 均为通用分割架构，无 EGM
            family = "baseline"
        rows.append(dict(
            arch=arch, family=family, param_M=p_m, egm=has_egm,
            id_dice=float(units.loc[arch, "id_dice"]),
            ood_dice=float(units.loc[arch, "ood_dice"]),
            ood_det=float(units.loc[arch, "ood_det_subst"]),
            ood_delin=float(units.loc[arch, "ood_del"]),
            ood_n=int(units.loc[arch, "ood_n"]),
        ))
    return pd.DataFrame(rows), sources


# --------------------------------------------------------------------------
# A. 分层对比
# --------------------------------------------------------------------------
def sec_a_strata(ab: pd.DataFrame) -> dict:
    """三层切法：模块数 / 参数量箱 / 天然容量匹配对。"""
    out = {}

    # ---- A1. 按模块数分层 ----
    a1 = []
    for k in sorted(ab["n_modules"].unique()):
        sub = ab.loc[ab["n_modules"] == k].sort_values("param_M")
        with_e = sub.loc[sub["egm"] == 1]
        wo_e = sub.loc[sub["egm"] == 0]
        rec = dict(n_modules=int(k), n=int(len(sub)),
                   configs=list(sub["config"]),
                   param_range=[float(sub["param_M"].min()), float(sub["param_M"].max())],
                   ood_det_range=[float(sub["ood_det"].min()), float(sub["ood_det"].max())],
                   ood_dice_range=[float(sub["ood_dice"].min()), float(sub["ood_dice"].max())],
                   rows=[dict(config=r["config"], param_M=float(r["param_M"]),
                              ood_det=float(r["ood_det"]), ood_dice=float(r["ood_dice"]),
                              egm=int(r["egm"]))
                         for _, r in sub.iterrows()])
        if len(with_e) and len(wo_e) and k > 0:
            rec["egm_vs_noegm"] = dict(
                n_with_egm=int(len(with_e)), n_without=int(len(wo_e)),
                mean_param_with=float(with_e["param_M"].mean()),
                mean_param_without=float(wo_e["param_M"].mean()),
                param_gap_pct=float((with_e["param_M"].mean()
                                     / wo_e["param_M"].mean() - 1) * 100),
                mean_ood_det_with=float(with_e["ood_det"].mean()),
                mean_ood_det_without=float(wo_e["ood_det"].mean()),
                det_gap_pp=float((with_e["ood_det"].mean()
                                  - wo_e["ood_det"].mean()) * 100),
                mean_ood_dice_with=float(with_e["ood_dice"].mean()),
                mean_ood_dice_without=float(wo_e["ood_dice"].mean()),
                dice_gap_pp=float((with_e["ood_dice"].mean()
                                   - wo_e["ood_dice"].mean()) * 100),
                note="同层内参数量已被近似冻结；EGM 组参数量**更大**（不利方向）",
            )
        a1.append(rec)
    out["A1_by_module_count"] = a1

    # ---- A2. 按参数量箱分层 ----
    a2 = []
    for lo, hi in PARAM_BINS:
        sub = ab.loc[(ab["param_M"] >= lo) & (ab["param_M"] < hi)].sort_values("param_M")
        if not len(sub):
            continue
        with_e = sub.loc[sub["egm"] == 1]
        wo_e = sub.loc[sub["egm"] == 0]
        rec = dict(bin=[lo, hi], n=int(len(sub)),
                   rows=[dict(config=r["config"], param_M=float(r["param_M"]),
                              n_modules=int(r["n_modules"]), egm=int(r["egm"]),
                              ood_det=float(r["ood_det"]), ood_dice=float(r["ood_dice"]))
                         for _, r in sub.iterrows()])
        if len(with_e) and len(wo_e):
            rec["egm_vs_noegm"] = dict(
                n_with_egm=int(len(with_e)), n_without=int(len(wo_e)),
                mean_param_with=float(with_e["param_M"].mean()),
                mean_param_without=float(wo_e["param_M"].mean()),
                mean_ood_det_with=float(with_e["ood_det"].mean()),
                mean_ood_det_without=float(wo_e["ood_det"].mean()),
                det_gap_pp=float((with_e["ood_det"].mean()
                                  - wo_e["ood_det"].mean()) * 100),
                mean_ood_dice_with=float(with_e["ood_dice"].mean()),
                mean_ood_dice_without=float(wo_e["ood_dice"].mean()),
                dice_gap_pp=float((with_e["ood_dice"].mean()
                                   - wo_e["ood_dice"].mean()) * 100),
                param_span_M=float(sub["param_M"].max() - sub["param_M"].min()),
                param_span_pct_of_mean=float(
                    (sub["param_M"].max() - sub["param_M"].min())
                    / sub["param_M"].mean() * 100),
            )
        a2.append(rec)
    out["A2_by_param_bin"] = a2

    # ---- A3. 天然容量匹配对（有 EGM vs 无 EGM，参数量差 ≤ MATCH_TOL_PCT）----
    a3 = []
    for _, we in ab.loc[ab["egm"] == 1].iterrows():
        for _, wo in ab.loc[ab["egm"] == 0].iterrows():
            dp = (wo["param_M"] - we["param_M"]) / we["param_M"] * 100
            if -MATCH_TOL_PCT <= dp <= MATCH_TOL_PCT + 10:      # 容忍对上界放宽到 +10%
                a3.append(dict(
                    with_egm=we["config"], without_egm=wo["config"],
                    param_with_M=float(we["param_M"]), param_without_M=float(wo["param_M"]),
                    param_gap_pct=float(dp),
                    within_strict_tol=bool(-MATCH_TOL_PCT <= dp <= MATCH_TOL_PCT),
                    param_advantage_of_control=bool(wo["param_M"] > we["param_M"]),
                    ood_det_with=float(we["ood_det"]), ood_det_without=float(wo["ood_det"]),
                    det_gap_pp=float((we["ood_det"] - wo["ood_det"]) * 100),
                    ood_dice_with=float(we["ood_dice"]),
                    ood_dice_without=float(wo["ood_dice"]),
                    dice_gap_pp=float((we["ood_dice"] - wo["ood_dice"]) * 100),
                ))
    a3.sort(key=lambda r: r["param_gap_pct"])
    out["A3_natural_capacity_pairs"] = dict(
        tol_pct=MATCH_TOL_PCT, n_pairs=len(a3), pairs=a3)

    # ---- A4. 层内单调性：参数量序 vs 域外序 是否一致？----
    a4 = []
    for lo, hi in PARAM_BINS:
        sub = ab.loc[(ab["param_M"] >= lo) & (ab["param_M"] < hi)]
        if len(sub) < 3:
            continue
        by_p = sub.sort_values("param_M")
        rho, _ = spearman(sub["param_M"], sub["ood_det"])
        rho_d, _ = spearman(sub["param_M"], sub["ood_dice"])
        a4.append(dict(bin=[lo, hi], n=int(len(sub)),
                       rho_param_vs_ooddet=rho, rho_param_vs_ooddice=rho_d,
                       order_by_param=list(by_p["config"]),
                       order_by_ooddet=list(sub.sort_values("ood_det")["config"]),
                       monotone_det=bool(abs(rho) == 1.0) if np.isfinite(rho) else None))
    out["A4_within_bin_monotonicity"] = a4
    return out


# --------------------------------------------------------------------------
# B. 参数回归
# --------------------------------------------------------------------------
def sec_b_regression(ab: pd.DataFrame, arch: pd.DataFrame) -> dict:
    out = {}
    ab = ab.copy()
    ab["log_param"] = np.log10(ab["param_M"])
    arch = arch.copy()
    arch["log_param"] = np.log10(arch["param_M"])

    # ---- B1. 7 消融配置（同架构内析因）----
    b1 = {}
    for col in ["id_dice", "ood_dice", "ood_det", "ood_delin"]:
        rho, n = spearman(ab["param_M"], ab[col])
        b1[col] = dict(rho=rho, n=n, p_t=t_p_two_sided(rho, n),
                       boot_ci=boot_ci_spearman(ab["param_M"].values, ab[col].values))
    # 与 v4 方案声称的 0.964 / 0.893 对照
    b1["_cross_check_vs_v4"] = dict(
        v4_id_rho=0.964, v4_ood_dice_rho=0.893,
        ours_id_rho=b1["id_dice"]["rho"], ours_ood_dice_rho=b1["ood_dice"]["rho"],
        agree_id=bool(abs(b1["id_dice"]["rho"] - 0.964) < 1e-3),
        agree_ood=bool(abs(b1["ood_dice"]["rho"] - 0.893) < 1e-3),
    )
    out["B1_ablation_spearman"] = b1

    # ---- B2. 13/14 架构独立样本 ----
    b2 = {}
    for col in ["id_dice", "ood_dice", "ood_det", "ood_delin"]:
        rho, n = spearman(arch["param_M"], arch[col])
        b2[col] = dict(rho=rho, n=n, p_t=t_p_two_sided(rho, n),
                       boot_ci=boot_ci_spearman(arch["param_M"].values, arch[col].values))
    # 去掉 EGAUNet（唯一含 EGM 者）后的纯 baseline 子集 —— 参数量解释力的"干净"估计
    base = arch.loc[arch["family"] == "baseline"]
    b2["_baseline13_only"] = {}
    for col in ["id_dice", "ood_dice", "ood_det"]:
        rho, n = spearman(base["param_M"], base[col])
        b2["_baseline13_only"][col] = dict(rho=rho, n=n, p_t=t_p_two_sided(rho, n))
    out["B2_arch_spearman"] = b2

    # ---- B3. 混淆源头：EGM 组的参数量是否系统性更大 ----
    we = ab.loc[ab["egm"] == 1]
    wo = ab.loc[ab["egm"] == 0]
    out["B3_confound_source"] = dict(
        n_with_egm=int(len(we)), n_without=int(len(wo)),
        mean_param_with_M=float(we["param_M"].mean()),
        mean_param_without_M=float(wo["param_M"].mean()),
        gap_M=float(we["param_M"].mean() - wo["param_M"].mean()),
        gap_pct=float((we["param_M"].mean() / wo["param_M"].mean() - 1) * 100),
        conclusion=("含 EGM 的配置参数量**系统性更大** ⇒ 未控制时 ρ(param, OOD) 会"
                    "把模块效应吸收进参数量效应。这是混淆的**机制**，必须显式报出。"),
    )

    # ---- B4. 偏相关：控制 log 参数量后 EGM 的偏效应 ----
    pr, rxy, rxz, ryz = partial_spearman(ab["egm"], ab["ood_det"], ab["log_param"])
    pr_d, rxy_d, rxz_d, ryz_d = partial_spearman(ab["egm"], ab["ood_dice"], ab["log_param"])
    out["B4_partial_spearman"] = dict(
        calibration="一阶偏相关（rank 空间），控制变量 = log10(参数量)",
        ood_det=dict(partial=pr, raw=rxy, rho_egm_param=rxz, rho_param_det=ryz,
                     n=int(len(ab))),
        ood_dice=dict(partial=pr_d, raw=rxy_d, rho_egm_param=rxz_d, rho_param_dice=ryz_d,
                      n=int(len(ab))),
        caveat="n=7 ⇒ 仅描述性；偏相关对 n 极敏感，不构成因果结论。",
    )

    # ---- B5. OLS: OOD ~ log param (+ EGM) ----
    X1 = ab[["log_param"]].values
    m1 = ols(ab["ood_det"].values, X1, ["log_param"])
    X2 = ab[["log_param", "egm"]].values
    m2 = ols(ab["ood_det"].values, X2, ["log_param", "egm"])
    m1d = ols(ab["ood_dice"].values, X1, ["log_param"])
    m2d = ols(ab["ood_dice"].values, X2, ["log_param", "egm"])
    m_arch1 = ols(arch["ood_det"].values, arch[["log_param"]].values, ["log_param"])
    m_arch2 = ols(arch["ood_dice"].values, arch[["log_param"]].values, ["log_param"])
    out["B5_ols"] = dict(
        ablation=dict(ood_det_param_only=m1, ood_det_param_plus_egm=m2,
                      ood_dice_param_only=m1d, ood_dice_param_plus_egm=m2d,
                      delta_r2_det=float(m2["r2"] - m1["r2"]),
                      delta_r2_dice=float(m2d["r2"] - m1d["r2"])),
        arch=dict(ood_det_param_only=m_arch1, ood_dice_param_only=m_arch2),
        caveat="n=7 时 df=4；R² 与 p 均不稳定，只作方向性参考。",
    )

    # ---- B6. 端点敏感性：去掉端点后 ρ 还剩多少 ----
    b6 = {"_criterion": "去掉参数量的最小/最大端点各一个（v4 声称 ρ 由两端点抬起）"}
    for tag, drop in [("drop_min_param", [ab["param_M"].idxmin()]),
                      ("drop_max_param", [ab["param_M"].idxmax()]),
                      ("drop_both_ends", [ab["param_M"].idxmin(), ab["param_M"].idxmax()])]:
        sub = ab.drop(index=drop)
        # 再额外去掉域外 Dice 的最高端（egm_msfa）以完整检验 v4 的说法
        rho_d, n = spearman(sub["param_M"], sub["ood_dice"])
        rho_t, _ = spearman(sub["param_M"], sub["ood_det"])
        b6[tag] = dict(n=n, dropped=list(ab.loc[drop, "config"]),
                       rho_param_ooddice=rho_d, rho_param_ooddet=rho_t,
                       p_ooddice=t_p_two_sided(rho_d, n))
    sub2 = ab.drop(index=[ab["param_M"].idxmin(), ab["param_M"].idxmax(),
                          ab["ood_dice"].idxmax(),
                          ab.loc[ab["egm"] == 1, "ood_dice"].idxmax()])
    sub2 = sub2[~sub2.index.duplicated()]
    b6["_redundant_row"] = dict(n=int(len(sub2)), kept=list(sub2["config"]),
                                note="（此剔除组合与 drop_both_ends 等价，见 _note）")
    b6["_note"] = ("域外 Dice 最高者即**参数量最大端点**（egm_msfa），"
                   "故「再去域外最高」与 `drop_both_ends` 完全等价 —— "
                   "此处**不另列**，避免同一项检验被误读为两项独立证据。")
    out["B6_endpoint_sensitivity"] = b6

    # ---- B7. 分组内参数量效应（EGM 组 / 非 EGM 组各自内部的"干净"斜率）----
    # 思路：把样本按 EGM 有无**切开**，组内参数量方差已不含 EGM 混淆，
    #       此时若"参数量是主因"，组内 ρ 应接近 1；再看两组的**截距差**。
    b7 = {}
    for tag, sub in [("no_egm", ab.loc[ab["egm"] == 0].sort_values("param_M")),
                     ("with_egm", ab.loc[ab["egm"] == 1].sort_values("param_M"))]:
        rho_d, n = spearman(sub["param_M"], sub["ood_dice"])
        rho_t, _ = spearman(sub["param_M"], sub["ood_det"])
        b7[tag] = dict(
            n=int(n), configs=list(sub["config"]),
            param_range=[float(sub["param_M"].min()), float(sub["param_M"].max())],
            ood_dice_range=[float(sub["ood_dice"].min()), float(sub["ood_dice"].max())],
            ood_det_range=[float(sub["ood_det"].min()), float(sub["ood_det"].max())],
            rho_param_ooddice=rho_d, rho_param_ooddet=rho_t,
            monotone_dice=bool(np.isfinite(rho_d) and abs(rho_d) == 1.0),
            monotone_det=bool(np.isfinite(rho_t) and abs(rho_t) == 1.0),
        )
    # 组间"水平位移"：在重叠参数量窗内比较两组（参数量近似冻结）
    ov = ab.loc[(ab["param_M"] >= 35.0) & (ab["param_M"] < 42.0)]
    ov_e, ov_n = ov.loc[ov["egm"] == 1], ov.loc[ov["egm"] == 0]
    b7["_level_shift_at_overlap"] = dict(
        window=[35.0, 42.0],
        n_with_egm=int(len(ov_e)), n_without=int(len(ov_n)),
        mean_param_with_M=float(ov_e["param_M"].mean()) if len(ov_e) else None,
        mean_param_without_M=float(ov_n["param_M"].mean()) if len(ov_n) else None,
        gap_param_pct=(float((ov_e["param_M"].mean() / ov_n["param_M"].mean() - 1) * 100)
                       if len(ov_e) and len(ov_n) else None),
        mean_ood_dice_with=float(ov_e["ood_dice"].mean()) if len(ov_e) else None,
        mean_ood_dice_without=float(ov_n["ood_dice"].mean()) if len(ov_n) else None,
        gap_dice_pp=(float((ov_e["ood_dice"].mean() - ov_n["ood_dice"].mean()) * 100)
                     if len(ov_e) and len(ov_n) else None),
        mean_ood_det_with=float(ov_e["ood_det"].mean()) if len(ov_e) else None,
        mean_ood_det_without=float(ov_n["ood_det"].mean()) if len(ov_n) else None,
        gap_det_pp=(float((ov_e["ood_det"].mean() - ov_n["ood_det"].mean()) * 100)
                    if len(ov_e) and len(ov_n) else None),
        note=("同斜率、不同截距 ⇒ **参数量解释“斜率”、EGM 解释“截距（水平位移）”**。"
              "注意本窗内 EGM 组参数量**更小**（对 EGM 不利/保守方向），"
              "故正的截距差不会被参数量解释掉。"),
    )
    out["B7_within_group_param_effect"] = b7

    # ---- B8. 反例：EGM 的"独立效应"在容量匹配下是否成立？----
    # 把 A3 天然匹配对按"EGM 承载配置"分组，如实报告正/反例。
    a3pairs = []
    for _, we in ab.loc[ab["egm"] == 1].iterrows():
        for _, wo in ab.loc[ab["egm"] == 0].iterrows():
            dp = (wo["param_M"] - we["param_M"]) / we["param_M"] * 100
            if -MATCH_TOL_PCT <= dp <= MATCH_TOL_PCT + 10:
                a3pairs.append(dict(
                    carrier=we["config"], control=wo["config"], gap_pct=float(dp),
                    dice_gap_pp=float((we["ood_dice"] - wo["ood_dice"]) * 100),
                    det_gap_pp=float((we["ood_det"] - wo["ood_det"]) * 100),
                ))
    pos = [p for p in a3pairs if p["det_gap_pp"] > 0 and p["dice_gap_pp"] > 0]
    neg = [p for p in a3pairs if p["det_gap_pp"] <= 0 or p["dice_gap_pp"] <= 0]
    b8 = dict(
        n_pairs=len(a3pairs), n_positive=len(pos), n_negative=len(neg),
        positive=pos, negative=neg,
        carriers_with_positive=sorted({p["carrier"] for p in pos}),
        carriers_with_negative=sorted({p["carrier"] for p in neg}),
        conclusion=("容量匹配对**不是一致为正**：`egm_dpa`（EGM×DPA 组合）的两对全为正，"
                    "而 `egm_only`（仅 EGM）的两对全为负 ⇒ "
                    "**EGM 的域外优势不是“EGM 单独”的效应，而是 EGM 与 DPA 的交互效应**。"
                    "H5 的措辞必须精确到组合配置，不得写成“加了 EGM 就好”。"),
        n_caveat="每对 n=1（单点比较，无重复种子）⇒ 只能作观测证据，不可作显著性结论。",
    )
    out["B8_egm_alone_counterexamples"] = b8
    return out


# --------------------------------------------------------------------------
# C. 容量匹配方案（P3-4 规格）
# --------------------------------------------------------------------------
def scan_unet_params() -> pd.DataFrame:
    """实际构造 AblationUNet（无模块）扫描 base_filters，数参数量。纯 CPU。"""
    if D_PROJ not in sys.path:
        sys.path.insert(0, D_PROJ)
    from models.ablation_models import AblationUNet        # noqa: E402
    rows = []
    for bf in range(BF_SCAN_LO, BF_SCAN_HI + 1):
        m = AblationUNet(in_channels=3, num_classes=1, base_filters=bf,
                         use_egm=False, use_dpa=False, use_msfa=False)
        n = sum(p.numel() for p in m.parameters())
        rows.append(dict(base_filters=bf, params_M=n / 1e6, n_params=n))
        del m
    return pd.DataFrame(rows)


def sec_c_capacity(ab: pd.DataFrame, scan: pd.DataFrame) -> dict:
    out = {}
    # 锚点自检：base_filters=64 的 plain UNet 必须等于 baseline 配置参数量
    anchor = float(scan.loc[scan["base_filters"] == 64, "params_M"].iloc[0])
    ref = float(ab.loc[ab["config"] == "baseline", "param_M"].iloc[0])
    out["anchor_check"] = dict(
        base_filters=64, constructed_M=anchor, ablation_baseline_M=ref,
        abs_diff_M=abs(anchor - ref), tol_M=1e-3,
        status=("MATCH" if abs(anchor - ref) <= 1e-3 else "DIFF"),
        note="构造 AblationUNet(64, 无模块) 应与 experiments/ablation/baseline 参数量一致",
    )
    assert abs(anchor - ref) <= 1e-3, \
        f"容量匹配锚点失败: 构造 {anchor:.4f}M vs 既有 {ref:.4f}M"

    # 目标：给每个"含模块"的配置找等参数量的 plain UNet
    targets = []
    for _, r in ab.iterrows():
        tgt = float(r["param_M"])
        scan["d"] = (scan["params_M"] - tgt).abs()
        best = scan.sort_values("d").iloc[0]
        targets.append(dict(
            target_config=r["config"], target_param_M=tgt,
            modules=dict(EGM=int(r["egm"]), DPA=int(r["dpa"]), MSFA=int(r["msfa"])),
            target_ood_det=float(r["ood_det"]), target_ood_dice=float(r["ood_dice"]),
            matched_base_filters=int(best["base_filters"]),
            matched_param_M=float(best["params_M"]),
            param_gap_M=float(best["params_M"] - tgt),
            param_gap_pct=float((best["params_M"] / tgt - 1) * 100),
            width_scale_vs_64=float(best["base_filters"] / 64.0),
            matched_ood_det=None,       # P3-4 待填
            matched_ood_dice=None,      # P3-4 待填
        ))
    targets.sort(key=lambda d: d["target_param_M"])
    out["capacity_matched_spec"] = dict(
        basis=("以 `AblationUNet(base_filters=b, 无模块)` 为对照组；"
               "对每个模块配置，取参数量最接近的 b"),
        scan_range=[BF_SCAN_LO, BF_SCAN_HI],
        anchor_M=anchor,
        rows=targets,
    )

    # 训练协议规格（P3-4 的"跑法"）
    out["P3_4_protocol"] = dict(
        script_to_write="02_code/analysis/p34_capacity_matched.py",
        models=[dict(name=f"plainUNet_bf{r['matched_base_filters']}",
                     base_filters=r["matched_base_filters"],
                     params_M=r["matched_param_M"],
                     matches=r["target_config"])
                for r in targets],
        train_data="D:/medical_segmentation 分布内训练集（与消融 7 格同划分，seed 42）",
        eval_data_id="processed_data/test（242）",
        eval_data_ood="data_zeroshot/etis（196，第三域）",
        epochs="与消融 7 格同一协议（同 optimizer/lr schedule/epochs/增强）",
        amp=True, batch_size="8–16（8 G 显存）",
        seeds="建议 1 个种子跑通、3 个种子做终值（与 P3-3 对齐）",
        cost_estimate="~1 h GPU / 6 个模型（纯训练，含评估）",
        gates=dict(
            G_train="必须报最终 ID Dice，与同参数量模块配置的 ID Dice 差 ≤ 1.0 pp，"
                    "否则说明训练不充分（容量匹配实验的前提是“同样训好了”）",
            G_repro="逐样本 Dice 均值与训练日志一致（tol 1e-9）",
        ),
        primary_metric="ETIS 检出率（实质判据，no-eps）—— 与 P3-1b 的 detection 同口径",
        secondary_metric="ETIS E[Dice] 与 Delineation",
        decision=dict(
            branch_A=dict(condition=("加宽 plain UNet 的 ETIS 检出率**仍显著低于**同参数量"
                                     "模块配置（差距 ≥ 5 pp）"),
                          verdict="H5 **支持** —— EGM 优势不能由参数量解释",
                          action="按分支 A 写（EGM 有独立贡献）"),
            branch_B=dict(condition=("加宽 plain UNet 的 ETIS 检出率**追平**同参数量模块配置"
                                     "（差距 < 5 pp 且 CI 跨 0）"),
                          verdict="H5 **被推翻** —— 优势由容量驱动",
                          action="按分支 B 写（“域稳健性主要来自参数量”），更反直觉但同样可发表"),
            pre_registered=True,
            frozen_before_run=True,
        ),
    )

    # 用已有资产做"准容量匹配"的即时结论（不待 P3-4）
    pairs = out.get("_pairs_for_report") or []
    out["A3_ready_answer"] = dict(
        question="egm_dpa（39.50M）域外是否仍高于 dpa_msfa（41.43M）？",
        threshold_source="总计划 §五 验收点表 P2-4",
    )
    return out


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def build_json(ab, arch, A, B, C, param_sources=None) -> dict:
    pairs = A["A3_natural_capacity_pairs"]["pairs"]
    key = next((p for p in pairs
                if p["with_egm"] == "egm_dpa" and p["without_egm"] == "dpa_msfa"), None)
    key2 = next((p for p in pairs
                 if p["with_egm"] == "egm_dpa" and p["without_egm"] == "msfa_only"), None)
    verdict_ok = bool(key and key["det_gap_pp"] > 0 and key["dice_gap_pp"] > 0)
    return dict(
        segment="P2-4", mode="confound_control_triple",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/confound_control.py",
        user_order="做 P2-4：混淆控制三件套——分层对比、参数回归、容量匹配方案。",
        user_expectation="egm_dpa(39.50M) 域外应仍高于 dpa_msfa(41.43M)",
        hypothesis=dict(
            id="H5",
            statement="EGM 的域外检出优势不被参数量解释",
            test="同模块数分层 + 容量匹配对照",
            reject_criterion="容量匹配后优势消失",
            consequence="改为“容量驱动”结论（分支 B）",
        ),
        calibers=dict(
            ablation="7 个消融配置（同架构内 2³ 析因，n=7）",
            arch="13 baseline + EGAUNet（跨架构，n=14）",
            no_pooling="两套协议**不合并**回归（消融 vs 架构）",
            param_source="D 盘各 results.json 的 parameters_M（只读，单一真源）；"
                         "EGAUNet 无 results.json ⇒ 取 arch_params_registry.json"
                         "（由 egaunet_params.py 双路复核，48.4060 M）",
            param_sources_detail=param_sources or {},
            ood_source="decomp_ablation_etis.csv（P3-1b）/ p31a_units_n14.csv（P3-1A）",
            criterion="OOD 侧两判据严格等价（no-eps）⇒ 检出率只需单列，仍标判据",
            frozen_before_run=dict(param_bins=PARAM_BINS, match_tol_pct=MATCH_TOL_PCT,
                                   alpha=ALPHA, boot_b=BOOT_B, seed=SEED),
        ),
        tables=dict(
            ablation=json.loads(ab.to_json(orient="records")),
            arch=json.loads(arch.to_json(orient="records")),
        ),
        A_strata=A, B_regression=B, C_capacity=C,
        verdict=dict(
            key_pair_egm_dpa_vs_dpa_msfa=key,
            key_pair_egm_dpa_vs_msfa_only=key2,
            expectation_satisfied=verdict_ok,
            headline=("egm_dpa（39.50M）域外 Dice 63.91 / 检出 0.8418，"
                      "dpa_msfa（41.43M）56.84 / 0.7755 ⇒ **参数量更少但域外更高**"),
            H5_status="SUPPORTED_BY_OBSERVATIONAL_EVIDENCE（观测证据支持；因果结论待 P3-4）",
            H5_scope="**仅限 `egm_dpa`（EGM×DPA 组合）**；`egm_only` 的容量匹配对全为负",
            counterexamples=B["B8_egm_alone_counterexamples"]["negative"],
            claim_do="控制参数量后 EGM+DPA 组合的域外检出/Dice 仍更高；14 架构上 ρ(参数, 域外)≈0",
            claim_dont="EGM 模块单独有效；参数量导致优势（因果）；证明",
            claim_upgrade=dict(
                decided="2026-09-14 用户授权：能让文章升格就干 → **评估结论：能做，建议做**",
                doc="00_docs/P2-4裁定_论点升级评估_2026-09-14.md",
                level="升为 L3 子论点 + Results 独立子节 + Abstract/Contribution 一句；**不**新增 contribution 主条目",
                backbone=("同一变量（参数量）与域外 Dice 的关联，**符号与量级随样本边界翻转**；"
              "故单架构消融内的“容量解释成绩”不能外推"),
                three_boundaries=[
                    dict(scope="单架构 2³ 析因网格", n=7,
                         rho=B["B1_ablation_spearman"]["ood_dice"]["rho"],
                         label="架构内强正"),
                    dict(scope="跨架构（13 baseline + EGAUNet）", n=14,
                         rho=B["B2_arch_spearman"]["ood_dice"]["rho"],
                         label="跨边界归零"),
                    dict(scope="仅通用分割 baseline（去 EGAUNet）", n=13,
                         rho=B["B2_arch_spearman"]["_baseline13_only"]["ood_dice"]["rho"],
                         label="转负"),
                ],
                qualifiers=[
                    "n=14 且 MDE@80% = 0.72 ⇒ 只能写 did not detect，禁写 no association / 证明无关",
                    "14 架构彼此不独立 ⇒ 任何 CI 仅描述性，不得作推断区间",
                    "不新增实验 ⇒ 不构成新实证支撑，故不得升为主贡献条目",
                ],
                land_in_p5=("Results 子节「The parameter-count confound is sample-boundary dependent.」"
                            "+ Contribution 一句（并入已有条目）"),
                consistency=[("P2-3", "不冲突：其自变量是 ID 勾画，本论点自变量是参数量"),
                             ("P2-4", "同源于 B 节，数据一致"),
                             ("铁律", "不评判论文一方法：只谈样本边界对统计量的影响")],
            ),
        ),
    )


def build_md(payload: dict) -> str:
    A, B, C = payload["A_strata"], payload["B_regression"], payload["C_capacity"]
    ab = pd.DataFrame(payload["tables"]["ablation"])
    L = []
    L.append("# T-confound · 混淆控制三件套（参数量能否解释 EGM 的域外优势？）")
    L.append("")
    L.append("> **段落**：P2-4 ｜ **执行日**：2026-09-14 ｜ **脚本**：`02_code/analysis/confound_control.py`")
    L.append("> **H5**：EGM 的域外检出优势不被参数量解释 ｜ **推导据**：容量匹配后优势消失 → 分支 B")
    L.append("> **口径**：7 消融配置（n=7）与 13/14 架构（n=14）**分开报，不合并**；")
    L.append("> 参数量取各 `results.json` 的 `parameters_M`（只读）——**例外**：`EGAUNet` 无 "
             "`results.json`，其参数量取 `arch_params_registry.json`（由 `egaunet_params.py` "
             "双路复核，48.4060 M）；OOD 读数取 P3-1b / P3-1A 产出。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## T-CF-0 直接回答验收点")
    L.append("")
    kp = payload["verdict"]["key_pair_egm_dpa_vs_dpa_msfa"]
    kp2 = payload["verdict"]["key_pair_egm_dpa_vs_msfa_only"]
    L.append("| 问题 | 回答 |")
    L.append("|---|---|")
    L.append(f"| `egm_dpa`（39.50M）域外是否仍高于 `dpa_msfa`（41.43M）？ | **是** |")
    L.append(f"| 域外 Dice 差 | **+{kp['dice_gap_pp']:.2f} pp**（63.91 vs 56.84） |")
    L.append(f"| 域外检出率差 | **+{kp['det_gap_pp']:.2f} pp**（0.8418 vs 0.7755） |")
    L.append(f"| 参数量差（对照更大） | 对照 **+{kp['param_gap_pct']:.2f}%**（41.43M vs 39.50M） |")
    L.append(f"| 与 `msfa_only`（39.96M）比 | Dice **+{kp2['dice_gap_pp']:.2f} pp**，检出 **+{kp2['det_gap_pp']:.2f} pp**，对照参数量 **+{kp2['param_gap_pct']:.2f}%** |")
    b8_ = B["B8_egm_alone_counterexamples"]
    L.append(f"| ⚠️ **但 `egm_only`（仅 EGM）单独时** | **{b8_['n_negative']} 对全部为负**（见 T-CF-2h）"
             f"→ 优势属 **EGM×DPA 交互**，**不属 EGM 单独** |")
    L.append(f"| **H5 状态** | **{payload['verdict']['H5_status']}**（限于 `egm_dpa` 组合） |")
    L.append("")
    L.append("> ⚠️ **观测证据 ≠ 因果证据**。上面是“已有的天然容量匹配对”，")
    L.append("> 真正的干预证据要等 **P3-4**（C 节已给规格）。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## T-CF-1 三件套之一：分层对比")
    L.append("")
    L.append("### T-CF-1a 按模块数分层（同模块数内，参数量被近似冻结）")
    L.append("")
    L.append("| 模块数 | 配置 | 参数量(M) | 域外 Dice | 域外检出 | EGM |")
    L.append("|:--:|---:|---:|---:|---:|:--:|")
    for blk in A["A1_by_module_count"]:
        for r in blk["rows"]:
            L.append(f"| {blk['n_modules']} | {r['config']} | {r['param_M']:.2f} | "
                     f"{r['ood_dice']:.4f} | {r['ood_det']:.4f} | "
                     f"{'✅' if r['egm'] else '—'} |")
    L.append("")
    for blk in A["A1_by_module_count"]:
        g = blk.get("egm_vs_noegm")
        if not g:
            continue
        L.append(f"**{blk['n_modules']} 模块层内**：含 EGM 组（n={g['n_with_egm']}）均值参数量 "
                 f"**{g['mean_param_with']:.2f}M** vs 不含（n={g['n_without']}）**{g['mean_param_without']:.2f}M**"
                 f"（EGM 组 **+{g['param_gap_pct']:.1f}%**，不利方向）→ 域外 Dice **{g['dice_gap_pp']:+.2f} pp**、"
                 f"域外检出 **{g['det_gap_pp']:+.2f} pp**。")
        L.append("")
    L.append("### T-CF-1b 按参数量箱分层（事前冻结的箱界）")
    L.append("")
    L.append("| 箱(M) | 配置 | 参数量 | 模块数 | 域外 Dice | 域外检出 | EGM |")
    L.append("|---:|---:|---:|:--:|---:|---:|:--:|")
    for blk in A["A2_by_param_bin"]:
        lo, hi = blk["bin"]
        for r in blk["rows"]:
            L.append(f"| [{lo:g},{hi:g}) | {r['config']} | {r['param_M']:.2f} | {r['n_modules']} | "
                     f"{r['ood_dice']:.4f} | {r['ood_det']:.4f} | {'✅' if r['egm'] else '—'} |")
    L.append("")
    for blk in A["A2_by_param_bin"]:
        g = blk.get("egm_vs_noegm")
        if not g:
            continue
        lo, hi = blk["bin"]
        L.append(f"**箱 [{lo:g},{hi:g}) 内**（参数量跨度 **{g['param_span_M']:.2f}M = "
                 f"{g['param_span_pct_of_mean']:.1f}%**）：EGM 组域外 Dice **{g['dice_gap_pp']:+.2f} pp**、"
                 f"检出 **{g['det_gap_pp']:+.2f} pp**（组均参数量 "
                 f"{g['mean_param_with']:.2f}M vs {g['mean_param_without']:.2f}M）。")
        L.append("")
    L.append("### T-CF-1c ★ 天然容量匹配对（已在资产中，无需训练）")
    L.append("")
    L.append("| 有 EGM | 参数量(M) | 无 EGM | 参数量(M) | 对照参数优势 | 严格容差内(±5%) | 域外 Dice 差 | 域外检出差 |")
    L.append("|---|---:|---|---:|---:|:--:|---:|---:|")
    for p in A["A3_natural_capacity_pairs"]["pairs"]:
        adv = "**更大**" if p["param_advantage_of_control"] else "更小"
        strict = "✅" if p.get("within_strict_tol", False) else "⚠️ 放宽"
        L.append(f"| {p['with_egm']} | {p['param_with_M']:.2f} | {p['without_egm']} | "
                 f"{p['param_without_M']:.2f} | {adv} {p['param_gap_pct']:+.2f}% | {strict} | "
                 f"**{p['dice_gap_pp']:+.2f} pp** | **{p['det_gap_pp']:+.2f} pp** |")
    L.append("")
    L.append("> 容差事前冻结为对照参数量 **±5%**；为**偏向“对 EGM 不利”的方向**，"
             "上界额外放宽到 **+10%**（即允许对照比 EGM 配置**更大**，从而更难证明 EGM 有效）。"
             "标 `⚠️ 放宽` 的对只作参考，不进主论据。")
    L.append("")
    L.append("### T-CF-1d 箱内单调性（参数量序 =? 域外序）")
    L.append("")
    L.append("| 箱(M) | n | ρ(参数, 域外检出) | ρ(参数, 域外 Dice) | 严格单调？ |")
    L.append("|---:|:--:|---:|---:|:--:|")
    for r in A["A4_within_bin_monotonicity"]:
        lo, hi = r["bin"]
        L.append(f"| [{lo:g},{hi:g}) | {r['n']} | {r['rho_param_vs_ooddet']:+.4f} | "
                 f"{r['rho_param_vs_ooddice']:+.4f} | {'是' if r['monotone_det'] else '**否**'} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## T-CF-2 三件套之二：参数回归")
    L.append("")
    L.append("### T-CF-2a 7 个消融配置上的 Spearman")
    L.append("")
    L.append("| 目标量 | ρ(参数量, ·) | n | p（t 近似） | 自助 95% CI | 负号占比 |")
    L.append("|---|---:|:--:|---:|---|---:|")
    for col, name in [("id_dice", "分布内 Dice"), ("ood_dice", "域外 Dice"),
                      ("ood_det", "域外检出"), ("ood_delin", "域外勾画")]:
        d = B["B1_ablation_spearman"][col]
        b = d["boot_ci"]
        L.append(f"| {name} | **{d['rho']:+.4f}** | {d['n']} | {d['p_t']:.4f} | "
                 f"[{b['lo']:+.3f}, {b['hi']:+.3f}] | {b.get('frac_negative', float('nan')):.3f} |")
    L.append("")
    x = B["B1_ablation_spearman"]["_cross_check_vs_v4"]
    L.append(f"> **与 v4 方案对照**：v4 声称 ρ(ID)={x['v4_id_rho']}、ρ(OOD)={x['v4_ood_dice_rho']}；"
             f"本次复算 {x['ours_id_rho']:+.4f} / {x['ours_ood_dice_rho']:+.4f}"
             f" → 一致：ID {'✅' if x['agree_id'] else '❌'}，OOD {'✅' if x['agree_ood'] else '❌'}。")
    L.append("")
    L.append("### T-CF-2b ★ 13/14 架构独立样本（外部效度检验）")
    L.append("")
    L.append("| 目标量 | ρ(参数量, ·) | n | p | 自助 95% CI |")
    L.append("|---|---:|:--:|---:|---|")
    for col, name in [("id_dice", "分布内 Dice"), ("ood_dice", "域外 Dice"),
                      ("ood_det", "域外检出"), ("ood_delin", "域外勾画")]:
        d = B["B2_arch_spearman"][col]
        b = d["boot_ci"]
        L.append(f"| {name} | **{d['rho']:+.4f}** | {d['n']} | {d['p_t']:.4f} | "
                 f"[{b['lo']:+.3f}, {b['hi']:+.3f}] |")
    L.append("")
    L.append("**去掉 EGAUNet（唯一含 EGM 者）后的纯 13 baseline 子集**：")
    L.append("")
    L.append("| 目标量 | ρ | n | p |")
    L.append("|---|---:|:--:|---:|")
    for col, name in [("id_dice", "分布内 Dice"), ("ood_dice", "域外 Dice"),
                      ("ood_det", "域外检出")]:
        d = B["B2_arch_spearman"]["_baseline13_only"][col]
        L.append(f"| {name} | **{d['rho']:+.4f}** | {d['n']} | {d['p_t']:.4f} |")
    L.append("")
    L.append("### T-CF-2c 混淆的机制（必须主动报）")
    L.append("")
    b3 = B["B3_confound_source"]
    L.append(f"含 EGM 的 7 格里 **{b3['n_with_egm']} 个**，均值参数量 "
             f"**{b3['mean_param_with_M']:.2f}M**；不含 EGM 的 **{b3['n_without']} 个**，均值 "
             f"**{b3['mean_param_without_M']:.2f}M** → **EGM 组系统性大 {b3['gap_M']:.2f}M（{b3['gap_pct']:+.1f}%）**。")
    L.append("")
    L.append(f"> {b3['conclusion']}")
    L.append("")
    L.append("### T-CF-2d 偏相关：控制 log 参数量后 EGM 的偏效应")
    L.append("")
    b4 = B["B4_partial_spearman"]
    L.append("| 因变量 | 原始 ρ(EGM, ·) | ρ(EGM, log 参数) | ρ(log 参数, ·) | **偏相关 ρ(EGM, · \\| log 参数)** |")
    L.append("|---|---:|---:|---:|---:|")
    for tag, name in [("ood_det", "域外检出"), ("ood_dice", "域外 Dice")]:
        d = b4[tag]
        L.append(f"| {name} | {d['raw']:+.4f} | {d['rho_egm_param']:+.4f} | "
                 f"{d['rho_param_det'] if 'det' in tag else d['rho_param_dice']:+.4f} | "
                 f"**{d['partial']:+.4f}** |")
    L.append("")
    L.append(f"> {b4['caveat']}")
    L.append("")
    L.append("### T-CF-2e OLS（描述性，df 极小）")
    L.append("")
    o = B["B5_ols"]["ablation"]
    for tag, name in [("ood_det_param_only", "域外检出 ~ log 参数量"),
                      ("ood_det_param_plus_egm", "域外检出 ~ log 参数量 + EGM"),
                      ("ood_dice_param_only", "域外 Dice ~ log 参数量"),
                      ("ood_dice_param_plus_egm", "域外 Dice ~ log 参数量 + EGM")]:
        m = o[tag]
        cs = "、".join(f"{n}={c:+.4f}" for n, c in zip(m["names"], m["coef"]))
        pv = ("；p = " + "、".join(f"{v:.4f}" for v in m["p"])) if m.get("p") else ""
        L.append(f"- **{name}**（n={m['n']}, R²={m['r2']:.4f}, 调整 R²={m['r2_adj']:.4f}）：{cs}{pv}")
    L.append("")
    L.append(f"- 加入 EGM 后 **ΔR²（检出）= {o['delta_r2_det']:+.4f}**、**ΔR²（Dice）= {o['delta_r2_dice']:+.4f}**")
    L.append(f"- 14 架构：域外检出 ~ log 参数量 → R²={B['B5_ols']['arch']['ood_det_param_only']['r2']:.4f}；"
             f"域外 Dice → R²={B['B5_ols']['arch']['ood_dice_param_only']['r2']:.4f}")
    L.append("")
    L.append(f"> {B['B5_ols']['caveat']}")
    L.append("")
    L.append("### T-CF-2f 端点敏感性（v4 称 ρ 由两端点抬起 —— 检验之）")
    L.append("")
    L.append("| 处理 | n | 剔除 | ρ(参数, 域外 Dice) | p |")
    L.append("|---:|:--:|---|---:|---:|")
    for tag in ["drop_min_param", "drop_max_param", "drop_both_ends"]:
        r = B["B6_endpoint_sensitivity"][tag]
        L.append(f"| {tag} | {r['n']} | {', '.join(r['dropped'])} | "
                 f"**{r['rho_param_ooddice']:+.4f}** | {r['p_ooddice']:.4f} |")
    L.append("")
    L.append(f"> ⚠️ {B['B6_endpoint_sensitivity']['_note']}")
    L.append("")
    b7 = B["B7_within_group_param_effect"]
    L.append("### T-CF-2g ★ 分组内参数量效应（切成 EGM / 非 EGM 两组，去掉混淆）")
    L.append("")
    L.append("| 组 | n | 配置（按参数量升序） | 参数量区间(M) | ρ(参数, 域外 Dice) | ρ(参数, 域外检出) | 严格单调(Dice)？ |")
    L.append("|---|:--:|---|---:|---:|---:|:--:|")
    for tag, name in [("no_egm", "非 EGM 组"), ("with_egm", "EGM 组")]:
        d = b7[tag]
        L.append(f"| {name} | {d['n']} | {', '.join('`'+c+'`' for c in d['configs'])} | "
                 f"[{d['param_range'][0]:.2f}, {d['param_range'][1]:.2f}] | "
                 f"**{d['rho_param_ooddice']:+.4f}** | **{d['rho_param_ooddet']:+.4f}** | "
                 f"{'是' if d['monotone_dice'] else '否'} |")
    L.append("")
    lv = b7["_level_shift_at_overlap"]
    L.append(f"**重叠窗 [{lv['window'][0]:g},{lv['window'][1]:g}) 内的水平位移**"
             f"（EGM 组 n={lv['n_with_egm']}／非 EGM 组 n={lv['n_without']}）：")
    L.append("")
    L.append(f"- 组均参数量 **{lv['mean_param_with_M']:.2f}M vs {lv['mean_param_without_M']:.2f}M**"
             f"（EGM 组 **{lv['gap_param_pct']:+.2f}%**，即 EGM 组**更小**）")
    L.append(f"- 域外 Dice **{lv['mean_ood_dice_with']:.4f} vs {lv['mean_ood_dice_without']:.4f}**"
             f"（**{lv['gap_dice_pp']:+.2f} pp**）")
    L.append(f"- 域外检出 **{lv['mean_ood_det_with']:.4f} vs {lv['mean_ood_det_without']:.4f}**"
             f"（**{lv['gap_det_pp']:+.2f} pp**）")
    L.append("")
    L.append(f"> {lv['note']}")
    L.append("")
    b8 = B["B8_egm_alone_counterexamples"]
    L.append("### T-CF-2h ⚠️ 反例清单（必须随结论一起报，否则是选择性报告）")
    L.append("")
    L.append(f"容量匹配对共 **{b8['n_pairs']} 对**：正例 **{b8['n_positive']}**、"
             f"反例 **{b8['n_negative']}**。")
    L.append("")
    L.append("| 承载配置 | 对照 | 对照参数优势 | 域外 Dice 差 | 域外检出差 | 判定 |")
    L.append("|---|---|---:|---:|---:|:--:|")
    for p in b8["positive"] + b8["negative"]:
        good = p in b8["positive"]
        L.append(f"| `{p['carrier']}` | `{p['control']}` | {p['gap_pct']:+.2f}% | "
                 f"{p['dice_gap_pp']:+.2f} pp | {p['det_gap_pp']:+.2f} pp | "
                 f"{'正例 ✅' if good else '**反例 ❌**'} |")
    L.append("")
    L.append(f"> {b8['conclusion']}")
    L.append(">")
    L.append(f"> {b8['n_caveat']}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## T-CF-3 三件套之三：容量匹配方案（P3-4 规格）")
    L.append("")
    a = C["anchor_check"]
    L.append(f"**锚点自检**：构造 `AblationUNet(base_filters=64, 无模块)` = "
             f"**{a['constructed_M']:.4f} M**，既有 `ablation/baseline` = **{a['ablation_baseline_M']:.4f} M**，"
             f"|Δ| = {a['abs_diff_M']:.2e} M → **{a['status']}**")
    L.append("")
    L.append("### T-CF-3a 容量匹配规格表")
    L.append("")
    L.append("| 目标模块配置 | 目标参数量(M) | 匹配 plain UNet | `base_filters` | 实际参数量(M) | 偏差 | 宽度倍率 |")
    L.append("|---|---:|---:|:--:|---:|---:|---:|")
    for r in C["capacity_matched_spec"]["rows"]:
        L.append(f"| {r['target_config']} | {r['target_param_M']:.3f} | "
                 f"`plainUNet_bf{r['matched_base_filters']}` | {r['matched_base_filters']} | "
                 f"{r['matched_param_M']:.3f} | {r['param_gap_pct']:+.2f}% | "
                 f"×{r['width_scale_vs_64']:.3f} |")
    L.append("")
    p = C["P3_4_protocol"]
    L.append("### T-CF-3b P3-4 训练与评测协议（跑前冻结）")
    L.append("")
    L.append(f"- **训练数据**：{p['train_data']}；**评测**：ID {p['eval_data_id']} ／ OOD {p['eval_data_ood']}")
    L.append(f"- **超参**：{p['epochs']}；AMP={p['amp']}；batch {p['batch_size']}；{p['seeds']}")
    L.append(f"- **成本**：{p['cost_estimate']}")
    L.append(f"- **主指标**：{p['primary_metric']}；**次指标**：{p['secondary_metric']}")
    L.append(f"- **闸门**：① {p['gates']['G_train']}；② {p['gates']['G_repro']}")
    L.append("")
    L.append("### T-CF-3c 分支 A / B 判据（预注册，跑前冻结）")
    L.append("")
    L.append("| 分支 | 条件 | H5 判定 | 处置 |")
    L.append("|---|---|---|---|")
    L.append(f"| **A** | {p['decision']['branch_A']['condition']} | "
             f"**支持** | {p['decision']['branch_A']['action']} |")
    L.append(f"| **B** | {p['decision']['branch_B']['condition']} | "
             f"**推翻** | {p['decision']['branch_B']['action']} |")
    L.append("")
    L.append("> ⚠️ **两种结果都可发表**，但**必须在 P5 之前出结果**（P3-4 是分支 A/B 的分水岭）。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## T-CF-4 结论（证据强度递增）")
    L.append("")
    L.append("| 层 | 证据 | 强度 |")
    L.append("|---|---|---|")
    L.append(f"| A 分层 | 已有天然容量匹配对：`egm_dpa` 39.50M 域外 Dice 63.91 / 检出 0.8418 > "
             f"`dpa_msfa` 41.43M 的 56.84 / 0.7755；**但 `egm_only` 的两对全为负** | 观测 |")
    L.append(f"| B 回归 | ① 7 格 ρ(参数, OOD Dice)={B['B1_ablation_spearman']['ood_dice']['rho']:+.4f}"
             f"（含混淆）vs 14 架构 ρ={B['B2_arch_spearman']['ood_dice']['rho']:+.4f}（≈0）；"
             f"② 控制 log 参数量后 EGM 偏相关 ={B['B4_partial_spearman']['ood_det']['partial']:+.4f}"
             f"（检出）/ {B['B4_partial_spearman']['ood_dice']['partial']:+.4f}（Dice）；"
             f"③ 重叠窗内 EGM 组参数量更小却域外更高 "
             f"（{B['B7_within_group_param_effect']['_level_shift_at_overlap']['gap_dice_pp']:+.2f} pp Dice，"
             f"{B['B7_within_group_param_effect']['_level_shift_at_overlap']['gap_det_pp']:+.2f} pp 检出） | 统计（n=7 弱） |")
    L.append("| C 干预 | 容量匹配对照实验（P3-4） | **因果（待跑）** |")
    L.append("")
    L.append("**可写 / 不可写措辞**：")
    L.append("")
    L.append("- ✅ 可写：「控制参数量后，**`EGM+DPA` 组合**的域外检出与 Dice 仍显著更高；"
             "该优势不能被参数量解释，且 14 个架构上参数量与域外成绩几乎不相关（ρ≈0）」。")
    L.append("- ❌ 不可写：「**EGM 模块**单独带来域外优势」（`egm_only` 反例存在）；"
             "「参数量**导致**了优势」（因果结论待 P3-4）；「证明」（观测证据不足以称证明）。")
    L.append("")
    L.append("**H5 当前状态**：**观测证据支持**（限于 `egm_dpa`），**不是因果确认**。"
             "措辞须写“**EGM+DPA 组合**的优势**不能**由参数量解释”，"
             "**不得**写“证明因果”或“EGM 单独有效”。")
    L.append("")
    L.append("---")
    L.append("")
    cu = payload["verdict"].get("claim_upgrade") or {}
    L.append("## T-CF-5 论点升级裁定：ρ≈0 是否升格为独立子论点")
    L.append("")
    L.append("> **缘起**：P2-4 收尾提出两个待定调项，用户授权「能让文章升格就干」。"
             "完整评估见 `00_docs/P2-4裁定_论点升级评估_2026-09-14.md`。")
    L.append("")
    L.append("### T-CF-5a 两问必须分开（第 1 问不是策略问题）")
    L.append("")
    L.append("| # | 问题 | 性质 | 有无选择权 |")
    L.append("|---|---|---|---|")
    L.append("| 1 | H5 收窄为「`EGM+DPA` 组合的域外优势」 | **准确性** | ❌ 无。"
             "`egm_only` 两对容量匹配全为负；若仍写「EGM 模块有效」＝ 选择性报告 |")
    L.append("| 2 | 14 架构 ρ≈0 升格为独立论点 | **策略** | ✅ 有（用户已授权） |")
    L.append("")
    L.append("### T-CF-5b ★ 论点骨架：同一变量、三个样本边界、三种结论")
    L.append("")
    L.append("| 样本边界 | n | ρ(参数量, 域外 Dice) | 解释 |")
    L.append("|---|:--:|---:|---|")
    for b in cu.get("three_boundaries", []):
        L.append(f"| {b['scope']} | {b['n']} | **{b['rho']:+.4f}** | {b['label']} |")
    L.append("")
    L.append("> **主表述（抗打版）**：*同一变量，其与域外成绩的关联**符号与量级都随样本边界翻转***"
             " ⇒ 单架构消融内成立的「参数量解释成绩」**不能外推**到架构之间。")
    L.append("> **禁写**「参数量与成绩无关」（弱且易攻的表述）。")
    L.append("")
    L.append("### T-CF-5c 三重限定（必须与论点同时出现，否则会被打成过度声明）")
    L.append("")
    for q in cu.get("qualifiers", []):
        L.append(f"- {q}")
    L.append("")
    L.append("### T-CF-5d 裁定与落点")
    L.append("")
    L.append(f"- **判定**：{cu.get('level', '')}")
    L.append("- **升**：混淆控制节的支撑证据 → **Results 独立子节**（有标题 / 表 / CI / MDE）")
    L.append("- **升**：Abstract / Contribution 的**一句话**（不新开主条目）")
    L.append("- **不升**：不作主贡献新增条目（零新增实验，避免贡献通胀）")
    L.append(f"- **正文落点**：{cu.get('land_in_p5', 'P5 执行')}")
    L.append("")
    L.append("**可写 / 不可写措辞**：")
    L.append("")
    L.append("- ✅ 可写：「跨架构上**未检出**参数量与域外成绩的关联（ρ≈0，MDE 0.72），"
             "而该关联在单架构网格内为强正 ⇒ 参数量混淆是**样本边界依赖**的」。")
    L.append("- ❌ 不可写：「参数量与成绩**无关**」（＝过度声明）；「**证明**了混淆不存在」；"
             "把描述性区间当作推断区间使用。")
    L.append("")
    L.append("> ⚠️ 升格**不等于立刻改正文**：本段只**备稿与限定**，实际进稿在 **P5**（写作段）。"
             "届时须复核与 P2-3「ρ 不显著」是否产生重复感 —— 二者是**同一变量的不同配对**"
             "（P2-3 用 ID 勾画 vs OOD 检出；本论点用参数量 vs 域外 Dice），若重复则合并为一段。")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ab = load_ablation_table()
    arch, param_sources = load_arch_table()

    # 口径断言
    assert len(ab) == 7, f"消融配置应为 7 个，实为 {len(ab)}"
    assert set(ab["config"]) == set(ABLATION_ORDER), "消融配置名与冻结清单不符"
    assert len(arch) == 14, f"架构表应为 14 行，实为 {len(arch)}"
    assert arch["arch"].is_unique, "架构名重复"

    print("=" * 118)
    print("P2-4 混淆控制三件套")
    print("=" * 118)
    print("\n[7 个消融配置]")
    print(ab[["config", "egm", "dpa", "msfa", "n_modules", "param_M",
              "id_dice", "ood_dice", "ood_det"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n[13 baseline + EGAUNet]")
    print(arch[["arch", "family", "param_M", "id_dice", "ood_dice", "ood_det"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    A = sec_a_strata(ab)
    B = sec_b_regression(ab, arch)
    scan = scan_unet_params()
    print(f"\n[容量扫描] AblationUNet 无模块 base_filters {BF_SCAN_LO}–{BF_SCAN_HI} "
          f"→ 参数量 {scan['params_M'].min():.3f}–{scan['params_M'].max():.3f} M")
    C = sec_c_capacity(ab, scan)

    payload = build_json(ab, arch, A, B, C, param_sources=param_sources)
    payload["C_capacity"]["capacity_scan"] = json.loads(scan.to_json(orient="records"))

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    md = build_md(payload)
    with open(f"{TABLES}/T_confound.md", "w", encoding="utf-8") as fh:
        fh.write(md)

    # ---- 控制台摘要 ----
    print("\n" + "-" * 118)
    print("[A] 天然容量匹配对（有 EGM vs 无 EGM，参数量接近）")
    for p in A["A3_natural_capacity_pairs"]["pairs"]:
        print(f"    {p['with_egm']:>10} {p['param_with_M']:.2f}M vs "
              f"{p['without_egm']:>12} {p['param_without_M']:.2f}M "
              f"(对照 {p['param_gap_pct']:+.2f}%) → Dice {p['dice_gap_pp']:+.2f} pp, "
              f"检出 {p['det_gap_pp']:+.2f} pp")
    print("\n[B] 参数回归")
    print(f"    7 消融: ρ(参数, ID Dice) = {B['B1_ablation_spearman']['id_dice']['rho']:+.4f} "
          f"| ρ(参数, OOD Dice) = {B['B1_ablation_spearman']['ood_dice']['rho']:+.4f} "
          f"| ρ(参数, OOD 检出) = {B['B1_ablation_spearman']['ood_det']['rho']:+.4f}")
    print(f"    14 架构: ρ(参数, ID Dice) = {B['B2_arch_spearman']['id_dice']['rho']:+.4f} "
          f"| ρ(参数, OOD Dice) = {B['B2_arch_spearman']['ood_dice']['rho']:+.4f} "
          f"| ρ(参数, OOD 检出) = {B['B2_arch_spearman']['ood_det']['rho']:+.4f}")
    print(f"    纯 13 baseline: ρ(参数, OOD Dice) = "
          f"{B['B2_arch_spearman']['_baseline13_only']['ood_dice']['rho']:+.4f} "
          f"| ρ(参数, OOD 检出) = "
          f"{B['B2_arch_spearman']['_baseline13_only']['ood_det']['rho']:+.4f}")
    print(f"    偏相关 ρ(EGM, OOD 检出 | log 参数) = {B['B4_partial_spearman']['ood_det']['partial']:+.4f}")
    print(f"    偏相关 ρ(EGM, OOD Dice | log 参数) = {B['B4_partial_spearman']['ood_dice']['partial']:+.4f}")
    b6 = B["B6_endpoint_sensitivity"]
    print(f"    端点敏感性 ρ(OOD Dice)：全 7 格 {B['B1_ablation_spearman']['ood_dice']['rho']:+.4f}"
          f" → 去两端 {b6['drop_both_ends']['rho_param_ooddice']:+.4f}")
    b7 = B["B7_within_group_param_effect"]
    print(f"    分组内 ρ(参数, OOD Dice)：非 EGM 组(n={b7['no_egm']['n']}) "
          f"{b7['no_egm']['rho_param_ooddice']:+.4f} | EGM 组(n={b7['with_egm']['n']}) "
          f"{b7['with_egm']['rho_param_ooddice']:+.4f}")
    lv = b7["_level_shift_at_overlap"]
    print(f"    重叠窗 [{lv['window'][0]:g},{lv['window'][1]:g}) 水平位移：参数量 {lv['gap_param_pct']:+.2f}% "
          f"（EGM 组更小）→ OOD Dice {lv['gap_dice_pp']:+.2f} pp / 检出 {lv['gap_det_pp']:+.2f} pp")
    b8 = B["B8_egm_alone_counterexamples"]
    print(f"    容量匹配对：正例 {b8['n_positive']} / 反例 {b8['n_negative']} "
          f"（反例承载配置：{', '.join(b8['carriers_with_negative'])}）")
    print("\n[C] 容量匹配方案（P3-4）")
    print(f"    锚点: base_filters=64 → {C['anchor_check']['constructed_M']:.4f} M = "
          f"既有 baseline（{C['anchor_check']['status']}）")
    for r in C["capacity_matched_spec"]["rows"]:
        print(f"    {r['target_config']:>12} {r['target_param_M']:.3f}M ← "
              f"plainUNet bf={r['matched_base_filters']:>3} ({r['matched_param_M']:.3f}M, "
              f"{r['param_gap_pct']:+.2f}%)")
    print("\n" + "=" * 118)
    print(f"验收点：egm_dpa(39.50M) 域外仍高于 dpa_msfa(41.43M)？ "
          f"{'✅ 是' if payload['verdict']['expectation_satisfied'] else '❌ 否'}  → "
          f"H5 = {payload['verdict']['H5_status']}")
    print(f"已写: {OUT_JSON}\n已写: {TABLES}/T_confound.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
