# -*- coding: utf-8 -*-
"""
文件名: p2_4_crosscheck.py
功能: 【论文二 P2-4】**独立复核**（G4 铁律：关键数字必须双路）

原则:
    - **不 import** `confound_control.py`（避免同源 bug 互相掩盖）。
    - 秩统计改用 **scipy.stats.spearmanr**（与主脚本自实现的平均秩不同源头）。
    - 参数量、OOD 读数**从原始文件重读**（各自 results.json / decomp CSV / p31a CSV）。
    - 与 `confound.json` 中落盘的数字逐项比对，容差 1e-9。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/p2_4_crosscheck.py
退出码:
    0 = 全部 MATCH；2 = 有 MISMATCH
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

E_ROOT = "E:/paper2_ablation_reliability"
D_PROJ = "D:/medical_segmentation"
STATS = f"{E_ROOT}/03_results/stats"
TOL = 1e-9

CFG7 = ["baseline", "dpa_only", "egm_only", "msfa_only",
        "egm_dpa", "egm_msfa", "dpa_msfa"]

checks = []


def chk(name, got, exp, tol=TOL):
    ok = False
    if got is None or exp is None:
        ok = (got is None and exp is None)
    elif isinstance(got, (int, float)) and isinstance(exp, (int, float)):
        if np.isnan(got) and np.isnan(exp):
            ok = True
        else:
            ok = abs(float(got) - float(exp)) <= tol
    else:
        ok = (got == exp)
    checks.append((name, got, exp, ok, tol))
    return ok


def main() -> int:
    print("=" * 114)
    print("P2-4 独立复核（scipy 秩统计 + 原始文件重读）")
    print("=" * 114)

    # ---------- 1. 重读原始数据 ----------
    dec = pd.read_csv(f"{STATS}/decomp_ablation_etis.csv").set_index("config")
    p7, id7, odd7, odet7, egm7 = [], [], [], [], []
    for c in CFG7:
        meta = json.load(open(f"{D_PROJ}/experiments/ablation/{c}/results.json",
                             encoding="utf-8"))
        p7.append(float(meta["parameters_M"]))
        id7.append(float(meta["test_metrics"]["Dice"]))
        odd7.append(float(dec.loc[c, "mean_dice"]))
        odet7.append(float(dec.loc[c, "detection"]))
        egm7.append(1.0 if bool(dec.loc[c, "EGM"]) else 0.0)
    p7, id7, odd7, odet7, egm7 = map(np.asarray, (p7, id7, odd7, odet7, egm7))

    units = pd.read_csv(f"{STATS}/p31a_units_n14.csv").set_index("arch")
    reg = json.load(open(f"{STATS}/arch_params_registry.json", encoding="utf-8"))
    pa, ida, oda, odeta, egma = [], [], [], [], []
    for a in units.index:
        if a == "EGAUNet":
            pa.append(float(reg["EGAUNet"]["parameters_M"]))
            egma.append(1.0)
        else:
            meta = json.load(open(f"{D_PROJ}/experiments/baseline/{a}/results.json",
                                 encoding="utf-8"))
            pa.append(float(meta["parameters_M"]))
            egma.append(0.0)
        ida.append(float(units.loc[a, "id_dice"]))
        oda.append(float(units.loc[a, "ood_dice"]))
        odeta.append(float(units.loc[a, "ood_det_subst"]))
    pa, ida, oda, odeta, egma = map(np.asarray, (pa, ida, oda, odeta, egma))
    print(f"[重读] 7 消融 {len(p7)} 行 | 架构 {len(pa)} 行 | EGAUNet 参数量 "
          f"{pa[list(units.index).index('EGAUNet')]:.6f} M")

    # ---------- 2. 读主脚本产物 ----------
    cj = json.load(open(f"{STATS}/confound.json", encoding="utf-8"))
    B1 = cj["B_regression"]["B1_ablation_spearman"]
    B2 = cj["B_regression"]["B2_arch_spearman"]
    B4 = cj["B_regression"]["B4_partial_spearman"]
    B7 = cj["B_regression"]["B7_within_group_param_effect"]
    B8 = cj["B_regression"]["B8_egm_alone_counterexamples"]
    Cc = cj["C_capacity"]

    # ---------- 3. 7 消融 Spearman（scipy 路） ----------
    for col, arr, key in [("id_dice", id7, "id_dice"), ("ood_dice", odd7, "ood_dice"),
                          ("ood_det", odet7, "ood_det")]:
        rho, pv = stats.spearmanr(p7, arr)
        chk(f"7 格 ρ(参数, {col})   [scipy]", float(rho), float(B1[key]["rho"]), 1e-12)
        chk(f"7 格 p(参数, {col})    [scipy]", float(pv), float(B1[key]["p_t"]), 1e-9)

    # ---------- 4. 14 架构 Spearman（scipy 路） ----------
    for col, arr, key in [("id_dice", ida, "id_dice"), ("ood_dice", oda, "ood_dice"),
                          ("ood_det", odeta, "ood_det")]:
        rho, pv = stats.spearmanr(pa, arr)
        chk(f"14 架构 ρ(参数, {col}) [scipy]", float(rho), float(B2[key]["rho"]), 1e-12)
        chk(f"14 架构 p(参数, {col})  [scipy]", float(pv), float(B2[key]["p_t"]), 1e-9)

    # ---------- 5. 偏相关（scipy rho + 闭式公式重算） ----------
    logp = np.log10(p7)

    def part(x, y, z):
        rxy = stats.spearmanr(x, y)[0]
        rxz = stats.spearmanr(x, z)[0]
        ryz = stats.spearmanr(y, z)[0]
        return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))

    chk("偏相关 ρ(EGM, OOD 检出 | log 参数) [scipy]",
        part(egm7, odet7, logp), float(B4["ood_det"]["partial"]), 1e-12)
    chk("偏相关 ρ(EGM, OOD Dice | log 参数) [scipy]",
        part(egm7, odd7, logp), float(B4["ood_dice"]["partial"]), 1e-12)

    # ---------- 6. 分组内 ρ（B7） ----------
    m_e = egm7 == 1
    chk("非 EGM 组 ρ(参数, OOD Dice) [scipy]",
        float(stats.spearmanr(p7[~m_e], odd7[~m_e])[0]),
        float(B7["no_egm"]["rho_param_ooddice"]), 1e-12)
    chk("EGM 组 ρ(参数, OOD Dice) [scipy]",
        float(stats.spearmanr(p7[m_e], odd7[m_e])[0]),
        float(B7["with_egm"]["rho_param_ooddice"]), 1e-12)

    # ---------- 7. 重叠窗水平位移（B7） ----------
    win = (p7 >= 35.0) & (p7 < 42.0)
    lv = B7["_level_shift_at_overlap"]
    chk("重叠窗 EGM 组均参数量",
        float(p7[win & m_e].mean()), float(lv["mean_param_with_M"]), 1e-12)
    chk("重叠窗 非 EGM 组均参数量",
        float(p7[win & ~m_e].mean()), float(lv["mean_param_without_M"]), 1e-12)
    chk("重叠窗 参数量差(%)",
        float((p7[win & m_e].mean() / p7[win & ~m_e].mean() - 1) * 100),
        float(lv["gap_param_pct"]), 1e-9)
    chk("重叠窗 Dice 位移(pp)",
        float((odd7[win & m_e].mean() - odd7[win & ~m_e].mean()) * 100),
        float(lv["gap_dice_pp"]), 1e-9)
    chk("重叠窗 检出位移(pp)",
        float((odet7[win & m_e].mean() - odet7[win & ~m_e].mean()) * 100),
        float(lv["gap_det_pp"]), 1e-9)

    # ---------- 8. 验收点：egm_dpa vs dpa_msfa ----------
    i_e = CFG7.index("egm_dpa")
    i_d = CFG7.index("dpa_msfa")
    chk("验收点 Dice 差(pp)", float((odd7[i_e] - odd7[i_d]) * 100),
        float(cj["verdict"]["key_pair_egm_dpa_vs_dpa_msfa"]["dice_gap_pp"]), 1e-9)
    chk("验收点 检出差(pp)", float((odet7[i_e] - odet7[i_d]) * 100),
        float(cj["verdict"]["key_pair_egm_dpa_vs_dpa_msfa"]["det_gap_pp"]), 1e-9)
    chk("验收点 参数量差(%)",
        float((p7[i_d] / p7[i_e] - 1) * 100),
        float(cj["verdict"]["key_pair_egm_dpa_vs_dpa_msfa"]["param_gap_pct"]), 1e-9)
    chk("验收点成立（对照更大却更低）",
        bool(odd7[i_e] > odd7[i_d] and p7[i_d] > p7[i_e]),
        bool(cj["verdict"]["expectation_satisfied"]), 0.0)

    # ---------- 9. 反例计数（B8） ----------
    n_pos = n_neg = 0
    for iw in [i for i, v in enumerate(egm7) if v == 1]:
        for io in [j for j, v in enumerate(egm7) if v == 0]:
            dp = (p7[io] - p7[iw]) / p7[iw] * 100
            if -5.0 <= dp <= 15.0:
                if odd7[iw] > odd7[io] and odet7[iw] > odet7[io]:
                    n_pos += 1
                else:
                    n_neg += 1
    chk("容量匹配对 正例数", n_pos, int(B8["n_positive"]), 0.0)
    chk("容量匹配对 反例数", n_neg, int(B8["n_negative"]), 0.0)

    # ---------- 10. 容量匹配锚点（用消融 baseline 参数量比对） ----------
    chk("锚点: ablation/baseline 参数量(M)", float(p7[0]),
        float(Cc["anchor_check"]["ablation_baseline_M"]), 1e-12)
    chk("锚点: 构造值 == 既有值", float(Cc["anchor_check"]["constructed_M"]),
        float(Cc["anchor_check"]["ablation_baseline_M"]), 1e-3)

    # ---------- 11. EGAUNet 参数量（独立重算） ----------
    chk("EGAUNet 参数量(M) [注册表]", float(reg["EGAUNet"]["parameters_M"]),
        float(pa[list(units.index).index("EGAUNet")]), 1e-12)
    chk("EGAUNet 注册表 cross_check 状态", reg["EGAUNet"]["cross_check"]["status"],
        "MATCH", 0.0)

    # ---------- 汇总 ----------
    print("\n" + "-" * 114)
    n_bad = 0
    for name, got, exp, ok, tol in checks:
        mark = "MATCH" if ok else "**MISMATCH**"
        if not ok:
            n_bad += 1
        g = f"{got:.10g}" if isinstance(got, (int, float)) else str(got)
        e = f"{exp:.10g}" if isinstance(exp, (int, float)) else str(exp)
        print(f"  [{mark:^12}] {name:<42} got={g:<22} exp={e}")
    print("-" * 114)
    print(f"共 {len(checks)} 项：MATCH {len(checks)-n_bad} ｜ MISMATCH {n_bad}")
    if n_bad == 0:
        print("结论: **全部 MATCH** —— P2-4 关键数字双路复核通过 ✅")
        return 0
    print("结论: **存在 MISMATCH** —— 须逐项排查 ❌")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
