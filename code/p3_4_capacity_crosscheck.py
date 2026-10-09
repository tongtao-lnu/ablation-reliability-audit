# -*- coding: utf-8 -*-
"""
p3_4_capacity_crosscheck.py -- P3-4 容量匹配对照的**独立复核**（G4 双路复核）

铁律：**不 import** `p3_4_etis_capacity.py`，不读它的中间变量。
本脚本自己实现 dice / 检出 / 配对自助，从**原始产物**重算，再与已发布的
`03_results/stats/capacity_matched.json`（及 `T_p3_4_capacity.md` 同源）逐项对照。

对照项
------
  1. 模块侧 ETIS 指标：从 `results_ablation_etis/<cfg>/predictions.npy` 独立重算
     （并用 npy 内自带的逐样本 `dice` 字段作第三路对照）
  2. 容量侧 ETIS 指标：从 `03_results/raw/capacity_matched/<name>.json` 的逐样本数组重算，
     并校验「汇总值 == 逐样本均值」（产物内部自洽）
  3. Δ检出 点估计 与 自助 95% CI（同 spec：B=2000, seed=42, 配对百分位法）
  4. ID 对齐门禁 Δ(pp)：直接从两侧 `results.json` 的 `test_metrics.Dice` 重算
  5. 分支判定 A / B / ambiguous 重推

产物
----
  03_results/stats/p3_4_capacity_crosscheck.json   机器可读对照结果
  控制台打印 MATCH / MISMATCH 与最大偏差
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

D = "D:/medical_segmentation"
E = "E:/paper2_ablation_reliability"

RAW_DIR = E + "/03_results/raw/capacity_matched"
SUM_JSON = E + "/03_results/stats/capacity_matched.json"
OUT_JSON = E + "/03_results/stats/p3_4_capacity_crosscheck.json"

MOD_PRED_ROOT = D + "/results_ablation_etis"
ABL_ROOT = D + "/experiments/ablation"
CAP_ROOT = D + "/experiments/capacity_matched/ablation"

# 架构恒等别名（与主脚本 CKPT_ALIAS 同义，独立声明）：`plainUNet_bf64` 的权重/ID 记录
# 实际在 `experiments/ablation/baseline/`（两者 architectural 同一模型，锚点 abs_diff_M = 0.0）。
CAP_ALIAS = {"plainUNet_bf64": ABL_ROOT + "/baseline"}

# 独立声明（不 import 主脚本）：模块配置 -> (plain UNet 名, base_filters)
PAIRS = {
    "baseline":  ("plainUNet_bf64", 64),
    "dpa_only":  ("plainUNet_bf66", 66),
    "egm_only":  ("plainUNet_bf71", 71),
    "egm_dpa":   ("plainUNet_bf72", 72),
    "msfa_only": ("plainUNet_bf73", 73),
    "dpa_msfa":  ("plainUNet_bf74", 74),
    "egm_msfa":  ("plainUNet_bf79", 79),
}

EPS = 1e-8
B = 2000
SEED = 42
BRANCH_A_PP = 5.0
ID_GATE_PP = 1.0


# --------------------------------------------------------------------------- #
# 独立实现的算式
# --------------------------------------------------------------------------- #
def my_dice(inter, npred, ngt):
    return (2.0 * inter + EPS) / (npred + ngt + EPS)


def my_metrics(preds, gts):
    """独立重算：preds/gts 为同长可迭代的 bool 数组。返回汇总 + 逐样本。"""
    inter, npred, ngt, dice = [], [], [], []
    for P, G in zip(preds, gts):
        P = np.asarray(P).astype(bool).ravel()
        G = np.asarray(G).astype(bool).ravel()
        i = int(np.logical_and(P, G).sum())
        inter.append(i)
        npred.append(int(P.sum()))
        ngt.append(int(G.sum()))
        dice.append(my_dice(i, npred[-1], ngt[-1]))
    inter = np.asarray(inter, float)
    dice = np.asarray(dice, float)
    det = inter >= 1.0                      # no-eps 整数判据
    return dict(n=len(preds), dice=dice, det=det,
                detection=float(det.mean()), mean_dice=float(dice.mean()),
                n_undetected=int((~det).sum()))


def my_boot(a, b, B=B, seed=SEED):
    """配对自助：Δ = mean(a) − mean(b)；同 spec（B、seed、百分位法）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert len(a) == len(b), "配对长度不一致"
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    stat = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))
    return dict(point=float(d.mean()), lo=lo, hi=hi,
                crosses_zero=bool(lo <= 0 <= hi))


def read_results_dice(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return float(json.load(f)["test_metrics"]["Dice"])


# --------------------------------------------------------------------------- #
def main():
    if not os.path.exists(SUM_JSON):
        print(f"[XC] 缺汇总 {SUM_JSON} —— 先跑主脚本")
        return 3
    with open(SUM_JSON, encoding="utf-8") as f:
        pub = json.load(f)
    pub_pairs = {r["config"]: r for r in pub.get("pairs", [])}
    if not pub_pairs:
        print("[XC] 汇总里没有任何对 —— 无法复核")
        return 3

    checks, rows = [], []
    maxdev = 0.0

    def rec(name, dev, tol, note=""):
        nonlocal maxdev
        maxdev = max(maxdev, dev)
        checks.append(dict(check=name, deviation=dev, tol=tol,
                           ok=bool(dev <= tol), note=note))

    for cfg, (plain, bf) in PAIRS.items():
        if cfg not in pub_pairs:
            continue
        p = pub_pairs[cfg]

        # ---- 模块侧：从 predictions.npy 独立重算 ---- #
        npy_p = os.path.join(MOD_PRED_ROOT, cfg, "predictions.npy")
        if not os.path.exists(npy_p):
            checks.append(dict(check=f"{cfg}:module_npy", deviation=None,
                               tol=None, ok=False, note="predictions.npy 缺失"))
            continue
        arr = np.load(npy_p, allow_pickle=True)
        P = [np.asarray(it["pred"]) for it in arr]
        G = [np.asarray(it["mask"]) for it in arr]
        dice_in_npy = np.asarray([float(it["dice"]) for it in arr], float)
        M = my_metrics(P, G)
        assert M["n"] == 196, f"{cfg} ETIS 样本数 {M['n']} != 196"

        # 3 路：我的 Dice vs npy 自带 dice 字段（**信息性**：该字段为 float32 存储，
        # 实测偏差 ~3e-8，故容差放宽到 1e-6；真正的判据是下面与已发布值的对照）
        rec(f"{cfg}:module_dice_vs_npy_field",
            float(np.max(np.abs(M["dice"] - dice_in_npy))), 1e-6,
            "独立 Dice vs npy 自带逐样本 dice（float32 字段，信息性）")
        # 我的汇总 vs 已发布
        rec(f"{cfg}:module_detection", abs(M["detection"] - p["module_etis_detection"]), 1e-12)
        rec(f"{cfg}:module_dice", abs(M["mean_dice"] - p["module_etis_dice"]), 1e-9)

        # ---- 容量侧：从 raw json 重算 + 内部自洽 ---- #
        cap_p = os.path.join(RAW_DIR, plain + ".json")
        if not os.path.exists(cap_p):
            checks.append(dict(check=f"{cfg}:capacity_raw", deviation=None,
                               tol=None, ok=False, note="raw json 缺失"))
            continue
        with open(cap_p, encoding="utf-8") as f:
            cap = json.load(f)
        cd = np.asarray(cap["per_sample_dice"], float)
        ct = np.asarray(cap["per_sample_det"], float)
        rec(f"{cfg}:capacity_det_internal", abs(ct.mean() - cap["etis_detection"]), 1e-12,
            "raw json: mean(per_sample_det) vs etis_detection")
        rec(f"{cfg}:capacity_dice_internal", abs(cd.mean() - cap["etis_dice"]), 1e-12,
            "raw json: mean(per_sample_dice) vs etis_dice")
        rec(f"{cfg}:capacity_detection", abs(ct.mean() - p["capacity_etis_detection"]), 1e-12)
        rec(f"{cfg}:capacity_dice", abs(cd.mean() - p["capacity_etis_dice"]), 1e-9)

        # ---- Δ检出 + 自助 CI（模块 − 容量）---- #
        boot = my_boot(M["det"], ct)
        gain = (M["detection"] - float(ct.mean())) * 100
        rec(f"{cfg}:gain_pp", abs(gain - p["gain_detection_pp"]), 1e-9)
        rec(f"{cfg}:delta_point", abs(boot["point"] * 100 - p["gain_detection_pp"]), 1e-9)
        rec(f"{cfg}:delta_ci_lo", abs(boot["lo"] * 100 - p["delta_detection"]["lo"] * 100), 1e-9)
        rec(f"{cfg}:delta_ci_hi", abs(boot["hi"] * 100 - p["delta_detection"]["hi"] * 100), 1e-9)
        rec(f"{cfg}:crosses_zero", 0.0 if boot["crosses_zero"] == p["delta_detection"]["crosses_zero"]
            else 1.0, 0.0, "CI 是否跨 0")

        # ---- ID 对齐门禁（T-CF-3b 闸门①）---- #
        # 主脚本口径：|容量侧**重算** ID Dice − 模块侧 results.json Dice|；
        # 此处照同口径复算（容量侧重算值 = raw json 的 id_dice_recomputed）。
        id_mod = read_results_dice(os.path.join(ABL_ROOT, cfg, "results.json"))
        id_cap_rec = cap.get("id_dice_recomputed")
        id_cap_json = read_results_dice(os.path.join(CAP_ALIAS.get(plain, os.path.join(CAP_ROOT, plain)), "results.json"))
        if id_mod is not None and id_cap_rec is not None:
            rec(f"{cfg}:id_gate_gap_pp",
                abs(abs(id_cap_rec - id_mod) * 100 - p["id_gate_gap_pp"]), 1e-9,
                "与主脚本同口径（容量侧重算 vs 模块侧记录）")
        if id_mod is not None and id_cap_json is not None:
            # 替代口径：两侧都取「记录值」。它与主口径之差应恰等于容量侧自洽 Δ×100
            # （主口径用重算值、本口径用记录值）⇒ 两者一致到 1e-3 pp 即证明
            # 「用重算还是用记录值」对门禁结论无实质影响。
            alt = abs(id_cap_json - id_mod) * 100
            lag = abs(id_cap_json - id_cap_rec) * 100 if id_cap_rec is not None else 0.0
            rec(f"{cfg}:id_gate_gap_pp_alt",
                abs(alt - p["id_gate_gap_pp"]), 1e-3,
                f"替代口径（两侧记录值）与主口径差 {lag:.3e} pp（= 容量侧自洽 Δ×100）")

        # ---- 分支重推 ---- #
        expect = ("A" if gain >= BRANCH_A_PP
                  else ("B" if (gain < BRANCH_A_PP and boot["crosses_zero"]) else "ambiguous"))
        rec(f"{cfg}:branch", 0.0 if expect == p["branch"] else 1.0, 0.0,
            f"期望 {expect} / 已发布 {p['branch']}")

        rows.append(dict(config=cfg, plain=plain, base_filters=bf,
                         module_detection=float(M["detection"]),
                         capacity_detection=float(ct.mean()),
                         gain_pp=float(gain),
                         boot_point_pp=float(boot["point"] * 100),
                         boot_ci=[float(boot["lo"] * 100), float(boot["hi"] * 100)],
                         branch=expect))

    ok = all(c["ok"] for c in checks)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(n_checks=len(checks), n_fail=sum(1 for c in checks if not c["ok"]),
                       max_deviation=maxdev, verdict="MATCH" if ok else "MISMATCH",
                       pairs=rows, checks=checks), f, indent=2, ensure_ascii=False)

    print("=" * 84)
    print(f"[XC] P3-4 独立复核：{len(checks)} 项检查，失败 {sum(1 for c in checks if not c['ok'])} 项")
    for c in checks:
        if not c["ok"]:
            print(f"  ✗ {c['check']}: dev={c['deviation']} tol={c['tol']} {c['note']}")
    print(f"[XC] 最大偏差 {maxdev:.3e}")
    print(f"[XC] 判定 -> {'MATCH' if ok else 'MISMATCH'}")
    print(f"[XC] 产物 -> {OUT_JSON}")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
