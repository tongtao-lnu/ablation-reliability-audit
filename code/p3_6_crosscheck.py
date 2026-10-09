#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P3-6 独立复核（G4：关键数字双路复核）

本脚本**不 import** 主模块 `p3_6_fragmentation_caliber.py`，而是：
  * 从**书面规格**（主模块 docstring 的 A1–A5 / B1–B8）重新实现口径；
  * 连通域改用 **skimage.measure.label(connectivity=2)**，而主路用
    `scipy.ndimage.label(structure=np.ones((3,3)))` —— 两套独立实现；
  * 口径 V 用**移除量恒等式**重算：
        dice_V = (2*(TP - t) + eps) / ((nP - t - f) + (nG - t - u) + eps)
    其中 t/f/u 是被遮挡区内的 TP / FP / FN，而主路直接从布尔数组算 dice_V。
    两条路径在代数上等价但代码路径完全不同 ⇒ 若一致，则既验证了 V，也验证了 Δ 分解。

对照对象：`03_results/stats/fragmentation.json` 与 `caliber_swap.json` 的**逐单元**读数。

跑法
----
D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/p3_6_crosscheck.py
"""

from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from skimage.measure import label as sklabel

E_ROOT = "E:/paper2_ablation_reliability"
D_ROOT = "D:/medical_segmentation"

ID_PRED_ROOT = E_ROOT + "/03_results/raw/indist_pred"
ID_GT_DIR = D_ROOT + "/processed_data/test/masks"
ABL_ETIS_ROOT = D_ROOT + "/results_ablation_etis"

IN_FRAG = E_ROOT + "/03_results/stats/fragmentation.json"
IN_SWAP = E_ROOT + "/03_results/stats/caliber_swap.json"
OUT_JSON = E_ROOT + "/03_results/stats/p3_6_crosscheck.json"

ID_ARCHS = ["AttentionUNet", "CaraNet", "M2SNet", "MultiResUNet", "PSPNet", "PolypPVT",
            "PraNet", "ResUNet", "SANet", "SegNet", "TransUNet", "UACANet", "UNet"]
ABL_ORDER = ["baseline", "egm_only", "dpa_only", "msfa_only", "egm_dpa", "egm_msfa", "dpa_msfa"]

# —— 冻结常量：照抄规格，不引用主模块 ——
EPS = 1e-8
MIN_COMP_PX = 10
RUBRIC_T_NCOMP = 2
RUBRIC_T_LRATIO = 0.90
AXIS_RATIO = 1.6
COND = (0.30, "target")


def dice_from_counts(inter, np_, ng):
    return float((2.0 * inter + EPS) / (np_ + ng + EPS))


def n_comp_ge(mask, min_px=MIN_COMP_PX):
    """skimage 路径的连通域计数（8-邻接 = connectivity=2）。"""
    if not mask.any():
        return 0, float("nan")
    lab = sklabel(mask, connectivity=2, background=0)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= min_px
    n = int(keep.sum())
    if n == 0:
        return 0, float("nan")
    ratio = float(int(np.where(keep, sizes, 0).max()) / int(mask.sum()))
    return n, ratio


def ellipse(shape, center, area_frac, axis_ratio=AXIS_RATIO):
    """照规格重建椭圆遮挡（独立实现）。"""
    H, W = shape
    if area_frac <= 0:
        return np.zeros((H, W), dtype=bool)
    b = float(np.sqrt(area_frac * H * W / (np.pi * axis_ratio)))
    c = float(b * axis_ratio)
    cy, cx = float(center[0]), float(center[1])
    cy = min(max(cy, b), H - 1 - b) if H - 1 - b > b else (H - 1) / 2.0
    cx = min(max(cx, c), W - 1 - c) if W - 1 - c > c else (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy - cy) / b) ** 2 + ((xx - cx) / c) ** 2) <= 1.0


def occ_center(gt, placement, name, area):
    H, W = gt.shape
    if placement == "target":
        idx = np.argwhere(gt)
        return idx.mean(axis=0) if idx.size else np.array([H / 2.0, W / 2.0])
    s = zlib.crc32(("{}|caliberswap|{:.2f}".format(name, area)).encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(s)
    return np.array([rng.uniform(H * 0.15, H * 0.85), rng.uniform(W * 0.15, W * 0.85)])


def load_id_gt():
    out = {}
    for f in sorted(os.listdir(ID_GT_DIR)):
        if f.endswith(".npy"):
            out[os.path.splitext(f)[0]] = (np.load(os.path.join(ID_GT_DIR, f)) > 0.5)
    return out


def iter_id(arch, gt_map):
    d = os.path.join(ID_PRED_ROOT, arch)
    csv = pd.read_csv(os.path.join(d, "sample_metrics.csv"))
    stored = {}
    for _, r in csv.iterrows():
        nm = str(r["name"])
        stored[os.path.splitext(nm)[0] if nm.endswith(".npy") else nm] = float(r["Dice"])
    for f in sorted(os.listdir(os.path.join(d, "masks"))):
        if not f.endswith(".npy"):
            continue
        base = os.path.splitext(f)[0]
        if base in gt_map:
            yield base, np.load(os.path.join(d, "masks", f)).astype(bool), gt_map[base], stored.get(base, float("nan"))


def iter_ood(cfg):
    arr = np.load(os.path.join(ABL_ETIS_ROOT, cfg, "predictions.npy"), allow_pickle=True)
    try:
        for it in arr:
            yield (str(it["name"]), np.asarray(it["pred"]).astype(bool),
                   (np.asarray(it["mask"]) > 0.5), float(it["dice"]))
    finally:
        del arr


def main():
    print("[1/4] load reference outputs ...")
    frag = json.load(open(IN_FRAG, "r", encoding="utf-8"))
    swap = json.load(open(IN_SWAP, "r", encoding="utf-8"))
    ref_frag = frag["by_unit"]                                     # {set: {unit: agg}}
    key = "%d_%s" % (int(COND[0] * 100), COND[1])
    ref_swap = {}
    for r in swap["per_condition"][key]["unit_means"]:             # 逐单元 mean F/V
        ref_swap.setdefault(r["unit_set"], {})[r["unit"]] = r

    gt_map = load_id_gt()
    print("[2/4] recompute (skimage CC + removal-identity V) ...")

    rows = []
    for unit_set, units, it in (("ID", ID_ARCHS, "id"), ("OOD", ABL_ORDER, "ood")):
        for unit in units:
            src = iter_id(unit, gt_map) if it == "id" else iter_ood(unit)
            dices, ncs, usable, diceF, diceV = [], [], [], [], []
            for name, pred, gt, stored in src:
                nc, lr = n_comp_ge(pred)
                ncg, _ = n_comp_ge(gt)
                d = float(stored)
                dices.append(d)
                ncs.append(nc)
                usable.append(bool(nc <= RUBRIC_T_NCOMP and np.isfinite(lr) and lr >= RUBRIC_T_LRATIO))
                # ---- F
                inter = int(np.logical_and(pred, gt).sum())
                nP, nG = int(pred.sum()), int(gt.sum())
                F = dice_from_counts(inter, nP, nG)
                diceF.append(F)
                # ---- V（移除量恒等式路径）
                occ = ellipse(gt.shape, occ_center(gt, COND[1], name, COND[0]), COND[0])
                p_occ = np.logical_and(pred, occ)
                g_occ = np.logical_and(gt, occ)
                t = int(np.logical_and(p_occ, g_occ).sum())
                f = int(p_occ.sum()) - t
                u = int(g_occ.sum()) - t
                V = dice_from_counts(inter - t, nP - t - f, nG - t - u)
                diceV.append(V)
                # 一致性：与布尔路径比
                vis = ~occ
                Vb = dice_from_counts(int(np.logical_and(np.logical_and(pred, vis),
                                                         np.logical_and(gt, vis)).sum()),
                                      int(np.logical_and(pred, vis).sum()),
                                      int(np.logical_and(gt, vis).sum()))
                if abs(V - Vb) > 1e-12:
                    rows.append(dict(_warn="V identity mismatch", unit=unit, name=name,
                                     delta=abs(V - Vb)))
            n = len(dices)
            rec = dict(unit_set=unit_set, unit=unit, n=n,
                       mean_dice=float(np.mean(dices)),
                       mean_ncomp=float(np.mean(ncs)),
                       frac_fragmented=float(np.mean([x >= 2 for x in ncs])),
                       frac_usable=float(np.mean(usable)),
                       mean_F=float(np.mean(diceF)), mean_V=float(np.mean(diceV)))
            rf = ref_frag.get(unit_set, {}).get(unit)
            rs = ref_swap.get(unit_set, {}).get(unit)
            rec["d_dice"] = abs(rec["mean_dice"] - rf["mean_dice"]) if rf else None
            rec["d_ncomp"] = abs(rec["mean_ncomp"] - rf["mean_ncomp_pred"]) if rf else None
            rec["d_frag"] = abs(rec["frac_fragmented"] - rf["frac_fragmented"]) if rf else None
            rec["d_usable"] = abs(rec["frac_usable"] - rf["frac_usable"]) if rf else None
            rec["d_F"] = abs(rec["mean_F"] - rs["dice_F"]) if rs else None
            rec["d_V"] = abs(rec["mean_V"] - rs["V_" + key]) if rs else None
            rows.append(rec)
            print("   %-4s %-14s n=%4d  d_dice=%.2e d_ncomp=%.2e d_frag=%.2e d_usable=%.2e "
                  "d_F=%.2e d_V=%.2e"
                  % (unit_set, unit, n, rec["d_dice"], rec["d_ncomp"], rec["d_frag"],
                     rec["d_usable"], rec["d_F"], rec["d_V"]))

    print("[3/4] verdict ...")
    real = [r for r in rows if "unit" in r and "d_dice" in r]
    worst = {k: max((r[k] for r in real if r[k] is not None), default=0.0)
             for k in ("d_dice", "d_ncomp", "d_frag", "d_usable", "d_F", "d_V")}
    warns = [r for r in rows if "_warn" in r]
    tol = dict(d_dice=1e-9, d_ncomp=1e-9, d_frag=1e-9, d_usable=1e-9, d_F=1e-9, d_V=1e-9)
    ok = all(worst[k] <= tol[k] for k in worst) and not warns
    print("   worst:", ", ".join("%s=%.3e" % (k, v) for k, v in worst.items()))
    print("   V-identity warnings:", len(warns))
    print("   VERDICT:", "MATCH" if ok else "MISMATCH")

    out = dict(script="02_code/analysis/p3_6_crosscheck.py",
               note="G4 dual-path recheck; skimage.measure.label for CC; V via removal-identity",
               independent_of_main_module=True, condition=key,
               n_units=len(real), worst=worst, tol=tol,
               v_identity_warnings=warns, verdict="MATCH" if ok else "MISMATCH",
               per_unit=rows)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("[4/4] wrote %s (%d B)" % (OUT_JSON, os.path.getsize(OUT_JSON)))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
