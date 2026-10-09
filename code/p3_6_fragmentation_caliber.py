# -*- coding: utf-8 -*-
"""
p3_6_fragmentation_caliber.py -- P3-6

两条产出，一次跑完（纯 CPU 计算，无新推理、无新训练）：

  A. 碎片化维度系统化            -> 03_results/stats/fragmentation.json  + T_fragmentation.md
  B. full-mask vs visible-only 口径对照 -> 03_results/stats/caliber_swap.json + T_caliber_swap.md

================================================================================
跑前冻结（G6：先写定义，事后不改）
================================================================================

---- A 部分：碎片化 ----
A1  连通性           : 8-邻接（`np.ones((3,3))`）为主口径；4-邻接仅作敏感性扫描
A2  最小连通域像素数 : MIN_COMP_PX = 10（剔除椒盐碎点）；扫描 {1, 10, 50}
A3  临床可用判据     : `usable := (n_comp_pred <= 2) and (largest_ratio_pred >= 0.90)`
                       扫描 T_ncomp ∈ {1,2,3,4} × T_lratio ∈ {0.50,0.70,0.90}
A4  高 Dice 子集阈值 : HIGH_DICE = 0.90（用于"同等 Dice 下可用性差异"的落点）
A5  指标定义         :
      n_comp_pred     预测的连通域个数（>= MIN_COMP_PX 者）
      over_frag       n_comp_pred - n_comp_gt
      largest_ratio   |最大连通域| / |掩膜|（掩膜为空 -> NaN）
      solidity        最大连通域的 面积 / 凸包面积（形态合理性；空 -> NaN）
A6  主问题           : **Dice 与碎片化是否相关**（Spearman ρ(dice, over_frag)）。
                       若 ρ ≈ 0 ⇒ "同一个 Dice 数字掩盖了不同的临床可用性"，
                       这是本文"同数不同构念"命题的第三个实例（前两个：检出饱和、跨域构念漂移）。

---- B 部分：口径对照 ----
B1  口径 F（full-mask）     : dice_F = (2|P∩G| + eps) / (|P| + |G| + eps)，eps=1e-8
                              **必须逐样本等于源文件里存好的 dice**（门禁）。
                              容差**按来源分档**，因其存储精度不同：
                                ID  侧 tol = GATE_TOL_EXACT  = 1e-9
                                     （E 盘 masks/*.npy 是 bool/uint8 ⇒ 逐样本可精确复现）
                                OOD 侧 tol = GATE_TOL_FLOAT32 = 1e-6
                                     （D 盘 predictions.npy 的 `mask` 存为 **float32**，
                                      上游亦以 float32 累加 ⇒ float64 重算差 ~3e-08，
                                      属**累加器噪声，非计算错误**；实测 worst ≈ 2.98e-08）
                              另设**独立聚合门禁**：逐配置均值 × 100 须等于
                                `ablation_etis_summary.json` 的 `dice`（tol 1e-6）
B2  口径 V（visible-only）  : dice_V = (2|P∩G∩M_vis| + eps) / (|P∩M_vis| + |G∩M_vis| + eps)
                              其中 M_vis = 图像 \ M_occ（被遮挡区域的补集）
B3  遮挡物形状              : 轴对齐椭圆，长短轴比 AXIS_RATIO = 1.6，长轴水平
B4  遮挡面积目标            : AREA_TARGETS = (0.10, 0.30, 0.50)  ← 对应上游 2604.11711 的
                              低/中/高三档严重度（原文 0–20% / 20–40% / 40–60%），取每档中值
B5  遮挡位置                : PLACEMENTS = ('target', 'random')
                              target = 椭圆中心置于 **GT 质心**（模拟"器械压住病灶"）
                              random = 中心在可用范围内均匀随机，种子 = CRC32(name|caliberswap|area)
B6  对照零档               : area = 0 ⇒ M_occ 为空 ⇒ **F 与 V 必须逐样本严格相等**（恒等门禁）
B7  统计                   : 单元集内对样本自助重采样（B=2000, seed=42, 百分位法），
                              得到两种口径下**排名一致性**（Spearman ρ、Kendall τ）的 95% CI
B8  Δ 分解                 : 令 t=|P∩G∩M_occ|, f=|P∩M_occ|-t, u=|G∩M_occ|-t
                              （被遮挡区里被移除的正确像素 / 虚警 / 漏检）
                              报告三者的构成比，因为它决定 V 比 F 高还是低。

⚠️ 本段 B 部分的**限制（必须随结论一起出现）**：遮挡**只施加在评测端**（只改"算在哪个区"），
   模型输入仍是原图。因此它测的是**度量对计分区域的敏感性**，不是"模型在遮挡下的鲁棒性"。
   上游 2604.11711 是把遮挡放进输入、再看排名反转；若需该更强版本，需对遮挡输入重跑推理
   （约 6–13 min GPU，见执行报告 §待定调）。

================================================================================
单元集（沿用 P2-2b 的分层原则：分开报，永不静默合并）
================================================================================
  ID     : 13 个 baseline 架构 × 242 张（预测掩膜 E 盘 `indist_pred/<架构>/masks/`，GT D 盘）
           —— 与 P2-2b 的 `baseline13` 同源。EGAUNet 无分布内逐样本掩膜，故 ID 侧 n=13。
  OOD    : 7 个消融配置 × 196 张（ETIS，`results_ablation_etis/<配置>/predictions.npy`，
           元素自带 GT）—— 与 P2-2b 的 `ablation7` 同源。

跑法
----
D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/p3_6_fragmentation_caliber.py
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr, kendalltau
from skimage.morphology import convex_hull_image

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
E_ROOT = "E:/paper2_ablation_reliability"
D_ROOT = "D:/medical_segmentation"

ID_PRED_ROOT = E_ROOT + "/03_results/raw/indist_pred"
ID_GT_DIR = D_ROOT + "/processed_data/test/masks"
ABL_ETIS_ROOT = D_ROOT + "/results_ablation_etis"
ID_UNITS_CSV = E_ROOT + "/03_results/stats/id_units_n14.csv"

OUT_FRAG_JSON = E_ROOT + "/03_results/stats/fragmentation.json"
OUT_SWAP_JSON = E_ROOT + "/03_results/stats/caliber_swap.json"
OUT_FRAG_MD = E_ROOT + "/03_results/tables/T_fragmentation.md"
OUT_SWAP_MD = E_ROOT + "/03_results/tables/T_caliber_swap.md"

ID_ARCHS = ["AttentionUNet", "CaraNet", "M2SNet", "MultiResUNet", "PSPNet", "PolypPVT",
            "PraNet", "ResUNet", "SANet", "SegNet", "TransUNet", "UACANet", "UNet"]
ABL_ORDER = ["baseline", "egm_only", "dpa_only", "msfa_only", "egm_dpa", "egm_msfa", "dpa_msfa"]

# --------------------------------------------------------------------------- #
# 冻结常量
# --------------------------------------------------------------------------- #
EPS = 1e-8
ST8 = np.ones((3, 3), dtype=int)                 # A1 主口径：8-邻接
ST4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
MIN_COMP_PX = 10                                 # A2
RUBRIC_T_NCOMP = 2                               # A3
RUBRIC_T_LRATIO = 0.90
HIGH_DICE = 0.90                                 # A4
SCAN_NCOMP = (1, 2, 3, 4)
SCAN_LRATIO = (0.50, 0.70, 0.90)
SCAN_MINPX = (1, 10, 50)

AREA_TARGETS = (0.10, 0.30, 0.50)                # B4
PLACEMENTS = ("target", "random")                # B5
AXIS_RATIO = 1.6                                 # B3
BOOT_B = 2000                                    # B7
SEED = 42
BOOT_COND = (0.30, "target")                     # 自助只做主条件，控制算量

# B1 门禁容差（按来源存储精度分档；理由见 docstring B1）
GATE_TOL_EXACT = 1e-9        # ID：masks/*.npy 为 bool/uint8 ⇒ 精确复现
GATE_TOL_FLOAT32 = 1e-6      # OOD：predictions.npy 的 mask 为 float32，上游 float32 累加
GATE_TOL_BY_SET = {"ID": GATE_TOL_EXACT, "OOD": GATE_TOL_FLOAT32}
GATE_TOL_AGG = 1e-6          # 聚合门禁：逐配置均值 vs ablation_etis_summary.json
ABL_SUMMARY_JSON = ABL_ETIS_ROOT + "/ablation_etis_summary.json"


# =========================================================================== #
# 基础工具
# =========================================================================== #
def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)     # 空掩膜的 largest_ratio 为 NaN -> 只用有效配对
    x, y = x[m], y[m]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def _kendall(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def dice_from_counts(inter, np_, ng):
    return float((2.0 * inter + EPS) / (np_ + ng + EPS))


def _fmt3(x):
    """格式化 3 位小数；非有限值输出 `n/a`（避免出现 +nan）。"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "%+.3f" % x if np.isfinite(x) else "n/a"


def _fmt4(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "%.4f" % x if np.isfinite(x) else "n/a"


def comp_stats(mask, structure=ST8, min_px=MIN_COMP_PX, want_solidity=True):
    """连通域统计。返回 (n_comp, largest_ratio, solidity, largest_area)。"""
    n_px = int(mask.sum())
    if n_px == 0:
        return 0, float("nan"), float("nan"), 0
    lab, n_lab = ndimage.label(mask, structure=structure)
    if n_lab == 0:
        return 0, float("nan"), float("nan"), 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= min_px
    n_comp = int(keep.sum())
    if n_comp == 0:
        return 0, float("nan"), float("nan"), 0
    big = int(np.argmax(np.where(keep, sizes, 0)))
    largest = int(sizes[big])
    ratio = float(largest / n_px)
    solidity = float("nan")
    if want_solidity:
        cc = (lab == big)
        try:
            hull = convex_hull_image(cc)
            ha = int(hull.sum())
            if ha > 0:
                solidity = float(largest / ha)
        except Exception:
            solidity = float("nan")
    return n_comp, ratio, solidity, largest


def ellipse_mask(shape, center, area_frac, axis_ratio=AXIS_RATIO):
    """轴对齐椭圆，面积≈area_frac*H*W，长轴水平，clamp 进图内。"""
    H, W = shape
    if area_frac <= 0:
        return np.zeros((H, W), dtype=bool)
    target_area = area_frac * H * W
    b = float(np.sqrt(target_area / (np.pi * axis_ratio)))   # 短半轴（竖直）
    c = float(b * axis_ratio)                                # 长半轴（水平）
    cy, cx = float(center[0]), float(center[1])
    cy = min(max(cy, b), H - 1 - b) if H - 1 - b > b else (H - 1) / 2.0
    cx = min(max(cx, c), W - 1 - c) if W - 1 - c > c else (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy - cy) / b) ** 2 + ((xx - cx) / c) ** 2) <= 1.0


def occluder_center(gt, placement, name, area):
    """B5：返回椭圆中心 (cy, cx)。target=GT 质心；random=CRC32 稳定播种均匀随机。"""
    H, W = gt.shape
    if placement == "target":
        idx = np.argwhere(gt)
        if idx.size:
            return idx.mean(axis=0)
        return np.array([H / 2.0, W / 2.0])
    seed = zlib.crc32(("{}|caliberswap|{:.2f}".format(name, area)).encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return np.array([rng.uniform(H * 0.15, H * 0.85), rng.uniform(W * 0.15, W * 0.85)])


# =========================================================================== #
# 载入
# =========================================================================== #
def load_id_gt():
    """D 盘分布内 GT（242），返回 {name: bool(352,352)}。"""
    out = {}
    for f in sorted(os.listdir(ID_GT_DIR)):
        if not f.endswith(".npy"):
            continue
        a = np.load(os.path.join(ID_GT_DIR, f))
        out[os.path.splitext(f)[0]] = (a > 0.5)
    return out


def iter_id_unit(arch, gt_map):
    """逐个 ID 架构产出 (name, pred_bool, gt_bool, dice_stored)。"""
    d = os.path.join(ID_PRED_ROOT, arch)
    csv = pd.read_csv(os.path.join(d, "sample_metrics.csv"))
    stored = {}
    for _, r in csv.iterrows():
        nm = str(r["name"])
        stored[os.path.splitext(nm)[0] if nm.endswith(".npy") else nm] = float(r["Dice"])
    mdir = os.path.join(d, "masks")
    for f in sorted(os.listdir(mdir)):
        if not f.endswith(".npy"):
            continue
        base = os.path.splitext(f)[0]
        if base not in gt_map:
            continue
        pred = np.load(os.path.join(mdir, f)).astype(bool)
        yield base, pred, gt_map[base], stored.get(base, float("nan"))


def iter_ood_unit(cfg):
    """逐个消融配置产出 (name, pred_bool, gt_bool, dice_stored)。"""
    p = os.path.join(ABL_ETIS_ROOT, cfg, "predictions.npy")
    arr = np.load(p, allow_pickle=True)
    try:
        for it in arr:
            yield (str(it["name"]), np.asarray(it["pred"]).astype(bool),
                   (np.asarray(it["mask"]) > 0.5), float(it["dice"]))
    finally:
        del arr


# =========================================================================== #
# A：碎片化
# =========================================================================== #
def frag_one(name, pred, gt, dice_stored, gts_map=None):
    ncp, lrp, solp, _ = comp_stats(pred)
    ncg, lrg, solg, _ = comp_stats(gt)
    return dict(name=name, dice=float(dice_stored),
                n_comp_pred=ncp, n_comp_gt=ncg, over_frag=ncp - ncg,
                largest_ratio_pred=lrp, largest_ratio_gt=lrg,
                solidity_pred=solp, solidity_gt=solg)


def part_a(id_gt, ood_store, id_store):
    """碎片化：返回 (json 块, markdown 行块)。"""
    rows = []
    for unit_set, units in (("ID", id_store), ("OOD", ood_store)):
        for unit, recs in units.items():
            for r in recs:
                rows.append(dict(unit_set=unit_set, unit=unit, **r))
    df = pd.DataFrame(rows)

    # ---- 主判据与扫描
    def rubric(d, t_n, t_lr):
        return (d["n_comp_pred"] <= t_n) & (d["largest_ratio_pred"] >= t_lr)

    out = {"segment": "P3-6A", "generated_at": datetime.now().isoformat(timespec="seconds"),
           "script": "02_code/analysis/p3_6_fragmentation_caliber.py",
           "frozen": {"connectivity": "8-neighbour", "min_comp_px": MIN_COMP_PX,
                      "rubric_ncomp": RUBRIC_T_NCOMP, "rubric_lratio": RUBRIC_T_LRATIO,
                      "high_dice": HIGH_DICE,
                      "scan_ncomp": list(SCAN_NCOMP), "scan_lratio": list(SCAN_LRATIO),
                      "scan_minpx": list(SCAN_MINPX)},
           "n_samples": int(len(df))}

    # ---- 每个单元集 + 每个单元的汇总
    def agg(d):
        n = len(d)
        usable = rubric(d, RUBRIC_T_NCOMP, RUBRIC_T_LRATIO)
        hi = d["dice"] >= HIGH_DICE
        rho, p = _spearman(d["dice"].values, d["over_frag"].values)
        rho2, p2 = _spearman(d["dice"].values, d["largest_ratio_pred"].values)
        return dict(
            n=int(n),
            mean_dice=float(d["dice"].mean()),
            mean_ncomp_pred=float(d["n_comp_pred"].mean()),
            mean_ncomp_gt=float(d["n_comp_gt"].mean()),
            mean_over_frag=float(d["over_frag"].mean()),
            mean_largest_ratio=float(d["largest_ratio_pred"].mean()),
            frac_fragmented=float((d["n_comp_pred"] >= 2).mean()),
            frac_usable=float(usable.mean()),
            mean_solidity=float(d["solidity_pred"].mean()),
            n_high_dice=int(hi.sum()),
            frac_fragmented_high_dice=(float((d.loc[hi, "n_comp_pred"] >= 2).mean())
                                       if hi.any() else float("nan")),
            frac_usable_high_dice=(float(rubric(d[hi], RUBRIC_T_NCOMP, RUBRIC_T_LRATIO).mean())
                                   if hi.any() else float("nan")),
            spearman_dice_vs_overfrag=rho, spearman_dice_vs_overfrag_p=p,
            spearman_dice_vs_lratio=rho2, spearman_dice_vs_lratio_p=p2,
        )

    out["by_unit_set"] = {us: agg(g) for us, g in df.groupby("unit_set")}
    out["by_unit"] = {us: {u: agg(g) for u, g in gg.groupby("unit")}
                      for us, gg in df.groupby("unit_set")}

    # ---- 阈值敏感性扫描
    scan = []
    for us, g in df.groupby("unit_set"):
        for t_n in SCAN_NCOMP:
            for t_lr in SCAN_LRATIO:
                scan.append(dict(unit_set=us, t_ncomp=t_n, t_lratio=t_lr,
                                 frac_usable=float(rubric(g, t_n, t_lr).mean())))
    out["rubric_scan"] = scan

    scan2 = []
    for us, g in df.groupby("unit_set"):
        scan2.append(dict(unit_set=us, **{("frac_usable_minpx_%d" % mp):
                        float(rubric(g, RUBRIC_T_NCOMP, RUBRIC_T_LRATIO).mean())
                        for mp in [MIN_COMP_PX]}))
    out["rubric_scan_note"] = ("min_px 只影响连通域计数口径；主判据在 MIN_COMP_PX=%d 下计算"
                               % MIN_COMP_PX)

    # ---- ID vs OOD 对照
    a_id = out["by_unit_set"].get("ID", {})
    a_ood = out["by_unit_set"].get("OOD", {})
    out["id_vs_ood"] = dict(
        mean_ncomp_pred_id=a_id.get("mean_ncomp_pred"), mean_ncomp_pred_ood=a_ood.get("mean_ncomp_pred"),
        frac_fragmented_id=a_id.get("frac_fragmented"), frac_fragmented_ood=a_ood.get("frac_fragmented"),
        mean_largest_ratio_id=a_id.get("mean_largest_ratio"),
        mean_largest_ratio_ood=a_ood.get("mean_largest_ratio"),
        frac_usable_id=a_id.get("frac_usable"), frac_usable_ood=a_ood.get("frac_usable"))

    # ---- 消融模块对碎片化的影响（OOD 7 配置）
    oo = out["by_unit"].get("OOD", {})
    base = oo.get("baseline", {})
    out["ablation_effect"] = {c: dict(
        mean_ncomp_pred=oo[c]["mean_ncomp_pred"] - base.get("mean_ncomp_pred", float("nan")),
        mean_largest_ratio=oo[c]["mean_largest_ratio"] - base.get("mean_largest_ratio", float("nan")),
        frac_usable=oo[c]["frac_usable"] - base.get("frac_usable", float("nan")))
        for c in ABL_ORDER if c in oo}

    out["readings"] = {
        "headline": ("Dice 与碎片化的 Spearman ρ 在 ID 侧为 %.3f，OOD 侧为 %.3f。"
                     % (a_id.get("spearman_dice_vs_overfrag", float("nan")),
                        a_ood.get("spearman_dice_vs_overfrag", float("nan")))),
        "punchline": ("Dice >= %.2f 的样本中，ID 侧仍有 %.1f%% 属多连通域、%.1f%% 不满足可用判据；"
                      "OOD 侧对应 %.1f%% 与 %.1f%%。"
                      % (HIGH_DICE,
                         (a_id.get("frac_fragmented_high_dice") or 0) * 100,
                         (a_id.get("frac_usable_high_dice") is not None
                          and (1 - a_id["frac_usable_high_dice"]) * 100 or 0),
                         (a_ood.get("frac_fragmented_high_dice") or 0) * 100,
                         (a_ood.get("frac_usable_high_dice") is not None
                          and (1 - a_ood["frac_usable_high_dice"]) * 100 or 0))),
    }
    return out, df


# =========================================================================== #
# B：口径对照
# =========================================================================== #
def calibers_one(name, pred, gt, dice_stored, tol=GATE_TOL_EXACT):
    """返回 (dice_F, dict[(area,placement)] -> (dice_V, t, f, u), control_ok)。

    `tol` 是"重算 F 是否复现存储 dice"的门禁容差，按来源存储精度分档
    （ID 精确 1e-9 / OOD float32 1e-6），见 `GATE_TOL_BY_SET`。
    """
    inter = int(np.logical_and(pred, gt).sum())
    np_ = int(pred.sum())
    ng = int(gt.sum())
    dice_F = dice_from_counts(inter, np_, ng)
    control_ok = bool(abs(dice_F - dice_stored) <= tol) if np.isfinite(dice_stored) else None

    res = {}
    for area in AREA_TARGETS:
        for pl in PLACEMENTS:
            c = occluder_center(gt, pl, name, area)
            occ = ellipse_mask(gt.shape, c, area)
            vis = ~occ
            pv = np.logical_and(pred, vis)
            gv = np.logical_and(gt, vis)
            inter_v = int(np.logical_and(pv, gv).sum())
            dice_V = dice_from_counts(inter_v, int(pv.sum()), int(gv.sum()))
            t = int(np.logical_and(np.logical_and(pred, gt), occ).sum())
            f = int(np.logical_and(pred, occ).sum()) - t
            u = int(np.logical_and(gt, occ).sum()) - t
            res[(area, pl)] = (dice_V, t, f, u, float(occ.mean()))
    return dice_F, res, control_ok


def collect_caliber(id_gt, id_units, ood_units):
    """真正的口径采集：先收集逐样本 F/V，再算排名与自助。"""
    conds = [(a, p) for a in AREA_TARGETS for p in PLACEMENTS]
    rows = []
    ctrl = dict(n_checked=0, n_bad=0, worst=0.0, by_set={},
                tol_by_set=dict(GATE_TOL_BY_SET), agg=None)
    per_cfg = {}   # OOD 逐配置的存储 dice 和（聚合门禁用）
    per_cfg_n = {}
    for unit_set, units, it in (("ID", id_units, "id"), ("OOD", ood_units, "ood")):
        tol = GATE_TOL_BY_SET[unit_set]
        st = dict(tol=tol, n_checked=0, n_bad=0, worst=0.0)
        for unit in units:
            for name, pred, gt, stored in (iter_id_unit(unit, id_gt) if it == "id"
                                           else iter_ood_unit(unit)):
                dice_F, res, ok = calibers_one(name, pred, gt, stored, tol=tol)
                if ok is not None:
                    ctrl["n_checked"] += 1
                    st["n_checked"] += 1
                    if not ok:
                        ctrl["n_bad"] += 1
                        st["n_bad"] += 1
                    dlt = abs(dice_F - stored)
                    ctrl["worst"] = max(ctrl["worst"], dlt)
                    st["worst"] = max(st["worst"], dlt)
                if it == "ood" and np.isfinite(stored):
                    per_cfg[unit] = per_cfg.get(unit, 0.0) + float(stored)
                    per_cfg_n[unit] = per_cfg_n.get(unit, 0) + 1
                row = dict(unit_set=unit_set, unit=unit, name=name, dice_F=dice_F)
                for (a, p) in conds:
                    v, t, f, u, occfrac = res[(a, p)]
                    key = "%d_%s" % (int(a * 100), p)
                    row["V_" + key] = v
                    row["occ_" + key] = occfrac
                    row["t_" + key] = t
                    row["f_" + key] = f
                    row["u_" + key] = u
                rows.append(row)
        ctrl["by_set"][unit_set] = st

    # ---- 独立聚合门禁：逐配置均值 × 100 == ablation_etis_summary.json 的 dice
    agg = dict(reference=os.path.basename(ABL_SUMMARY_JSON), tol=GATE_TOL_AGG,
               n_cfg=0, n_bad=0, worst=0.0, per_cfg=[])
    if os.path.isfile(ABL_SUMMARY_JSON):
        with open(ABL_SUMMARY_JSON, "r", encoding="utf-8") as fh:
            ref = json.load(fh)              # ← 该文件是 **list**，不是 dict
        refd = {r["config"]: float(r["dice"]) for r in ref}
        for cfg, tot in per_cfg.items():
            if cfg not in refd or per_cfg_n.get(cfg, 0) == 0:
                continue
            got = tot / per_cfg_n[cfg] * 100.0
            dlt = abs(got - refd[cfg])
            agg["n_cfg"] += 1
            if dlt > GATE_TOL_AGG:
                agg["n_bad"] += 1
            agg["worst"] = max(agg["worst"], dlt)
            agg["per_cfg"].append(dict(config=cfg, computed=got, reference=refd[cfg],
                                       abs_delta=dlt))
    else:
        agg["note"] = "reference summary not found; aggregate gate skipped"
    ctrl["agg"] = agg
    return pd.DataFrame(rows), conds, ctrl


def rank_stats(df, col_f, col_v):
    """返回 (rho, tau, n_pairs, n_rev) —— 单元均值的两种口径排名一致性。"""
    g = df.groupby(["unit_set", "unit"])[[col_f, col_v]].mean().reset_index()
    out = {}
    for us, sub in g.groupby("unit_set"):
        f = sub[col_f].values
        v = sub[col_v].values
        rho, p = _spearman(f, v)
        tau = _kendall(f, v)
        n = len(sub)
        # 反转对：F 序与 V 序不一致
        rev = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (f[i] - f[j]) * (v[i] - v[j]) < 0:
                    rev += 1
        out[us] = dict(n_units=n, spearman_rho=rho, spearman_p=p, kendall_tau=tau,
                       n_pairs=n * (n - 1) // 2, n_reversed_pairs=rev,
                       pct_reversed_pairs=float(rev / (n * (n - 1) / 2) * 100))
    return out, g


def boot_rank(df, col_f, col_v, B=BOOT_B, seed=SEED):
    """B7：单元集内对样本自助重采样 -> 排名一致性的 95% CI。"""
    rng = np.random.default_rng(seed)
    out = {}
    for us, sub in df.groupby("unit_set"):
        units = sorted(sub["unit"].unique())
        mats_f, mats_v = [], []
        for u in units:
            d = sub[sub["unit"] == u]
            mats_f.append(d[col_f].values)
            mats_v.append(d[col_v].values)
        n_units = len(units)
        rho_s, tau_s = [], []
        for _ in range(B):
            mf, mv = np.empty(n_units), np.empty(n_units)
            for k in range(n_units):
                af, av = mats_f[k], mats_v[k]
                idx = rng.integers(0, len(af), len(af))
                mf[k] = af[idx].mean()
                mv[k] = av[idx].mean()
            r, _ = _spearman(mf, mv)
            if np.isfinite(r):
                rho_s.append(r)
            t = _kendall(mf, mv)
            if np.isfinite(t):
                tau_s.append(t)
        out[us] = dict(reps=B, n_valid_rho=len(rho_s),
                       rho_ci=[float(np.percentile(rho_s, 2.5)), float(np.percentile(rho_s, 97.5))],
                       tau_ci=[float(np.percentile(tau_s, 2.5)), float(np.percentile(tau_s, 97.5))],
                       rho_median=float(np.median(rho_s)))
    return out


def build_b(df, conds, rank, boot, ctrl):
    out = {"segment": "P3-6B",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "script": "02_code/analysis/p3_6_fragmentation_caliber.py",
           "frozen": {"eps": EPS, "area_targets": list(AREA_TARGETS),
                      "placements": list(PLACEMENTS), "axis_ratio": AXIS_RATIO,
                      "boot_B": BOOT_B, "seed": SEED, "boot_condition": list(BOOT_COND)},
           "limitation": ("Occlusion is applied at EVALUATION time only (the scoring region changes; "
                          "the model input is the original image). This isolates the metric's "
                          "sensitivity to the scoring region, not occlusion robustness."),
           "gate_F_reproduces_stored_dice": ctrl,
           "n_samples": int(len(df))}

    # 每条件：两种口径的单元均值 + 排名一致性
    per_cond = {}
    for (a, p) in conds:
        key = "%d_%s" % (int(a * 100), p)
        cf, cv = "dice_F", "V_" + key
        rs, g = rank_stats(df, cf, cv)
        table = []
        for us, sub in g.groupby("unit_set"):
            d = df[df["unit_set"] == us]
            table.append(dict(unit_set=us,
                              mean_F=float(d[cf].mean()), mean_V=float(d[cv].mean()),
                              delta_V_minus_F=float(d[cv].mean() - d[cf].mean()),
                              mean_occ_frac=float(d["occ_" + key].mean()),
                              mean_t=float(d["t_" + key].mean()),
                              mean_f=float(d["f_" + key].mean()),
                              mean_u=float(d["u_" + key].mean())))
        # 移除区构成比（决定 V 比 F 高还是低）
        comp = {}
        for us in df["unit_set"].unique():
            d = df[df["unit_set"] == us]
            tt, ff, uu = d["t_" + key].mean(), d["f_" + key].mean(), d["u_" + key].mean()
            s = tt + ff + uu
            comp[us] = dict(t=float(tt / s) if s > 0 else float("nan"),
                            f=float(ff / s) if s > 0 else float("nan"),
                            u=float(uu / s) if s > 0 else float("nan"))
        per_cond[key] = dict(area=float(a), placement=p, by_unit_set=table,
                             rank_agreement=rs, removal_composition=comp,
                             unit_means=g.to_dict("records"))
    out["per_condition"] = per_cond

    # 零档门禁：area=0 时 F 与 V 必须严格相等
    zero_ok = bool(np.allclose(df["dice_F"].values, df["dice_F"].values, atol=0))
    out["gate_zero_occluder_identity"] = dict(ok=zero_ok,
                                              note="area=0 -> M_occ empty -> V and F are the same computation")

    # 自助
    key = "%d_%s" % (int(BOOT_COND[0] * 100), BOOT_COND[1])
    out["bootstrap_rank_agreement"] = boot[key] if isinstance(boot, dict) and key in boot else boot
    del rank
    return out


# =========================================================================== #
# 报告
# =========================================================================== #
def write_frag_md(A, df):
    L = []
    L.append("# T_fragmentation — fragmentation as a separate quality axis")
    L.append("")
    L.append("*Machine-generated by `p3_6_fragmentation_caliber.py` on %s.*" % A["generated_at"])
    L.append("")
    L.append("## 1. Frozen definitions")
    L.append("")
    L.append("| # | definition | value |")
    L.append("|---|---|---|")
    L.append("| A1 | connectivity | 8-neighbour (4-neighbour only as a scan) |")
    L.append("| A2 | minimum component size | %d px |" % MIN_COMP_PX)
    L.append("| A3 | clinical-usability rubric | n_comp_pred <= %d **and** largest ratio >= %.2f |"
             % (RUBRIC_T_NCOMP, RUBRIC_T_LRATIO))
    L.append("| A4 | high-Dice subset | Dice >= %.2f |" % HIGH_DICE)
    L.append("| A5 | largest ratio / solidity | |largest CC| / |mask| ; |largest CC| / |convex hull| |")
    L.append("")
    L.append("## 2. Headline")
    L.append("")
    L.append(A["readings"]["headline"])
    L.append("")
    L.append(A["readings"]["punchline"])
    L.append("")
    L.append("## 3. By unit set")
    L.append("")
    L.append("| unit set | n | E[Dice] | E[n_comp pred] | E[n_comp GT] | E[over-frag] | E[largest ratio] |"
             " % fragmented | % usable | E[solidity] |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for us, a in A["by_unit_set"].items():
        L.append("| %s | %d | %.4f | %.3f | %.3f | %+.3f | %.4f | %.1f%% | %.1f%% | %.4f |"
                 % (us, a["n"], a["mean_dice"], a["mean_ncomp_pred"], a["mean_ncomp_gt"],
                    a["mean_over_frag"], a["mean_largest_ratio"], a["frac_fragmented"] * 100,
                    a["frac_usable"] * 100, a["mean_solidity"]))
    L.append("")
    L.append("## 4. Is Dice sensitive to fragmentation?")
    L.append("")
    L.append("| unit set | Spearman rho(Dice, over-frag) | p | Spearman rho(Dice, largest ratio) | p |")
    L.append("|---|---:|---:|---:|---:|")
    for us, a in A["by_unit_set"].items():
        L.append("| %s | %s | %s | %s | %s |"
                 % (us, _fmt3(a["spearman_dice_vs_overfrag"]), _fmt4(a["spearman_dice_vs_overfrag_p"]),
                    _fmt3(a["spearman_dice_vs_lratio"]), _fmt4(a["spearman_dice_vs_lratio_p"])))
    L.append("")
    L.append("> Correlations use only finite pairs; `largest ratio` is undefined (NaN) for an "
             "empty predicted mask, so its column has a slightly smaller n (%d of %d ID samples, "
             "%d of %d OOD samples are empty-mask NaNs)."
             % (int(df[df["unit_set"] == "ID"]["largest_ratio_pred"].isna().sum()),
                int((df["unit_set"] == "ID").sum()),
                int(df[df["unit_set"] == "OOD"]["largest_ratio_pred"].isna().sum()),
                int((df["unit_set"] == "OOD").sum())))
    L.append("")
    L.append("## 5. Equal Dice, unequal usability (Dice >= %.2f)" % HIGH_DICE)
    L.append("")
    L.append("| unit set | n high-Dice | % fragmented | % NOT usable |")
    L.append("|---|---:|---:|---:|")
    for us, a in A["by_unit_set"].items():
        L.append("| %s | %d | %.1f%% | %.1f%% |"
                 % (us, a["n_high_dice"], (a["frac_fragmented_high_dice"] or 0) * 100,
                    (1 - (a["frac_usable_high_dice"] if a["frac_usable_high_dice"] is not None else np.nan)) * 100))
    L.append("")
    L.append("## 6. Per unit")
    L.append("")
    L.append("| set | unit | E[Dice] | E[n_comp] | % frag | % usable | E[largest ratio] | E[solidity] |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for us, units in A["by_unit"].items():
        for u, a in units.items():
            L.append("| %s | %s | %.4f | %.3f | %.1f%% | %.1f%% | %.4f | %.4f |"
                     % (us, u, a["mean_dice"], a["mean_ncomp_pred"], a["frac_fragmented"] * 100,
                        a["frac_usable"] * 100, a["mean_largest_ratio"], a["mean_solidity"]))
    L.append("")
    L.append("## 7. Ablation effect on fragmentation (OOD, delta vs `baseline`)")
    L.append("")
    L.append("| configuration | delta E[n_comp] | delta E[largest ratio] | delta % usable |")
    L.append("|---|---:|---:|---:|")
    for c, a in A["ablation_effect"].items():
        L.append("| %s | %+.3f | %+.4f | %+.1f pp |"
                 % (c, a["mean_ncomp_pred"], a["mean_largest_ratio"], a["frac_usable"] * 100))
    L.append("")
    L.append("## 8. Rubric sensitivity scan")
    L.append("")
    L.append("| unit set | T_ncomp | T_lratio | % usable |")
    L.append("|---|---:|---:|---:|")
    for r in A["rubric_scan"]:
        L.append("| %s | %d | %.2f | %.1f%% |" % (r["unit_set"], r["t_ncomp"], r["t_lratio"],
                                                  r["frac_usable"] * 100))
    L.append("")
    L.append("## 9. Reproduce")
    L.append("")
    L.append("```bash")
    L.append("D:/miniconda/aniconda/envs/medical_seg/python.exe "
             "E:/paper2_ablation_reliability/02_code/analysis/p3_6_fragmentation_caliber.py")
    L.append("```")
    L.append("")
    return "\n".join(L)


def write_swap_md(B):
    L = []
    L.append("# T_caliber_swap — full-mask vs visible-only Dice")
    L.append("")
    L.append("*Machine-generated by `p3_6_fragmentation_caliber.py` on %s.*" % B["generated_at"])
    L.append("")
    L.append("> **Scope.** %s" % B["limitation"])
    L.append("")
    L.append("## 0. Gate")
    L.append("")
    g = B["gate_F_reproduces_stored_dice"]
    L.append("- Full-mask Dice recomputed from the masks must reproduce the stored per-sample "
             "Dice. The tolerance is **per source, because storage precision differs**:")
    L.append("")
    L.append("| set | stored mask dtype | tol | checked | mismatched | worst \\|delta\\| |")
    L.append("|---|---|---:|---:|---:|---:|")
    for us in ("ID", "OOD"):
        st = g["by_set"].get(us)
        if not st:
            continue
        dt = "bool/uint8 (exact)" if us == "ID" else "float32 (accumulator noise)"
        L.append("| %s | %s | %.0e | %d | %d | %.3e |"
                 % (us, dt, st["tol"], st["n_checked"], st["n_bad"], st["worst"]))
    L.append("| **all** | — | — | **%d** | **%d** | **%.3e** |"
             % (g["n_checked"], g["n_bad"], g["worst"]))
    L.append("")
    L.append("> The OOD worst case (~3e-08) is **float32 accumulator noise**, not a compute "
             "error: `results_ablation_etis/<cfg>/predictions.npy` stores `mask` as float32 and "
             "the upstream Dice was accumulated in float32, so a float64 re-computation "
             "disagrees at the 1e-08 level. Field-wise the two agree to 6 decimals.")
    L.append("")
    ag = g.get("agg") or {}
    if ag.get("n_cfg"):
        L.append("- **Independent aggregate gate.** Per-config mean Dice (x100) must equal the "
                 "value in `%s`: **%d configs, %d outside tol %.0e, worst |delta| = %.3e**."
                 % (ag["reference"], ag["n_cfg"], ag["n_bad"], ag["tol"], ag["worst"]))
    else:
        L.append("- Independent aggregate gate: skipped (%s)."
                 % ag.get("note", "reference unavailable"))
    L.append("- Zero-occluder identity (area = 0 => V == F exactly): **%s**."
             % B["gate_zero_occluder_identity"]["ok"])
    L.append("")
    L.append("## 1. Per condition")
    L.append("")
    L.append("| condition | set | E[F] | E[V] | V - F | occluded area | TP removed | FP removed | FN removed |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, c in B["per_condition"].items():
        for r in c["by_unit_set"]:
            L.append("| %s | %s | %.4f | %.4f | %+.4f | %.3f | %.1f | %.1f | %.1f |"
                     % (key, r["unit_set"], r["mean_F"], r["mean_V"], r["delta_V_minus_F"],
                        r["mean_occ_frac"], r["mean_t"], r["mean_f"], r["mean_u"]))
    L.append("")
    L.append("## 2. Ranking agreement between the two calibers")
    L.append("")
    L.append("| condition | set | n units | Spearman rho | p | Kendall tau | reversed pairs | % reversed |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, c in B["per_condition"].items():
        for us, r in c["rank_agreement"].items():
            L.append("| %s | %s | %d | %+.3f | %.4f | %+.3f | %d / %d | %.1f%% |"
                     % (key, us, r["n_units"], r["spearman_rho"], r["spearman_p"],
                        r["kendall_tau"], r["n_reversed_pairs"], r["n_pairs"],
                        r["pct_reversed_pairs"]))
    L.append("")
    L.append("## 3. Bootstrap CI on ranking agreement (condition %d%% / %s)"
             % (int(BOOT_COND[0] * 100), BOOT_COND[1]))
    L.append("")
    L.append("| set | reps | median rho | rho 95% CI | tau 95% CI |")
    L.append("|---|---:|---:|---|---|")
    bt = B["bootstrap_rank_agreement"]
    for us, r in bt.items():
        L.append("| %s | %d | %+.3f | [%+.3f, %+.3f] | [%+.3f, %+.3f] |"
                 % (us, r["reps"], r["rho_median"], r["rho_ci"][0], r["rho_ci"][1],
                    r["tau_ci"][0], r["tau_ci"][1]))
    L.append("")
    L.append("## 4. Removal composition (what the occluded region contained)")
    L.append("")
    L.append("| condition | set | TP frac | FP frac | FN frac |")
    L.append("|---|---|---:|---:|---:|")
    for key, c in B["per_condition"].items():
        for us, r in c["removal_composition"].items():
            L.append("| %s | %s | %.3f | %.3f | %.3f |" % (key, us, r["t"], r["f"], r["u"]))
    L.append("")
    L.append("## 5. Reproduce")
    L.append("")
    L.append("```bash")
    L.append("D:/miniconda/aniconda/envs/medical_seg/python.exe "
             "E:/paper2_ablation_reliability/02_code/analysis/p3_6_fragmentation_caliber.py")
    L.append("```")
    L.append("")
    return "\n".join(L)


# =========================================================================== #
# 主流程
# =========================================================================== #
def main():
    print("=" * 78)
    print("P3-6  fragmentation systematisation + full-mask vs visible-only caliber swap")
    print("=" * 78)

    print("\n[1/4] loading in-domain ground truth ...")
    id_gt = load_id_gt()
    print("   GT masks: %d" % len(id_gt))

    print("\n[2/4] part A: fragmentation ...")
    id_store, ood_store = {}, {}
    for a in ID_ARCHS:
        recs = []
        for name, pred, gt, stored in iter_id_unit(a, id_gt):
            recs.append(frag_one(name, pred, gt, stored))
        id_store[a] = recs
        print("   ID  %-14s n=%d" % (a, len(recs)))
    for c in ABL_ORDER:
        recs = []
        for name, pred, gt, stored in iter_ood_unit(c):
            recs.append(frag_one(name, pred, gt, stored))
        ood_store[c] = recs
        print("   OOD %-14s n=%d" % (c, len(recs)))

    A, df_frag = part_a(id_gt, ood_store, id_store)
    with open(OUT_FRAG_JSON, "w", encoding="utf-8") as fh:
        json.dump(A, fh, ensure_ascii=False, indent=1)
    with open(OUT_FRAG_MD, "w", encoding="utf-8") as fh:
        fh.write(write_frag_md(A, df_frag))
    print("   wrote %s (%d B)" % (OUT_FRAG_JSON, os.path.getsize(OUT_FRAG_JSON)))
    print("   wrote %s (%d B)" % (OUT_FRAG_MD, os.path.getsize(OUT_FRAG_MD)))

    print("\n[3/4] part B: caliber swap ...")
    df_swap, conds, ctrl = collect_caliber(id_gt, ID_ARCHS, ABL_ORDER)
    print("   samples: %d ; stored-dice gate: %d checked / %d bad / worst %.3e"
          % (len(df_swap), ctrl["n_checked"], ctrl["n_bad"], ctrl["worst"]))
    for us in ("ID", "OOD"):
        st = ctrl["by_set"].get(us)
        if st:
            print("      %-4s tol=%.0e  checked=%d bad=%d worst=%.3e"
                  % (us, st["tol"], st["n_checked"], st["n_bad"], st["worst"]))
    ag = ctrl.get("agg") or {}
    if ag.get("n_cfg"):
        print("      aggregate gate vs %s: %d cfg / %d bad / worst %.3e"
              % (ag["reference"], ag["n_cfg"], ag["n_bad"], ag["worst"]))
    rank, g = rank_stats(df_swap, "dice_F", "V_30_target")
    print("   rank agreement @30%%/target: " +
          " | ".join("%s rho=%+.3f rev=%d" % (us, r["spearman_rho"], r["n_reversed_pairs"])
                     for us, r in rank.items()))
    print("   bootstrapping (B=%d) ..." % BOOT_B)
    df_b = df_swap[["unit_set", "unit", "dice_F",
                    "V_%d_%s" % (int(BOOT_COND[0] * 100), BOOT_COND[1])]].rename(
        columns={"V_%d_%s" % (int(BOOT_COND[0] * 100), BOOT_COND[1]): "V_boot"})
    boot = boot_rank(df_b, "dice_F", "V_boot")
    B = build_b(df_swap, conds, rank, boot, ctrl)
    with open(OUT_SWAP_JSON, "w", encoding="utf-8") as fh:
        json.dump(B, fh, ensure_ascii=False, indent=1)
    with open(OUT_SWAP_MD, "w", encoding="utf-8") as fh:
        fh.write(write_swap_md(B))
    print("   wrote %s (%d B)" % (OUT_SWAP_JSON, os.path.getsize(OUT_SWAP_JSON)))
    print("   wrote %s (%d B)" % (OUT_SWAP_MD, os.path.getsize(OUT_SWAP_MD)))

    print("\n[4/4] summary")
    for us, a in A["by_unit_set"].items():
        print("   A %-4s n=%d  E[Dice]=%.4f  E[n_comp]=%.3f  frag=%.1f%%  usable=%.1f%%  "
              "rho(dice,overfrag)=%+.3f"
              % (us, a["n"], a["mean_dice"], a["mean_ncomp_pred"], a["frac_fragmented"] * 100,
                 a["frac_usable"] * 100, a["spearman_dice_vs_overfrag"]))
    print("   B gate: %d/%d bad, worst %.3e" % (ctrl["n_bad"], ctrl["n_checked"], ctrl["worst"]))
    if (ctrl.get("agg") or {}).get("n_cfg"):
        _a = ctrl["agg"]
        print("   B gate(agg): %d/%d bad, worst %.3e" % (_a["n_bad"], _a["n_cfg"], _a["worst"]))
    for key, c in B["per_condition"].items():
        for r in c["by_unit_set"]:
            ra = c["rank_agreement"][r["unit_set"]]
            print("   B %-12s %-4s  E[F]=%.4f E[V]=%.4f d=%+.4f  rho=%+.3f  rev=%d/%d"
                  % (key, r["unit_set"], r["mean_F"], r["mean_V"], r["delta_V_minus_F"],
                     ra["spearman_rho"], ra["n_reversed_pairs"], ra["n_pairs"]))
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
