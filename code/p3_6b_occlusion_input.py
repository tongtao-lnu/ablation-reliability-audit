# -*- coding: utf-8 -*-
"""
文件名: p3_6b_occlusion_input.py
功能: 【论文二 P3-6b】遮挡**进输入**（真鲁棒性版本）—— 把 P3-6 的评测端遮挡升级为输入端遮挡

背景
----
P3-6 的遮挡**只施加在评测端**（只改"算在哪个区"，`M_vis = ¬M_occ`），**模型输入仍是原图**
⇒ 它测的是「**度量对计分区域的敏感性**」，**不是**「模型在遮挡下的鲁棒性」。
上游 arXiv:2604.11711 (*Seeing Through the Tool*) 是把遮挡**放进输入**再看排名反转。
P3-6b 补这一段，使 P3-6 §5.3 的"域间差距压缩/符号翻转"能以
"**输入侧遮挡下同样发生（或更强）**"的形态进主结果位（或如实降格）。

冻结规格: 00_docs/P3-6b规格冻结_2026-09-15.md（G6：本脚本常量与之一一对应，跑后不得回改）

★ 唯一变动原则
--------------
除"遮挡施加位置"外，一切与 P3-6 **逐位相同**：
  * 同一 `ellipse_mask`（AXIS_RATIO=1.6、长轴水平、clamp 进图）
  * 同一 `occluder_center`（target=GT 质心；random=**同一 CRC32 种子串** "{name}|caliberswap|{area:.2f}"）
  * 同一面积档 {0.10,0.30,0.50}、同一位置档 {target,random}
  * 同一模型权重、同一前向协议（autocast / sigmoid>0.5）
⇒ 每张图拿到的椭圆与 P3-6 **几何完全相同**；两次实验之间只有"椭圆挖在输入上、还是只从计分里剔除"这一处不同。

口径
----
条件（7 档）: 0_clean + {10,30,50} × {target,random}
填充（预注册）: zero（0.0，**主**）／mean（每图均值，**次**）
    x' = x.copy();  x'[M_occ] = fill;      P' = (sigmoid(f(x')) > 0.5)
    dice_F' = (2|P'∩G| + ε) / (|P'| + |G| + ε)
    dice_V' = (2|P'∩G∩M_vis| + ε) / (|P'∩M_vis| + |G∩M_vis| + ε)
    GT 永不遮挡（遮挡只作用于输入图像与计分区域）

单元集（沿用 P2-2b 分层，永不静默合并）
    ID : 13 baseline 架构 × 242 张   = 3,146
    OOD: 7 消融配置 × 196 张 ETIS    = 1,372   （与 P3-6 / P2-2b 的 ablation7 同源）
    合计 4,518

门禁（跑前冻结，任一不过 ⇒ 停手上报）
    G1 恒等档重跑须复现冻结预测    逐样本 max|Δdice| ≤ 1e-6   ← 顶梁门
    G2 遮挡确实生效                occ_frac > 0 且输入确有变化
    G3 area=0 ⇒ V ≡ F              逐样本严格相等
    G4 恒等档集合均值复现 P3-6      ID 0.8411039998288351 / OOD 0.5591206907338347 (tol 1e-6)
    G6 逐样本计数自洽              t+f_occ == |P'∩M_occ|、t+u_occ == |G∩M_occ|

输出（落盘 E）
    03_results/raw/occlusion_input/samples_<fill>.csv.gz   逐样本长表
    03_results/raw/occlusion_input/units_<fill>.json       逐单元统计（供复核）
    03_results/stats/occlusion_input.json                  总结果
    03_results/tables/T_p3_6b_input_occlusion.md           结果表

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    # 冒烟（1 架构 + 1 配置，1 条件，校吞吐）
    $PY .../p3_6b_occlusion_input.py --smoke
    # 全量
    $PY .../p3_6b_occlusion_input.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# import 副作用: chdir 到 D:/medical_segmentation（故 D 盘相对路径可用，E 盘一律走 X.eabs）
import p31_cross_eval as X                       # noqa: E402
import p3_7_stress_test as S                     # noqa: E402  (load_id_test / PIXEL_THRESH)
import p3_6_fragmentation_caliber as C           # noqa: E402  (几何与度量单一真源)

from models.ablation_models import get_ablation_model  # noqa: E402

# --------------------------------------------------------------------------- #
# 冻结常数（与 00_docs/P3-6b规格冻结_2026-09-15.md 一一对应）
# --------------------------------------------------------------------------- #
E_ROOT = "E:/paper2_ablation_reliability"
D_ROOT = "D:/medical_segmentation"

ID_PRED_ROOT = E_ROOT + "/03_results/raw/indist_pred"
ABL_ETIS_ROOT = D_ROOT + "/results_ablation_etis"
BASELINE_DIR = "experiments/baseline"            # 相对 D_ROOT
ABL_DIR = "experiments/ablation"                 # 相对 D_ROOT
CKPT_NAME = "best_model.pth"

OUT_DIR_REL = "03_results/raw/occlusion_input"
OUT_JSON_REL = "03_results/stats/occlusion_input.json"
OUT_MD_REL = "03_results/tables/T_p3_6b_input_occlusion.md"

# 13 baseline（ID 侧；与 P3-6 的 ID 单元集逐字一致）
ID_ARCHS = ["AttentionUNet", "CaraNet", "M2SNet", "MultiResUNet", "PSPNet", "PolypPVT",
            "PraNet", "ResUNet", "SANet", "SegNet", "TransUNet", "UACANet", "UNet"]
# 7 消融配置（OOD 侧；与 P3-6 / P2-2b 的 ablation7 逐字一致，第 8 格不进主表）
OOD_CFGS = ["baseline", "egm_only", "dpa_only", "msfa_only",
            "egm_dpa", "egm_msfa", "dpa_msfa"]

AREA_TARGETS = C.AREA_TARGETS                    # (0.10, 0.30, 0.50)
PLACEMENTS = C.PLACEMENTS                        # ("target", "random")
AXIS_RATIO = C.AXIS_RATIO                        # 1.6
EPS = C.EPS                                      # 1e-8
PIXEL_THRESH = 0.5

FILLS_DEFAULT = ("zero", "mean")
PRIMARY_FILL = "zero"

GATE_IDENT_TOL = 1e-6        # G1：逐样本 dice 绝对差
GATE_EF_TOL = 1e-6           # G4：集合均值绝对差
P3_6_REF_EF = {"ID": 0.8411039998288351, "OOD": 0.5591206907338347}   # 源: caliber_swap.json

# 条件清单（顺序与 P3-6 的键名生成一致）
CONDITIONS = [("0_clean", None, 0.0)] + [
    ("%d_%s" % (int(a * 100), p), p, a) for a in AREA_TARGETS for p in PLACEMENTS
]


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def cond_key(area, placement):
    return "%d_%s" % (int(round(area * 100)), placement)


def make_fill(img, occ, fill):
    """把被遮挡处改写为 fill，返回新数组（不改原图）。"""
    out = img.copy()
    if occ.any():
        if fill == "zero":
            out[occ] = 0.0
        elif fill == "mean":
            out[occ] = float(img.mean())
        else:
            raise ValueError("未知 fill: %s" % fill)
    return out


def occluder(gt, name, area, placement):
    """用 P3-6 的同一函数与同一种子串生成 occ / vis。area<=0 ⇒ 全 False。"""
    if area <= 0:
        return np.zeros(gt.shape, dtype=bool)
    c = C.occluder_center(gt, placement, name, area)
    return C.ellipse_mask(gt.shape, c, area)


def counts(pred, gt, occ):
    """返回 10 个混淆计数；occ 为空时 V 与 F 恒等（G3 由其自证）。"""
    vis = ~occ
    inter = int(np.logical_and(pred, gt).sum())
    p_tot = int(pred.sum())
    g_tot = int(gt.sum())

    pv = np.logical_and(pred, vis)
    gv = np.logical_and(gt, vis)
    inter_v = int(np.logical_and(pv, gv).sum())

    # 遮挡区三分（P3-6 B8 同款）
    pg = np.logical_and(pred, gt)
    t = int(np.logical_and(pg, occ).sum())
    f_occ = int(np.logical_and(np.logical_and(pred, ~gt), occ).sum())
    u_occ = int(np.logical_and(np.logical_and(~pred, gt), occ).sum())

    return dict(inter=inter, p_tot=p_tot, g_tot=g_tot,
                inter_v=inter_v, p_v=int(pv.sum()), g_v=int(gv.sum()),
                t=t, f_occ=f_occ, u_occ=u_occ,
                occ_px=int(occ.sum()), pos_occ=int(np.logical_and(pred, occ).sum()))


# --------------------------------------------------------------------------- #
# 单模型推理
# --------------------------------------------------------------------------- #
def _forward(model, img, device):
    """img: (H,W,3) f32 [0,1] -> pred bool (H,W)。协议与既有产物逐字一致。"""
    xt = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
    with torch.no_grad(), autocast():
        out = model(xt)
        if isinstance(out, tuple):
            out = out[0]
    prob = torch.sigmoid(out.float())[0, 0].cpu().numpy()
    return prob > PIXEL_THRESH


def build_id_model(arch, device):
    wpath = os.path.join(BASELINE_DIR, arch, CKPT_NAME)
    if not os.path.exists(wpath):
        raise FileNotFoundError("%s: 权重缺失 %s" % (arch, wpath))
    model = X.build_model(arch)
    fmt, miss, unexp = X.load_weights(model, wpath, device)
    if len(miss) or len(unexp):
        # 与 P3-1/P3-7 同铁律：键不匹配会静默全零预测，必须硬拦
        raise RuntimeError("%s: 权重键不匹配 (missing=%d, unexpected=%d) -> 拒绝运行。path=%s fmt=%s"
                           % (arch, len(miss), len(unexp), wpath, fmt))
    return model.to(device).eval(), fmt


def build_ood_model(cfg, device):
    wpath = os.path.join(ABL_DIR, cfg, CKPT_NAME)
    if not os.path.exists(wpath):
        raise FileNotFoundError("%s: 权重缺失 %s" % (cfg, wpath))
    model, acfg = get_ablation_model(cfg, in_channels=3, num_classes=1, base_filters=64)
    ckpt = torch.load(wpath, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval(), acfg


# --------------------------------------------------------------------------- #
# 单个 ID 单元
# --------------------------------------------------------------------------- #
def run_id_unit(arch, device, data, fills, conditions, keep_clean=False):
    model, fmt = build_id_model(arch, device)
    mdir = os.path.join(ID_PRED_ROOT, arch, "masks")
    rows = []
    for name_full, img, gt in data:
        stem = os.path.splitext(name_full)[0]
        frozen_path = os.path.join(mdir, stem + ".npy")
        frozen = np.load(frozen_path).astype(bool) if os.path.exists(frozen_path) else None
        if frozen is None:
            raise FileNotFoundError("冻结预测缺失: %s" % frozen_path)

        occ_cache = {}
        for cname, placement, area in conditions:
            if area > 0:
                occ = occluder(gt, stem, area, placement)
            else:
                occ = None
            for fill in fills:
                if occ is None:
                    x = img
                    o = np.zeros(gt.shape, dtype=bool)
                else:
                    o = occ
                    x = make_fill(img, o, fill)
                pred = _forward(model, x, device)
                c = counts(pred, gt, o)
                rec = dict(unit_set="ID", unit=arch, name=stem, fill=fill, cond=cname,
                           area=area, placement=(placement or "none"), **c)
                rec["dice_F"] = C.dice_from_counts(c["inter"], c["p_tot"], c["g_tot"])
                rec["dice_V"] = C.dice_from_counts(c["inter_v"], c["p_v"], c["g_v"])
                rec["dice_F_frozen"] = C.dice_from_counts(
                    int(np.logical_and(frozen, gt).sum()), int(frozen.sum()), int(gt.sum()))
                if keep_clean and cname == "0_clean":
                    rec["pix_mismatch"] = int(np.logical_xor(pred, frozen).sum())
                rows.append(rec)
    del model
    torch.cuda.empty_cache()
    return rows, fmt


# --------------------------------------------------------------------------- #
# 单个 OOD 单元
# --------------------------------------------------------------------------- #
def run_ood_unit(cfg, device, fills, conditions):
    model, acfg = build_ood_model(cfg, device)
    p = os.path.join(ABL_ETIS_ROOT, cfg, "predictions.npy")
    arr = np.load(p, allow_pickle=True)
    rows = []
    try:
        for it in arr:
            name = str(it["name"])
            img = np.asarray(it["image"], dtype=np.float32)      # ← 上游网络的真实输入
            gt = (np.asarray(it["mask"]) > PIXEL_THRESH)
            frozen = np.asarray(it["pred"]).astype(bool)
            frozen_dice = float(it["dice"])

            for cname, placement, area in conditions:
                if area > 0:
                    occ = occluder(gt, name, area, placement)
                else:
                    occ = None
                for fill in fills:
                    if occ is None:
                        x = img
                        o = np.zeros(gt.shape, dtype=bool)
                    else:
                        o = occ
                        x = make_fill(img, o, fill)
                    pred = _forward(model, x, device)
                    c = counts(pred, gt, o)
                    rec = dict(unit_set="OOD", unit=cfg, name=name, fill=fill, cond=cname,
                               area=area, placement=(placement or "none"), **c)
                    rec["dice_F"] = C.dice_from_counts(c["inter"], c["p_tot"], c["g_tot"])
                    rec["dice_V"] = C.dice_from_counts(c["inter_v"], c["p_v"], c["g_v"])
                    rec["dice_F_frozen"] = frozen_dice if cname == "0_clean" else np.nan
                    if cname == "0_clean":
                        rec["pix_mismatch"] = int(np.logical_xor(pred, frozen).sum())
                    rows.append(rec)
    finally:
        del arr
    del model
    torch.cuda.empty_cache()
    return rows, acfg


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summarize(df, fills, conditions, n_id, n_ood):
    out = {"segment": "P3-6b", "generated_at": datetime.now().isoformat(timespec="seconds"),
           "script": "02_code/analysis/p3_6b_occlusion_input.py",
           "spec": "00_docs/P3-6b规格冻结_2026-09-15.md",
           "frozen": {"area_targets": list(AREA_TARGETS), "placements": list(PLACEMENTS),
                      "axis_ratio": AXIS_RATIO, "eps": EPS, "pixel_thresh": PIXEL_THRESH,
                      "fills": list(fills), "primary_fill": PRIMARY_FILL,
                      "conditions_run": [c for c, _, _ in conditions],
                      "n_unit_id": n_id, "n_unit_ood": n_ood,
                      "names_id": list(ID_ARCHS), "names_ood": list(OOD_CFGS)},
           "gates": {}, "per_fill": {}}

    for fill in fills:
        d = df[df["fill"] == fill]
        blk = {"per_condition": {}, "by_fill": fill}
        for cname, placement, area in conditions:
            dc = d[d["cond"] == cname]
            rc = {"cond": cname, "area": area, "placement": (placement or "none"),
                  "by_unit_set": [], "rank_agreement": {}, "unit_means": []}
            for us in ("ID", "OOD"):
                ds = dc[dc["unit_set"] == us]
                rec = dict(unit_set=us, n_samples=int(len(ds)),
                           mean_F=float(ds["dice_F"].mean()),
                           mean_V=float(ds["dice_V"].mean()),
                           mean_F_frozen=float(ds["dice_F_frozen"].mean(skipna=True))
                           if ds["dice_F_frozen"].notna().any() else None,
                           mean_occ_frac=float(ds["occ_px"].sum() / (ds["occ_px"].sum() + 1)) if False else
                           float((ds["occ_px"] / (352.0 * 352.0)).mean()),
                           mean_t=float(ds["t"].mean()), mean_f=float(ds["f_occ"].mean()),
                           mean_u=float(ds["u_occ"].mean()),
                           # ★ 病灶相对遮挡覆盖率 = |G∩M_occ| / |G| = (g_tot - g_v) / g_tot
                           #   两个单元集用**同一图像面积比**的椭圆，但病灶大小不同
                           #   ⇒ 该量是**必须并报的诊断**（否则"域间翻转"与"OOD 病灶被抹得更彻底"混淆）
                           mean_gt_coverage=(float(((ds["g_tot"] - ds["g_v"]) / ds["g_tot"].replace(0, np.nan)).mean())
                                             if ds["g_tot"].sum() > 0 else None),
                           mean_gt_area=float(ds["g_tot"].mean()),
                           mean_pos_occ_frac=float((ds["pos_occ"] / ds["occ_px"].replace(0, np.nan)).mean())
                           if ds["occ_px"].sum() > 0 else None)
                rec["delta_V_minus_F"] = rec["mean_V"] - rec["mean_F"]
                rc["by_unit_set"].append(rec)

                gm = ds.groupby("unit")[["dice_F", "dice_V"]].mean()
                rho, p = C._spearman(gm["dice_F"].values, gm["dice_V"].values)
                tau = C._kendall(gm["dice_F"].values, gm["dice_V"].values)
                n = len(gm)
                rev = 0
                f_, v_ = gm["dice_F"].values, gm["dice_V"].values
                for i in range(n):
                    for j in range(i + 1, n):
                        if (f_[i] - f_[j]) * (v_[i] - v_[j]) < 0:
                            rev += 1
                rc["rank_agreement"][us] = dict(
                    n_units=n, spearman_rho=rho, spearman_p=p, kendall_tau=tau,
                    n_pairs=n * (n - 1) // 2, n_reversed_pairs=rev,
                    pct_reversed_pairs=float(rev / (n * (n - 1) / 2) * 100) if n > 1 else None)
                for u, r in gm.iterrows():
                    rc["unit_means"].append(dict(unit_set=us, unit=u,
                                                 mean_F=float(r["dice_F"]), mean_V=float(r["dice_V"])))
            # 域间差距
            idr = [r for r in rc["by_unit_set"] if r["unit_set"] == "ID"][0]
            oor = [r for r in rc["by_unit_set"] if r["unit_set"] == "OOD"][0]
            gap_F = idr["mean_F"] - oor["mean_F"]
            gap_V = idr["mean_V"] - oor["mean_V"]
            rc["interdomain_gap"] = dict(
                gap_F=gap_F, gap_V=gap_V, delta_gap=gap_V - gap_F,
                compression=float(1.0 - gap_V / gap_F) if abs(gap_F) > 1e-12 else None,
                sign_flip=bool(np.sign(gap_V) != np.sign(gap_F)))
            blk["per_condition"][cname] = rc
        out["per_fill"][fill] = blk
    return out


def gate_block(df, fills, conditions, full_sets=True):
    """G1/G2/G3/G4/G6。任一不过 => verdict=STOP。

    `full_sets=False`（冒烟/子集）⇒ **G4 记为 SKIP**：它的参考值是**全集均值**，
    子集算出来的数天然不等，不是装置问题。其余门禁照常判定。
    """
    g = {}
    # ---- G1 恒等档：逐样本 max|Δdice| + 像素不一致
    cl = df[df["cond"] == "0_clean"]
    d = np.abs(cl["dice_F"].values - cl["dice_F_frozen"].values)
    d = d[np.isfinite(d)]
    pix = cl["pix_mismatch"].dropna()
    worst = float(d.max()) if d.size else float("nan")
    g["G1_identity_rerun"] = dict(
        n=int(d.size), max_abs_dice_diff=worst, tol=GATE_IDENT_TOL,
        pass_=bool(worst <= GATE_IDENT_TOL),
        pix_mismatch_total=int(pix.sum()) if len(pix) else None,
        pix_mismatch_max_per_sample=int(pix.max()) if len(pix) else None,
        by_unit_set={us: dict(
            n=int(len(cl[cl.unit_set == us])),
            max_abs_dice_diff=float(np.nanmax(np.abs(
                cl[cl.unit_set == us]["dice_F"].values - cl[cl.unit_set == us]["dice_F_frozen"].values))))
            for us in ("ID", "OOD")})

    # ---- G4 恒等档集合均值 vs P3-6 参考（仅全单元集适用）
    g4 = {}
    for us in ("ID", "OOD"):
        ds = cl[cl.unit_set == us]
        v = float(ds["dice_F_frozen"].mean())
        ref = P3_6_REF_EF[us]
        ok = bool(abs(v - ref) <= GATE_EF_TOL)
        g4[us] = dict(mean_F_frozen_recomputed=v, p3_6_reference=ref,
                      abs_diff=abs(v - ref), tol=GATE_EF_TOL,
                      pass_=ok, skipped=(not full_sets))
    g["G4_clean_matches_p3_6"] = g4

    # ---- G2 遮挡确实生效
    g2 = {}
    for cname, placement, area in conditions:
        if area <= 0:
            continue
        ds = df[df["cond"] == cname]
        if not len(ds):
            g2[cname] = dict(area_target=area, occ_frac_mean=None, applied=True,
                             note="无该条件数据（子集运行），不参与判定")
            continue
        occ_frac = float((ds["occ_px"] / (352.0 * 352.0)).mean())
        g2[cname] = dict(area_target=area, occ_frac_mean=occ_frac,
                         applied=bool(ds["occ_px"].min() > 0))
    g["G2_occlusion_applied"] = g2

    # ---- G3 area=0 => V ≡ F
    d3 = np.abs(cl["dice_V"].values - cl["dice_F"].values)
    d3 = d3[np.isfinite(d3)]
    g["G3_zero_tier_V_eq_F"] = dict(n=int(d3.size), max_abs_diff=float(d3.max()) if d3.size else None,
                                    pass_=bool((d3.max() if d3.size else 1.0) == 0.0))

    # ---- G6 计数三分自洽
    #   t + f_occ == |P'∩M_occ|  （阳性像素在遮挡区内的 TP/FP 二分）
    #   遮挡区 GT 侧亦须自洽：t + u_occ == |G∩M_occ|
    m1 = (df["t"] + df["f_occ"]) == df["pos_occ"]
    # |G∩M_occ| 未单独存列，用 g_tot 与可见区 GT 反推：
    #   |G∩M_occ| = g_tot − g_v
    g_occ = df["g_tot"] - df["g_v"]
    m2 = (df["t"] + df["u_occ"]) == g_occ
    g["G6_partition"] = dict(
        n=int(len(df)),
        n_bad_pred_side=int((~m1).sum()),
        n_bad_gt_side=int((~m2).sum()),
        pass_=bool(bool(m1.all()) and bool(m2.all())))
    ok_g4 = all(v["pass_"] for v in g4.values()) or (not full_sets)
    g["verdict"] = ("PASS" if (g["G1_identity_rerun"]["pass_"]
                               and g["G3_zero_tier_V_eq_F"]["pass_"]
                               and ok_g4
                               and g["G6_partition"]["pass_"]
                               and all(v["applied"] for v in g2.values())) else "STOP")
    g["g4_mode"] = ("enforced" if full_sets else "SKIPPED (subset run)")
    return g


def write_md(res, g, conditions):
    L = []
    L.append("# T-P3-6b ｜ 遮挡**进输入**（真鲁棒性版本）")
    L.append("")
    L.append("> 规格冻结：`00_docs/P3-6b规格冻结_2026-09-15.md`（G6）")
    L.append("> 几何与 **P3-6 逐位相同**（同一 `ellipse_mask` / 同一 CRC32 种子串 / 同面积档）")
    L.append("> ⇒ 两次实验之间**只有**一处不同：椭圆挖在**输入**上，还是只从**计分**里剔除。")
    L.append("> ⚠️ `dice_F'` 与 P3-6 的冻结 Dice **不可比谁更高**；只可比 `V'-F'` 与**域间 gap**。")
    L.append("")
    L.append("## 门禁")
    L.append("")
    L.append("| 门 | 内容 | 结果 |")
    L.append("|---|---|---|")
    g1 = g["G1_identity_rerun"]
    L.append("| **G1** | 恒等档重跑复现冻结预测 | %s ｜ max\\|Δdice\\| = **%.3e**（tol 1e-6）｜像素不一致合计 **%s** |"
             % ("✅ PASS" if g1["pass_"] else "❌ FAIL", g1["max_abs_dice_diff"],
                g1["pix_mismatch_total"]))
    g3 = g["G3_zero_tier_V_eq_F"]
    L.append("| G3 | `area=0 ⇒ V ≡ F` | %s ｜ max\\|Δ\\| = %s |"
             % ("✅ PASS" if g3["pass_"] else "❌ FAIL", g3["max_abs_diff"]))
    g4 = g["G4_clean_matches_p3_6"]
    if g.get("g4_mode", "").startswith("SKIPPED"):
        L.append("| G4 | 恒等档集合均值复现 P3-6 | ⏸ **SKIPPED**（子集运行：ID Δ = %.3e ／ OOD Δ = %.3e，参考值为全集均值，不适用）|"
                 % (g4["ID"]["abs_diff"], g4["OOD"]["abs_diff"]))
    else:
        L.append("| **G4** | 恒等档集合均值复现 P3-6 | ID Δ = %.3e ／ OOD Δ = %.3e ｜ %s |"
                 % (g4["ID"]["abs_diff"], g4["OOD"]["abs_diff"],
                    "✅ PASS" if all(v["pass_"] for v in g4.values()) else "❌ FAIL"))
    g6 = g["G6_partition"]
    L.append("| G6 | 遮挡区三分自洽 | %s ｜ 阳性侧违例 %d / %d，GT 侧违例 %d |"
             % ("✅ PASS" if g6["pass_"] else "❌ FAIL",
                g6["n_bad_pred_side"], g6["n"], g6["n_bad_gt_side"]))
    L.append("| G2 | 遮挡确实生效 | %s |" % ("✅ PASS" if all(v["applied"] for v in g["G2_occlusion_applied"].values()) else "❌ FAIL"))
    L.append("")
    L.append("> **总判定：%s**" % g["verdict"])
    L.append("")

    for fill in res["per_fill"]:
        blk = res["per_fill"][fill]
        tag = "（**主**）" if fill == PRIMARY_FILL else "（次）"
        L.append("## 填充 = `%s`%s" % (fill, tag))
        L.append("")
        L.append("| 条件 | 集合 | E[F′] | E[V′] | V′−F′ | **病灶覆盖率** | 遮挡区阳性率 | 遮挡实际占比 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for cname, *_ in conditions:
            rc = blk["per_condition"][cname]
            for r in rc["by_unit_set"]:
                L.append("| `%s` | %s | %.4f | %.4f | **%+.4f** | **%.3f** | %s | %.4f |"
                         % (cname, r["unit_set"], r["mean_F"], r["mean_V"], r["delta_V_minus_F"],
                            (r["mean_gt_coverage"] if r["mean_gt_coverage"] is not None else float("nan")),
                            ("%.4f" % r["mean_pos_occ_frac"]) if r["mean_pos_occ_frac"] is not None else "n/a",
                            r["mean_occ_frac"]))
        L.append("")
        L.append("> ⚠️ **病灶覆盖率** = `|G∩M_occ| / |G|`（被遮挡掉的 GT 占比）。两集合用**同一图像面积比**的椭圆，")
        L.append("> 但病灶大小不同（ID 平均 GT %.0f px ／ OOD %.0f px）⇒ **同一椭圆对两集合的'相对破坏程度不同**。"
                 % (blk["per_condition"]["0_clean"]["by_unit_set"][0]["mean_gt_area"],
                    blk["per_condition"]["0_clean"]["by_unit_set"][1]["mean_gt_area"]))
        L.append("> 这是本段必须并报的**几何不对称**，解释'域间翻转'时不得忽略。")
        L.append("")
        L.append("### 域间差距（ID − OOD）")
        L.append("")
        L.append("| 条件 | gap_F′ | gap_V′ | 压缩比 | 符号翻转 |")
        L.append("|---|---:|---:|---:|:--:|")
        for cname, *_ in conditions:
            if cname == "0_clean":
                continue
            gg = blk["per_condition"][cname]["interdomain_gap"]
            L.append("| `%s` | %+.4f | %+.4f | %s | %s |"
                     % (cname, gg["gap_F"], gg["gap_V"],
                        ("%+.1f%%" % (gg["compression"] * 100)) if gg["compression"] is not None else "n/a",
                        "**yes**" if gg["sign_flip"] else "no"))
        L.append("")
        L.append("### 段内排名一致性 ρ(F′, V′)")
        L.append("")
        L.append("| 条件 | 集合 | n 单元 | Spearman ρ | p | Kendall τ | 反转对 |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for cname, *_ in conditions:
            if cname == "0_clean":
                continue
            ra = blk["per_condition"][cname]["rank_agreement"]
            for us in ("ID", "OOD"):
                r = ra[us]
                L.append("| `%s` | %s | %d | %+.3f | %.4f | %+.3f | %d / %d |"
                         % (cname, us, r["n_units"], r["spearman_rho"], r["spearman_p"],
                            r["kendall_tau"], r["n_reversed_pairs"], r["n_pairs"]))
        L.append("")

    L.append("## ★ 与 P3-6 对读（本段的落脚点）")
    L.append("")
    L.append("| 条件 | P3-6（评测端·gap_V） | **P3-6b（输入·gap_V）** | P3-6 压缩（÷其 gap_F=+0.2820） | **P3-6b 压缩（÷其自身 gap_F′）** |")
    L.append("|---|---:|---:|---:|---:|")
    p36 = {"10_target": 0.1072, "30_target": 0.0349, "50_target": -0.0367,
           "10_random": 0.2665, "30_random": 0.2180, "50_random": 0.1053}
    blk = res["per_fill"].get(PRIMARY_FILL)
    for cname, *_ in conditions:
        if cname == "0_clean":
            continue
        gg = blk["per_condition"][cname]["interdomain_gap"]
        comp = gg["compression"]
        L.append("| `%s` | %+.4f | **%+.4f** | %+.1f%% | **%s** |"
                 % (cname, p36[cname], gg["gap_V"], (1 - p36[cname] / 0.2820) * 100,
                    ("%+.1f%%" % (comp * 100)) if comp is not None else "n/a"))
    L.append("")
    L.append("> P3-6 的 gap_F 恒为 +0.2820（源：`03_results/tables/T_p3_6_gap.md`）。")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-6b: occlusion into the INPUT (true robustness)")
    ap.add_argument("--fills", default=",".join(FILLS_DEFAULT))
    ap.add_argument("--id-archs", nargs="+", default=ID_ARCHS)
    ap.add_argument("--ood-cfgs", nargs="+", default=OOD_CFGS)
    ap.add_argument("--limit", type=int, default=None, help="每单元只用前 N 张（冒烟）")
    ap.add_argument("--cond-limit", type=int, default=None, help="只用前 N 个条件（冒烟）")
    ap.add_argument("--smoke", action="store_true", help="1 架构 + 1 配置 + 2 条件 + 20 张")
    ap.add_argument("--out-dir", default=OUT_DIR_REL)
    ap.add_argument("--out-json", default=OUT_JSON_REL)
    ap.add_argument("--out-md", default=OUT_MD_REL)
    ap.add_argument("--skip-md", action="store_true")
    args = ap.parse_args(argv)

    fills = tuple(f.strip() for f in args.fills.split(",") if f.strip())
    conditions = CONDITIONS
    id_archs, ood_cfgs = list(args.id_archs), list(args.ood_cfgs)
    limit = args.limit
    if args.smoke:
        id_archs = ["UNet"]
        ood_cfgs = ["baseline"]
        conditions = [CONDITIONS[0], CONDITIONS[1], CONDITIONS[2], CONDITIONS[3]]
        limit = limit or 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=%s  cuda=%s" % (device, torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("gpu=%s" % torch.cuda.get_device_name(0))

    out_dir = X.eabs(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    rows = []
    meta = {}
    for arch in id_archs:
        data = S.load_id_test(limit)
        print(">>> ID %-14s n=%d 条件=%d fills=%d" % (arch, len(data), len(conditions), len(fills)))
        r, fmt = run_id_unit(arch, device, data, fills, conditions, keep_clean=True)
        rows.extend(r)
        meta["ID/" + arch] = dict(weight_format=fmt, n=len(data))
        print("    %d 行  用时 %.1f s" % (len(r), time.time() - t0))
    for cfg in ood_cfgs:
        print(">>> OOD %-14s" % cfg)
        r, acfg = run_ood_unit(cfg, device, fills, conditions)
        rows.extend(r)
        meta["OOD/" + cfg] = dict(modules={k: bool(v) for k, v in acfg.items() if k.startswith("use_")},
                                  n=len(r) // (len(conditions) * len(fills)))
        print("    %d 行  用时 %.1f s" % (len(r), time.time() - t0))

    df = pd.DataFrame(rows)
    # 逐样本长表落盘（gzip）
    for fill in fills:
        p = os.path.join(out_dir, "samples_%s.csv.gz" % fill)
        df[df["fill"] == fill].to_csv(p, index=False, compression="gzip")
        print("已写: %s  (%d 行)" % (p, int((df['fill'] == fill).sum())))

    res = summarize(df, fills, conditions, len(id_archs), len(ood_cfgs))
    res["runtime_sec"] = round(time.time() - t0, 1)
    res["throughput"] = dict(n_forward=int(len(df)),
                             sec_per_forward=round((time.time() - t0) / max(len(df), 1), 5))
    res["unit_meta"] = meta
    g = gate_block(df, fills, conditions,
                   full_sets=(len(id_archs) == len(ID_ARCHS) and len(ood_cfgs) == len(OOD_CFGS)))
    res["gates"] = g

    with open(X.eabs(args.out_json), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    print("已写: %s" % X.eabs(args.out_json))

    # 逐单元统计（供独立复核）
    for fill in fills:
        d = df[df["fill"] == fill]
        uj = {}
        for (us, u), sub in d.groupby(["unit_set", "unit"]):
            uj["%s/%s" % (us, u)] = {
                c: dict(mean_F=float(sub[sub.cond == c]["dice_F"].mean()),
                        mean_V=float(sub[sub.cond == c]["dice_V"].mean()),
                        n=int(len(sub[sub.cond == c]))) for c, *_ in conditions}
        with open(os.path.join(out_dir, "units_%s.json" % fill), "w", encoding="utf-8") as fh:
            json.dump(uj, fh, ensure_ascii=False, indent=2)

    if not args.skip_md:
        md = write_md(res, g, conditions)
        with open(X.eabs(args.out_md), "w", encoding="utf-8") as fh:
            fh.write(md)
        print("已写: %s" % X.eabs(args.out_md))

    print("\n" + "=" * 96)
    print("门禁判定：%s" % g["verdict"])
    print("  G1 max|Δdice| = %.3e (tol 1e-6)  像素不一致 = %s"
          % (g["G1_identity_rerun"]["max_abs_dice_diff"], g["G1_identity_rerun"]["pix_mismatch_total"]))
    for us in ("ID", "OOD"):
        r4 = g["G4_clean_matches_p3_6"][us]
        print("  G4 %-4s E[F]_frozen = %.12f vs P3-6 %.12f  Δ=%.2e"
              % (us, r4["mean_F_frozen_recomputed"], r4["p3_6_reference"], r4["abs_diff"]))
    print("  用时 %.1f s ｜ 前向 %d 次" % (res["runtime_sec"], res["throughput"]["n_forward"]))
    return 0 if g["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
