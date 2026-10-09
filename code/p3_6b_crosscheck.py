# -*- coding: utf-8 -*-
"""
文件名: p3_6b_crosscheck.py
功能: 【论文二 P3-6b · G4 独立复核】不 import 主脚本，从**书面规格**重实现并逐项对账

独立点（与主脚本的分歧设计，仿 P3-6 的复核手法）
------------------------------------------------
1. 椭圆几何：主脚本用 `np.mgrid` + 广播做布尔掩膜；**本脚本改用显式行/列索引算半径**，
   另用 `skimage.draw.ellipse_polygon`-类的**逐行弦长**思路交叉（保守起见先做精确重算 + 中心比对）。
2. 度量：主脚本从 `p3_6_fragmentation_caliber.dice_from_counts` 取；**本脚本自写**公式。
3. 均值/差距/翻转：主脚本在 `summarize()` 里算；**本脚本从落盘长表重算**。
4. ρ：主脚本用 `scipy.stats.spearmanr`；**本脚本用"秩 → Pearson"手写**。
5. G4 独立路径：主脚本比对 P3-6 的参考均值；**本脚本另从 `dice_F_frozen` 列直接复核同一常量**。

输入（只读）
    03_results/raw/occlusion_input/samples_<fill>.csv.gz
    03_results/stats/occlusion_input.json
输出
    03_results/stats/p3_6b_crosscheck.json

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_6b_crosscheck.py
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

E_ROOT = "E:/paper2_ablation_reliability"
RAW = E_ROOT + "/03_results/raw/occlusion_input"
MAIN = E_ROOT + "/03_results/stats/occlusion_input.json"
OUT = E_ROOT + "/03_results/stats/p3_6b_crosscheck.json"

# ---- 从 00_docs/P3-6b规格冻结_2026-09-15.md 逐字抄写的常数（第二实现） ----
EPS = 1e-8
AXIS_RATIO = 1.6
AREA_TARGETS = (0.10, 0.30, 0.50)
PLACEMENTS = ("target", "random")
P3_6_REF_EF = {"ID": 0.8411039998288351, "OOD": 0.5591206907338347}
H0, W0 = 352, 352

CHECKS = []


def rec(name, ok, detail=None, tol=None):
    CHECKS.append(dict(check=name, ok=bool(ok), tol=tol, detail=detail))
    return bool(ok)


# --------------------------------------------------------------------------- #
# 独立几何实现
# --------------------------------------------------------------------------- #
def my_ellipse_center(gt, placement, name, area):
    """与主脚本同语义：target=GT 质心；random=CRC32 播种均匀随机。"""
    H, W = gt.shape
    if placement == "target":
        idx = np.argwhere(gt)
        if idx.size:
            return float(idx[:, 0].mean()), float(idx[:, 1].mean())
        return H / 2.0, W / 2.0
    seed = zlib.crc32(("{}|caliberswap|{:.2f}".format(name, area)).encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return float(rng.uniform(H * 0.15, H * 0.85)), float(rng.uniform(W * 0.15, W * 0.85))


def my_ellipse_mask(shape, cy, cx, area_frac, axis_ratio=AXIS_RATIO):
    """独立实现：显式行列索引 + 半轴求解 + 中心 clamp（不依赖 mgrid 广播）。"""
    H, W = shape
    if area_frac <= 0:
        return np.zeros((H, W), dtype=bool)
    target_area = area_frac * H * W
    b = float(np.sqrt(target_area / (np.pi * axis_ratio)))     # 短半轴（竖直）
    c = float(b * axis_ratio)                                  # 长半轴（水平）
    if H - 1 - b > b:
        cy = min(max(cy, b), H - 1 - b)
    else:
        cy = (H - 1) / 2.0
    if W - 1 - c > c:
        cx = min(max(cx, c), W - 1 - c)
    else:
        cx = (W - 1) / 2.0
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    dy = ((rows - cy) / b) ** 2
    dx = ((cols - cx) / c) ** 2
    # 独立的"逐行弦长"写法：对每一行求满足 dx <= 1 - dy 的列区间
    out = np.zeros((H, W), dtype=bool)
    lim = 1.0 - dy
    for i in range(H):
        if lim[i] < 0:
            continue
        half = c * np.sqrt(lim[i])
        lo = int(np.ceil(cx - half))
        hi = int(np.floor(cx + half))
        if hi < 0 or lo > W - 1:
            continue
        out[i, max(lo, 0):min(hi, W - 1) + 1] = True
    return out


def my_dice(inter, np_, ng):
    return (2.0 * inter + EPS) / (np_ + ng + EPS)


def my_spearman(x, y):
    """秩 → Pearson（手写）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
def _load_json_any(path):
    """兼容明文 / gzip 两种 json 落盘。"""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_long():
    d = {}
    for p in sorted(glob.glob(os.path.join(RAW, "samples_*.csv.gz"))):
        fill = os.path.basename(p).replace("samples_", "").replace(".csv.gz", "")
        d[fill] = pd.read_csv(p, compression="gzip")
    return d


def main():
    global RAW, MAIN, OUT
    ap = argparse.ArgumentParser(description="P3-6b G4 独立复核")
    ap.add_argument("--raw-dir", default=RAW, help="长表目录（含 samples_<fill>.csv.gz）")
    ap.add_argument("--main-json", default=MAIN, help="主脚本结果 json")
    ap.add_argument("--out", default=OUT, help="复核结果 json 落点")
    ap.add_argument("--subset", action="store_true",
                    help="子集预飞模式：G4（对照 P3-6 全集参考均值）降级为信息项，不计入失败")
    args = ap.parse_args()
    RAW, MAIN, OUT = args.raw_dir, args.main_json, args.out

    main_j = _load_json_any(MAIN)
    long_tabs = load_long()
    print("读入 fill=%s   raw-dir=%s" % (list(long_tabs.keys()), RAW))

    # ---------- 1) dice 公式独立重算 ----------
    worstF = worstV = 0.0
    nF = nV = 0
    for fill, df in long_tabs.items():
        dF = np.abs(my_dice(df["inter"], df["p_tot"], df["g_tot"]).values - df["dice_F"].values)
        dV = np.abs(my_dice(df["inter_v"], df["p_v"], df["g_v"]).values - df["dice_V"].values)
        worstF = max(worstF, float(np.nanmax(dF)))
        worstV = max(worstV, float(np.nanmax(dV)))
        nF += len(df)
        nV += len(df)
    rec("dice_F_recompute", worstF <= 1e-12, dict(n=nF, max_abs_diff=worstF), 1e-12)
    rec("dice_V_recompute", worstV <= 1e-12, dict(n=nV, max_abs_diff=worstV), 1e-12)

    # ---------- 2) 遮挡区三分自洽（GT 侧独立路径） ----------
    bad1 = bad2 = 0
    for fill, df in long_tabs.items():
        bad1 += int(((df["t"] + df["f_occ"]) != df["pos_occ"]).sum())
        bad2 += int(((df["t"] + df["u_occ"]) != (df["g_tot"] - df["g_v"])).sum())
    rec("partition_pred_side", bad1 == 0, dict(n_bad=bad1))
    rec("partition_gt_side", bad2 == 0, dict(n_bad=bad2))

    # ---------- 3) G3：area=0 ⇒ V ≡ F ----------
    d3 = 0.0
    for fill, df in long_tabs.items():
        c = df[df["cond"] == "0_clean"]
        d3 = max(d3, float(np.abs(c["dice_V"].values - c["dice_F"].values).max()))
    rec("zero_tier_V_eq_F", d3 == 0.0, dict(max_abs_diff=d3))

    # ---------- 4) ★ G4 独立路径：clean 的冻结 Dice 直接复现 P3-6 常量 ----------
    g4 = {}
    for fill, df in long_tabs.items():
        c = df[df["cond"] == "0_clean"]
        for us, ref in P3_6_REF_EF.items():
            v = float(c[c["unit_set"] == us]["dice_F_frozen"].mean())
            g4.setdefault(us, []).append((fill, v, abs(v - ref)))
    g4_ok = True
    for us, lst in g4.items():
        for fill, v, ad in lst:
            if ad > 1e-6:
                g4_ok = False
    g4_detail = {us: [dict(fill=f, mean=v, abs_diff=ad) for f, v, ad in lst] for us, lst in g4.items()}
    if args.subset:
        g4_detail = {"mode": "SUBSET (参考为全集均值，本项仅信息)", "rows": g4_detail}
        rec("G4_frozen_dice_reproduces_p3_6", True, g4_detail, None)
    else:
        rec("G4_frozen_dice_reproduces_p3_6", g4_ok, g4_detail, 1e-6)

    # ---------- 5) 集合均值 / 差距 / 翻转 重算 ----------
    worst_mean = 0.0
    gap_rows = []
    for fill, df in long_tabs.items():
        for cname, rc in main_j["per_fill"][fill]["per_condition"].items():
            dc = df[df["cond"] == cname]
            for r in rc["by_unit_set"]:
                us = r["unit_set"]
                ds = dc[dc["unit_set"] == us]
                mF = float(ds["dice_F"].mean())
                mV = float(ds["dice_V"].mean())
                worst_mean = max(worst_mean, abs(mF - r["mean_F"]), abs(mV - r["mean_V"]))
            idr = [x for x in rc["by_unit_set"] if x["unit_set"] == "ID"][0]
            oor = [x for x in rc["by_unit_set"] if x["unit_set"] == "OOD"][0]
            gF = idr["mean_F"] - oor["mean_F"]
            gV = idr["mean_V"] - oor["mean_V"]
            gg = rc["interdomain_gap"]
            ok = (abs(gF - gg["gap_F"]) <= 1e-12 and abs(gV - gg["gap_V"]) <= 1e-12
                  and (np.sign(gV) != np.sign(gF)) == gg["sign_flip"])
            gap_rows.append(dict(fill=fill, cond=cname, gap_F=gF, gap_V=gV,
                                 sign_flip=bool(np.sign(gV) != np.sign(gF)), ok=bool(ok)))
    rec("set_means_recompute", worst_mean <= 1e-12, dict(max_abs_diff=worst_mean), 1e-12)
    rec("interdomain_gap_recompute", all(r["ok"] for r in gap_rows),
        dict(n=len(gap_rows), n_bad=sum(1 for r in gap_rows if not r["ok"])))

    # ---------- 6) ρ 独立重算 ----------
    worst_rho = 0.0
    nrho = 0
    for fill, df in long_tabs.items():
        for cname, rc in main_j["per_fill"][fill]["per_condition"].items():
            dc = df[df["cond"] == cname]
            for us in ("ID", "OOD"):
                gm = dc[dc["unit_set"] == us].groupby("unit")[["dice_F", "dice_V"]].mean()
                if len(gm) < 3:
                    continue
                r = my_spearman(gm["dice_F"].values, gm["dice_V"].values)
                ref = rc["rank_agreement"][us]["spearman_rho"]
                if np.isfinite(r) and np.isfinite(ref):
                    worst_rho = max(worst_rho, abs(r - ref))
                    nrho += 1
    rec("rank_rho_recompute", worst_rho <= 1e-9, dict(n=nrho, max_abs_diff=worst_rho), 1e-9)

    # ---------- 7) ★ 几何独立重算（覆盖全部单元 × 全部 ≤50% 条件） ----------
    geo_worst = 0
    geo_n = 0
    cyc_bad = 0
    for fill, df in long_tabs.items():
        sub = df[df["area"] > 0][["unit_set", "unit", "name", "cond", "occ_px", "area", "placement"]]
        uniq = sub.drop_duplicates(subset=["unit_set", "unit", "name", "cond"])
        for _, r in uniq.iterrows():
            gt = None
            gt = load_gt(r["unit_set"], r["unit"], r["name"])
            if gt is None:
                continue
            cy, cx = my_ellipse_center(gt, r["placement"], r["name"], float(r["area"]))
            occ = my_ellipse_mask(gt.shape, cy, cx, float(r["area"]))
            got = int(occ.sum())
            exp = int(r["occ_px"])
            geo_worst = max(geo_worst, abs(got - exp))
            geo_n += 1
    rec("geometry_occ_px_recompute", geo_worst == 0, dict(n=geo_n, max_abs_diff=geo_worst), 0)

    # ---------- 汇总 ----------
    n_fail = sum(1 for c in CHECKS if not c["ok"])
    verdict = "MATCH" if n_fail == 0 else "MISMATCH"
    out = dict(segment="P3-6b-crosscheck",
               generated_at=datetime.now().isoformat(timespec="seconds"),
               script="02_code/analysis/p3_6b_crosscheck.py",
               independent_of="02_code/analysis/p3_6b_occlusion_input.py（不 import）",
               raw_dir=RAW, main_json=MAIN, subset_mode=bool(args.subset),
               n_checks=len(CHECKS), n_fail=n_fail, verdict=verdict,
               checks=CHECKS, gap_rows=gap_rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    for c in CHECKS:
        print("  %-34s %s  %s" % (c["check"], "OK  " if c["ok"] else "FAIL",
                                  "" if c["detail"] is None else str(c["detail"])[:110]))
    print("-" * 92)
    print("  VERDICT = %s   (%d 项检查，%d 失败)" % (verdict, len(CHECKS), n_fail))
    print("  已写: %s" % OUT)
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------- #
# GT 载入（与主脚本同源但独立读取）
# --------------------------------------------------------------------------- #
_GT_CACHE = {}


def load_gt(unit_set, unit, name):
    key = (unit_set, unit)
    if key not in _GT_CACHE:
        _GT_CACHE[key] = None
        if unit_set == "ID":
            _GT_CACHE[key] = "ID"
        else:
            _GT_CACHE[key] = "OOD"
        # 实际 GT 从两处读
    if _GT_CACHE[key] == "ID":
        p = "D:/medical_segmentation/processed_data/test/masks/%s.npy" % name
        if not os.path.exists(p):
            return None
        return (np.load(p).astype(np.float32) > 0.5)
    # OOD：从 predictions.npy 的 mask 取
    cache_key = ("ood", unit)
    if cache_key not in _GT_CACHE:
        arr = np.load("D:/medical_segmentation/results_ablation_etis/%s/predictions.npy" % unit,
                      allow_pickle=True)
        _GT_CACHE[cache_key] = {str(it["name"]): (np.asarray(it["mask"]) > 0.5) for it in arr}
        del arr
    return _GT_CACHE[cache_key].get(name)


if __name__ == "__main__":
    sys.exit(main())
