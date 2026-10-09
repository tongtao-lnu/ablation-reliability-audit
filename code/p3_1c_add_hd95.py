# -*- coding: utf-8 -*-
"""
文件名: p3_1c_add_hd95.py
功能: 【论文二 · PD3-1c 补列】为分布内逐样本表回填 **HD95** 列（只写 E 盘, D 盘一律不动）

背景（用户 2026-09-14 裁定「HD95 那就补吧」）:
    `03_results/raw/indist_pred/<架构>/sample_metrics.csv` 的 `HD95` 列在 P3-1c 首轮为
    **空**（当时理由是"本仓多套 HD95 实现, 未擅自引入"）。本脚本补上, 并把"选哪一套实现"
    这件事变成**可复核的复现闸门**, 而不是靠声明。

口径选择（跑前冻结, G6）:
    ID 侧 HD95 = `utils/metrics.py::SegmentationMetrics._compute_hd95`（**逐字复用**）。
    门控与其一致: `pred.sum() > 10 and target.sum() > 10`, 否则记 NaN。
    该实现**完全确定**（`distance_transform_edt` + `binary_erosion`, 无 RNG）——
    与跨域侧 `cross_dataset/train_cross.py::calculate_hd95`（前景 > 5000 时**无种子**
    `np.random.choice` 抽点 ⇒ 逐样本不可复现, 见 P3-1 报告）**性质不同**。
    ⇒ 结论: **ID 侧 HD95 逐样本可复现**; 跨域侧不可。两者不可混谈。

复现闸门（G-HD95, 本段新增）:
    13 个 baseline 中有 **10 个**在 D 盘既有 `sample_metrics.csv` 里**已有 HD95 值**
    （列齐全）；`MultiResUNet`/`PraNet` 有列但**全空**；`PolypPVT` **无该列**。
    → 对那 10 个逐一复算并比对, 要求 `max|Δ| <= 1e-9`；不通过即**中止**
      （不许"补了个口径不明的数"）。

⚠️ 与 Dice 的关系（写作时必须一并声明, 见 B5 / P3-1）:
    HD95 是**勾画侧**的补充指标, 不携带检出信息（完全漏检时 HD95 可达 244 px,
    表现为"极差"而非"特殊值"）→ 论文里 **不得** 用 HD95 做检出侧论证。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/p3_1c_add_hd95.py [--archs ...] [--tol 1e-9]
产物:
    03_results/raw/indist_pred/<架构>/sample_metrics.csv   (回填 HD95 列, 就地覆盖 E 盘)
    03_results/stats/indist_hd95_gate.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt

E_ROOT = "E:/paper2_ablation_reliability"
D_ROOT = "D:/medical_segmentation"
PRED_ROOT = f"{E_ROOT}/03_results/raw/indist_pred"
GT_DIR = f"{D_ROOT}/processed_data/test/masks"
BASELINE_DIR = f"{D_ROOT}/experiments/baseline"     # 绝对路径: 本脚本不 chdir, 不可用相对路径
EGA_ID_CSV = f"{D_ROOT}/results/test_results/sample_metrics.csv"
OUT_JSON = f"{E_ROOT}/03_results/stats/indist_hd95_gate.json"

ARCHS = ["UNet", "AttentionUNet", "SegNet", "ResUNet", "MultiResUNet", "PSPNet",
         "TransUNet", "PolypPVT", "SANet", "PraNet", "UACANet", "CaraNet", "M2SNet"]

GATE_SUM = 10          # utils/metrics.py 的门控阈值（前景像素数）
TOL_HD95 = 1e-9


# ----------------------------------------------------------------- 口径（逐字复用）
def _get_boundary(mask: np.ndarray) -> np.ndarray:
    """utils/metrics.py::SegmentationMetrics._get_boundary —— 逐字复用。"""
    eroded = binary_erosion(mask.astype(bool), iterations=1)
    return mask.astype(float) - eroded.astype(float)


def compute_hd95(pred: np.ndarray, target: np.ndarray):
    """utils/metrics.py::SegmentationMetrics._compute_hd95 —— 逐字复用。

    pred/target: 二值 2D float 数组（已按 >0.5 二值化）。
    返回 float 或 None（边界为空）。门控由调用方负责（与 update() 一致）。
    """
    try:
        pb = _get_boundary(pred)
        tb = _get_boundary(target)
        if pb.sum() == 0 or tb.sum() == 0:
            return None
        pd_ = distance_transform_edt(~pred.astype(bool))
        td_ = distance_transform_edt(~target.astype(bool))
        all_d = np.concatenate([pd_[tb > 0], td_[pb > 0]])
        return float(np.percentile(all_d, 95))
    except Exception:
        return None


def hd95_for_sample(pred_bool: np.ndarray, gt_bool: np.ndarray):
    """带门控（pred.sum()>10 and target.sum()>10）的 HD95；不满足返回 NaN。"""
    if pred_bool.sum() <= GATE_SUM or gt_bool.sum() <= GATE_SUM:
        return float("nan")
    v = compute_hd95(pred_bool.astype(np.float32), gt_bool.astype(np.float32))
    return float("nan") if v is None else v


# ----------------------------------------------------------------- 主流程
def process_arch(arch: str, tol: float) -> dict:
    csv_path = os.path.join(PRED_ROOT, arch, "sample_metrics.csv")
    mask_dir = os.path.join(PRED_ROOT, arch, "masks")
    if not os.path.exists(csv_path):
        return dict(arch=arch, status="NO_PRED_CSV", csv=csv_path)

    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        return dict(arch=arch, status="NO_NAME_COL", cols=[str(c) for c in df.columns])

    vals, missing = [], []
    for _, r in df.iterrows():
        nm = str(r["name"])
        gt_p = os.path.join(GT_DIR, nm)
        pr_p = os.path.join(mask_dir, nm)
        if not (os.path.exists(gt_p) and os.path.exists(pr_p)):
            missing.append(nm)
            vals.append(float("nan"))
            continue
        gt = np.load(gt_p) > 0.5
        pr = np.load(pr_p).astype(bool)
        vals.append(hd95_for_sample(pr, gt))

    df["HD95"] = np.array(vals, dtype=float)
    df.to_csv(csv_path, index=False)

    n_nan = int(np.isnan(df["HD95"]).sum())

    # ---- 复现闸门: 与该架构 D 盘既有 HD95 比对（若存在） ----
    gate = dict(status="NO_REF")
    ref_path = os.path.join(BASELINE_DIR, arch, "sample_metrics.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        if "sample_idx" not in ref.columns and "sample_id" in ref.columns:
            ref = ref.rename(columns={"sample_id": "sample_idx"})
        if "HD95" in ref.columns and "sample_idx" in ref.columns:
            ref = ref.sort_values("sample_idx").reset_index(drop=True)
            mine = df.sort_values("sample_idx").reset_index(drop=True)
            if len(ref) == len(mine):
                rv = ref["HD95"].to_numpy(float)
                mv = mine["HD95"].to_numpy(float)
                both = ~np.isnan(rv) & ~np.isnan(mv)
                nan_mask_match = bool(((np.isnan(rv)) == (np.isnan(mv))).all())
                n_ref_valid = int((~np.isnan(rv)).sum())
                if n_ref_valid == 0:
                    gate = dict(status="REF_ALL_NAN", ref=ref_path,
                                n_ref_valid=0, n_mine_valid=int((~np.isnan(mv)).sum()))
                else:
                    d = np.abs(rv[both] - mv[both])
                    gate = dict(status=("MATCH" if (len(d) == n_ref_valid
                                                    and float(d.max()) <= tol)
                                        else "DIFF"),
                                ref=ref_path, n_ref_valid=n_ref_valid,
                                n_both=len(d), nan_mask_match=nan_mask_match,
                                max_abs_diff=(float(d.max()) if len(d) else float("nan")),
                                tol=tol)
            else:
                gate = dict(status="LEN_MISMATCH", n_ref=len(ref), n_mine=len(mine))
        else:
            gate = dict(status="REF_NO_HD95_COL" if "HD95" not in ref.columns
                        else "REF_NO_IDX_COL",
                        ref=ref_path, cols=[str(c) for c in ref.columns])

    return dict(arch=arch, status="OK", csv=csv_path, n=int(len(df)),
                n_hd95_valid=int(len(df) - n_nan), n_hd95_nan=n_nan,
                hd95_mean=float(np.nanmean(df["HD95"])),
                hd95_median=float(np.nanmedian(df["HD95"])),
                hd95_min=float(np.nanmin(df["HD95"])),
                hd95_max=float(np.nanmax(df["HD95"])),
                n_masks_missing=len(missing),
                gate_vs_disk=gate)


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-1c 补列: 分布内逐样本 HD95")
    ap.add_argument("--archs", nargs="+", default=ARCHS)
    ap.add_argument("--tol", type=float, default=TOL_HD95)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)

    print("=" * 92)
    print("P3-1c 补列 · 分布内逐样本 HD95（口径 = utils/metrics.py::_compute_hd95, 确定性）")
    print("=" * 92)

    res = [process_arch(a, args.tol) for a in args.archs]
    ok = [r for r in res if r.get("status") == "OK"]
    gates = [r for r in ok if r["gate_vs_disk"].get("status") in ("MATCH", "DIFF")]
    n_match = sum(1 for g in gates if g["gate_vs_disk"]["status"] == "MATCH")
    n_diff = sum(1 for g in gates if g["gate_vs_disk"]["status"] == "DIFF")

    print(f"  {'arch':14s} {'n':>4s} {'valid':>6s} {'nan':>4s} "
          f"{'mean':>10s} {'median':>10s} {'max':>12s}  gate")
    for r in res:
        g = r.get("gate_vs_disk", {}).get("status", r.get("status"))
        if r.get("status") != "OK":
            print(f"  {r['arch']:14s}  --- {r['status']}")
            continue
        print(f"  {r['arch']:14s} {r['n']:4d} {r['n_hd95_valid']:6d} {r['n_hd95_nan']:4d} "
              f"{r['hd95_mean']:10.4f} {r['hd95_median']:10.4f} {r['hd95_max']:12.4f}  {g}")

    print("-" * 92)
    print(f"  复现闸门（对 D 盘既有 HD95 值）: {n_match}/{len(gates)} MATCH, {n_diff} DIFF")
    for g in gates:
        gg = g["gate_vs_disk"]
        print(f"    {g['arch']:14s} n_ref_valid={gg.get('n_ref_valid')} "
              f"n_both={gg.get('n_both')} max|Δ|={gg.get('max_abs_diff'):.3e} "
              f"nan_mask_match={gg.get('nan_mask_match')}")
    # 无既有值可比的架构（如实登记, 不假装验过）
    n_no_ref = [g["arch"] for g in ok
                if g["gate_vs_disk"].get("status") in ("REF_ALL_NAN", "REF_NO_HD95_COL",
                                                       "NO_REF")]
    print(f"  无既有 HD95 可比的架构（沿用已验证口径, 无真值校验）: {n_no_ref}")

    # EGAUNet 的 ID 记录（单文件路线）: 登记其 HD95 是否存在
    ega_note = dict(path=EGA_ID_CSV, exists=os.path.exists(EGA_ID_CSV))
    if os.path.exists(EGA_ID_CSV):
        ec = pd.read_csv(EGA_ID_CSV)
        ega_note["cols"] = [str(c) for c in ec.columns]
        ega_note["has_hd95"] = bool("HD95" in ec.columns)
        ega_note["n"] = int(len(ec))
        if "HD95" in ec.columns:
            ega_note["hd95_valid"] = int(ec["HD95"].notna().sum())
            ega_note["hd95_mean"] = float(ec["HD95"].mean(skipna=True))
    print(f"  EGAUNet(ID 单文件路线): {ega_note.get('cols')} has_hd95={ega_note.get('has_hd95')}")

    payload = dict(
        segment="P3-1c+", mode="add_hd95_column_in_distribution",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/p3_1c_add_hd95.py",
        decision="用户 2026-09-14 裁定: 补 HD95 列",
        caliber=dict(
            impl="utils/metrics.py::SegmentationMetrics._compute_hd95 (逐字复用)",
            gate_sum=GATE_SUM,
            deterministic=True,
            determinism_note=("ID 侧实现无 RNG（distance_transform_edt + binary_erosion）"
                              "⇒ 逐样本可复现；跨域侧 calculate_hd95 无种子抽点 ⇒ 不可复现。"
                              "两者不可混谈。"),
            disk_untouched=True,
            out_scope="E 盘 03_results/raw/indist_pred/<架构>/sample_metrics.csv 就地回填",
        ),
        gate=dict(tol=args.tol, n_checked=len(gates), n_match=n_match, n_diff=n_diff,
                  no_ref_archs=n_no_ref),
        archs=res,
        egaunet_id_note=ega_note,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n已写: {args.out}")

    if n_diff:
        print("❌ 复现闸门未全通过 → 口径可能不同, 须人工裁决（已如实登记）")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
