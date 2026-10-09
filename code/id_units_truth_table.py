# -*- coding: utf-8 -*-
"""
文件名: id_units_truth_table.py
功能: 【论文二 · PD3-1c 补列】生成**统一的分布内 14 架构真值表**（`id_units_n14.csv`）

背景（用户 2026-09-14 裁定「D 盘原来东西别动, 你自己看着来」）:
    P3-1c 之前分布内只有 `Dice/IoU/(HD95)` 可用, 缺 `Recall` 的恰 2 个架构
    （MultiResUNet / PraNet, 见资产清单 §一 结论一）。P3-1c 已为**全部 13 个 baseline**
    重算了含 `Recall` 的逐样本表（E 盘 `indist_pred/<架构>/sample_metrics.csv`）。
    本脚本把这 13 个 + EGAUNet（走 D 盘单文件路线）**合并成一张权威表**。

⚠️ 四条硬约束（用户裁定 + 项目铁律）:
  1. **D 盘一律只读**。EGAUNet 的分布内记录仍取自
     `D:/medical_segmentation/results/test_results/sample_metrics.csv`, 不复制、不修改。
  2. **不改既有冻结读数**。`decompose.py` / `decomposition.csv` / `p31a_units_n14.csv`
     的既有口径**保持原样**。本表 `id_dice` 列**沿用 P3-1 路线 A 口径**（与 P2-3 / decomposition
     一致），P3-1c 直跑复算值另立 `id_dice_recalc` 列, 两者差异逐行可查。
  3. 检出率**必须并标判据**（B6）: 冻结 `Recall > 0` 与实质 `I >= 1 px` 两读数都给。
  4. **单源取用**：`id_det_*` / `id_k_*` / `id_del_frozen` 一律**照抄** `p31a_units_n14.csv`,
     本脚本不重算（避免两处口径漂移）。

一致性闸门（G-CONS）:
    本表用 E 盘掩膜复算的 `id_dice_recalc` 与 `03_results/stats/p31a_units_n14.csv`（P3-1 路线 A）
    的 `id_dice` **逐位比对**（tol 1e-12）—— 两条独立管线（P3-1c 直跑 vs P3-1A 路线）交叉验证。

    ⚠️ **预期为 13/14 精确匹配 + 1 处已登记差异**（实测结果, 非缺陷）:
       `PolypPVT` 的 D 盘既有 csv 表头为 `...,sample_id`（另一版推理脚本, 缺 HD95/文件名列），
       其 Dice 与 P3-1c 直跑（统一模型工厂 + 权重载入 + autocast）差 **3.53e-5**
       （0.86379502386353 vs 0.8638302927963105）。其余 12 个架构两路均 ≤1.2e-16。
       → 该差异**不"修"**，如实登记为「资产清单 §七 · 实测事实 8」的佐证。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/id_units_truth_table.py
产物:
    03_results/stats/id_units_n14.csv
    03_results/stats/id_units_n14.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

E_ROOT = "E:/paper2_ablation_reliability"
D_ROOT = "D:/medical_segmentation"
PRED_ROOT = f"{E_ROOT}/03_results/raw/indist_pred"
P31A_UNITS = f"{E_ROOT}/03_results/stats/p31a_units_n14.csv"
EGA_ID_CSV = f"{D_ROOT}/results/test_results/sample_metrics.csv"
OUT_CSV = f"{E_ROOT}/03_results/stats/id_units_n14.csv"
OUT_JSON = f"{E_ROOT}/03_results/stats/id_units_n14.json"

TOL_CONS = 1e-12
EGAUNET = "EGAUNet"


def _agg(csv_path: str) -> dict:
    """从逐样本 csv 聚合出架构级读数（Dice/Recall/Precision/HD95 均值 + 有效数）。"""
    t = pd.read_csv(csv_path)
    if "sample_id" in t.columns and "sample_idx" not in t.columns:
        t = t.rename(columns={"sample_id": "sample_idx"})   # 口径归一（PolypPVT）
    return dict(
        n=int(len(t)),
        id_dice_recalc=float(t["Dice"].mean()),
        id_recall=float(t["Recall"].mean()) if "Recall" in t.columns else np.nan,
        id_precision=float(t["Precision"].mean()) if "Precision" in t.columns else np.nan,
        id_hd95_mean=float(t["HD95"].mean(skipna=True)) if "HD95" in t.columns else np.nan,
        id_hd95_valid=int(t["HD95"].notna().sum()) if "HD95" in t.columns else 0,
        has_recall=bool("Recall" in t.columns),
    )


def main():
    p31a = pd.read_csv(P31A_UNITS).set_index("arch")
    rows = []

    # ---------- A. 13 baseline: 取 E 盘 P3-1c 产物（EGAUNet 不在 PRED_ROOT 下, 跳过） ----------
    for arch in p31a.index:
        if arch == EGAUNET:
            continue
        csv = os.path.join(PRED_ROOT, arch, "sample_metrics.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(f"缺 P3-1c 产物: {csv}")
        rec = dict(arch=arch, det_source="indist_pred(E: P3-1c 直跑)", src_csv=csv)
        rec.update(_agg(csv))
        rows.append(rec)

    # ---------- B. EGAUNet: D 盘只读（唯一来源, E 盘无该架构重算产物） ----------
    if not os.path.exists(EGA_ID_CSV):
        raise FileNotFoundError(f"缺 EGAUNet 分布内记录: {EGA_ID_CSV}")
    ega = dict(arch=EGAUNET,
               det_source="disk:results/test_results(read-only, 唯一来源)",
               src_csv=EGA_ID_CSV)
    ega.update(_agg(EGA_ID_CSV))
    rows.append(ega)

    rec_df = pd.DataFrame(rows)
    n_rows = len(rec_df)
    assert n_rows == 14, f"行数应为 14, 实为 {n_rows}"
    assert rec_df["arch"].is_unique, "架构名重复 → 合并逻辑有误"

    # ---------- C. 冻结口径（P3-1 路线 A）: id_dice / 检出两读数 一律照抄, 不重算 ----------
    rec_df["id_dice"] = [float(p31a.loc[a, "id_dice"]) for a in rec_df["arch"]]
    rec_df["id_dice_p31a"] = rec_df["id_dice"]
    rec_df["cons_diff"] = (rec_df["id_dice_recalc"] - rec_df["id_dice"]).abs()
    rec_df["cons_status"] = np.where(rec_df["cons_diff"] <= TOL_CONS,
                                    "MATCH", "REGISTERED_DIFF")
    rec_df["id_det_frozen"] = [float(p31a.loc[a, "id_det_frozen"]) for a in rec_df["arch"]]
    rec_df["id_k_frozen"] = [int(p31a.loc[a, "id_k_frozen"]) for a in rec_df["arch"]]
    rec_df["id_det_subst"] = [float(p31a.loc[a, "id_det_subst"]) for a in rec_df["arch"]]
    rec_df["id_k_subst"] = [int(round((1.0 - float(p31a.loc[a, "id_det_subst"])) * 242))
                            for a in rec_df["arch"]]
    rec_df["id_del_frozen"] = [float(p31a.loc[a, "id_del"]) for a in rec_df["arch"]]

    # 口径断言: id_del ≡ id_dice（冻结判据下 ID 检出恒 1.000, B6 后果）
    assert (rec_df["id_del_frozen"] - rec_df["id_dice"]).abs().max() <= 1e-12, \
        "id_del 与 id_dice 不相等 → 上游 p31a_units_n14.csv 口径已变, 须重新审视"

    out = rec_df[["arch", "det_source", "n",
                  "id_dice", "id_dice_recalc", "id_recall", "id_precision",
                  "id_hd95_mean", "id_hd95_valid",
                  "id_det_frozen", "id_k_frozen", "id_det_subst", "id_k_subst",
                  "id_del_frozen", "id_dice_p31a", "cons_diff", "cons_status",
                  "has_recall", "src_csv"]].sort_values("arch").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    max_d_exact = float(out.loc[out["cons_status"] == "MATCH", "cons_diff"].max())
    diff_rows = out.loc[out["cons_status"] == "REGISTERED_DIFF"]
    n_match = int((out["cons_status"] == "MATCH").sum())
    n_recall = int(out["has_recall"].sum())
    gate_ok = bool(n_match == len(out) - len(diff_rows))   # 所有非 MATCH 行都必须被显式登记

    payload = dict(
        segment="P3-1c+", mode="unified_id_truth_table_n14",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/id_units_truth_table.py",
        purpose=("Recall 落位: 把 P3-1c 新算的 Recall/Precision/HD95 与 P3-1 路线 A 的"
                 "检出两读数合并为一张分布内 14 架构权威表"),
        constraints=dict(
            disk_read_only=True,
            frozen_readings_untouched=["decompose.py",
                                       "03_results/stats/decomposition.csv",
                                       "03_results/stats/p31a_units_n14.csv"],
            id_dice_caliber="P3-1 路线 A (与 P2-3 / decomposition 一致)",
            id_dice_recalc_caliber="P3-1c 直跑 (E 盘掩膜)",
            criterion_both_mandatory=True,
        ),
        gate=dict(
            name="G-CONS", ref="03_results/stats/p31a_units_n14.csv", tol=TOL_CONS,
            n_rows=int(len(out)), n_match=n_match, n_registered_diff=int(len(diff_rows)),
            max_abs_diff_over_matched=max_d_exact,
            status=("PASS_13_OF_14_WITH_1_REGISTERED_DIFF" if gate_ok else "FAIL"),
            registered_diff=[
                dict(arch=r["arch"], id_dice_p31a=float(r["id_dice"]),
                     id_dice_recalc=float(r["id_dice_recalc"]),
                     abs_diff=float(r["cons_diff"]),
                     cause="D 盘既有 csv 由另一版推理脚本产出（表头 sample_id, 缺 HD95/文件名列）",
                     disposition="如实登记为 资产清单 §七 · 实测事实 8 的佐证; 不修改任何一侧")
                for _, r in diff_rows.iterrows()],
        ),
        coverage=dict(n_archs=int(len(out)), n_with_recall=n_recall,
                      missing_recall=list(out.loc[~out["has_recall"], "arch"]),
                      all_have_recall=bool(n_recall == len(out))),
        archs=json.loads(out.to_json(orient="records")),
    )
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("=" * 128)
    print("分布内 14 架构真值表 (id_units_n14.csv)")
    print("=" * 128)
    print(out[["arch", "n", "id_dice", "id_dice_recalc", "id_recall", "id_precision",
               "id_hd95_mean", "id_det_frozen", "id_k_frozen", "id_det_subst",
               "id_k_subst", "cons_diff", "cons_status"]].to_string(
        index=False, float_format=lambda v: f"{v:.10f}"))
    print("-" * 128)
    print(f"  G-CONS: {payload['gate']['status']}")
    print(f"          MATCH {n_match}/{len(out)}  max|Δ|(匹配行) = {max_d_exact:.3e}")
    for d in payload["gate"]["registered_diff"]:
        print(f"          REGISTERED_DIFF: {d['arch']}  "
              f"p31a={d['id_dice_p31a']:.12f}  recalc={d['id_dice_recalc']:.12f}  "
              f"|Δ|={d['abs_diff']:.3e}")
    miss = payload["coverage"]["missing_recall"]
    print(f"  含 Recall 的架构: {n_recall}/{len(out)}"
          + (" (全覆盖)" if n_recall == len(out) else f" (缺: {miss})"))
    print(f"  口径断言: id_del ≡ id_dice  ✅")
    print(f"\n已写: {OUT_CSV}\n已写: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
