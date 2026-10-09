# -*- coding: utf-8 -*-
"""
p3_2_etis_full_module.py -- P3-2 第 8 格（egm_dpa_msfa）的 ETIS 零样本推理

为什么单独写（不能直接用上游脚本）
--------------------------------
上游 `ablation_zeroshot_etis.py` 的 `main()` 末尾会把
`results_ablation_etis/ablation_etis_summary.json`
**整体重写**为"本次传入的配置列表"。若我们只传第 8 格，就会把已冻结的 7 格汇总
**覆盖成 1 条** —— 而该文件正是 P3-6 §3.2 **独立聚合门禁**的比对基准
（7/7 通过，最差 1.421e-14），属于**不可破坏的验收物**。

因此本脚本：
  * **复用**上游 `run_one()`（同一份 `CrossDataset` / `calculate_metrics` /
    `calculate_hd95` / 阈值 `sigmoid>0.5` / autocast）⇒ **协议与 7 格一致**；
  * **不触碰** `ablation_etis_summary.json`（冻结的 7 格基准原样保留）；
  * 第 8 格结果写入 `<OUT_ROOT>/egm_dpa_msfa/`，
    并**另存**一份 8 格汇总 `ablation_etis_summary_full8.json`。

冻结（G6）
---------
  配置   egm_dpa_msfa = EGM(开) + DPA(开) + MSFA(开)，见 `p3_2_train_full_module.py`
  参数   48.4154 M（与训练侧同一冻结值）
  数据   `data_zeroshot/etis`（ETIS-Larib，未见域，零样本），样本数须 = 196

门禁（G4：独立重算）
------------------
  1. **参数量门禁**：模型参数量须 = 48.4154 M（容差 1e-3 M）。
  2. **样本数门禁**：`len(dataset)` 须 = 196。
  3. **dice 独立重算门禁**：从刚落盘的 `predictions.npy` 用 **numpy** 逐样本重算
     `2|P∩G| / (|P|+|G|+eps)`，其均值 ×100 须等于 `results.json` 的 `dice`
     （tol 1e-6；掩膜来自 float32 预测，故不追求 1e-9 级）。
  4. **原有 7 格不被破坏**：运行前后 `ablation_etis_summary.json` 的字节数必须不变。

跑法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_2_etis_egm_dpa_msfa -- \\
        $PY E:/paper2_ablation_reliability/02_code/analysis/p3_2_etis_full_module.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

D_ROOT = "D:/medical_segmentation"
E_ROOT = "E:/paper2_ablation_reliability"

CONFIG_NAME = "egm_dpa_msfa"
EXPECTED_PARAMS_M = 48.4154
EXPECTED_N = 196
PARAMS_TOL_M = 1e-3
DICE_TOL = 1e-6

FULL_CFG = {
    "use_egm": True,
    "use_dpa": True,
    "use_msfa": True,
    "description": "U-Net + EGM + DPA + MSFA (2^3 full)",
}

sys.path.insert(0, D_ROOT)
os.chdir(D_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import models.ablation_models as am  # noqa: E402

am.ABLATION_CONFIGS[CONFIG_NAME] = FULL_CFG

from ablation_zeroshot_etis import run_one, ETIS_DIR, OUT_ROOT  # noqa: E402
from cross_dataset.dataset import CrossDataset  # noqa: E402

SUMMARY = os.path.join(OUT_ROOT, "ablation_etis_summary.json")
SUMMARY_FULL8 = os.path.join(OUT_ROOT, "ablation_etis_summary_full8.json")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"[P3-2/ETIS] 设备 {device} | 配置 {CONFIG_NAME}")

    ckpt = os.path.join("experiments", "ablation", CONFIG_NAME, "best_model.pth")
    if not os.path.exists(ckpt):
        print(f"[P3-2/ETIS] ✗ 权重缺失，先跑训练：{ckpt}")
        return 1
    print(f"[P3-2/ETIS] 权重 {ckpt}  ({os.path.getsize(ckpt) / 1e6:.1f} MB)")

    # ---- 门禁 4 前置：冻结汇总的指纹 -------------------------------------- #
    before_md5 = _md5(SUMMARY) if os.path.exists(SUMMARY) else None
    print(f"[P3-2/ETIS] 冻结汇总 md5(before) = {before_md5}")

    # ---- 门禁 1：参数量 --------------------------------------------------- #
    net, cfg = am.get_ablation_model(CONFIG_NAME, in_channels=3, num_classes=1, base_filters=64)
    p_m = sum(x.numel() for x in net.parameters()) / 1e6
    del net
    assert abs(p_m - EXPECTED_PARAMS_M) < PARAMS_TOL_M, (
        f"参数量 {p_m:.4f} M ≠ 冻结值 {EXPECTED_PARAMS_M} M"
    )
    print(f"[P3-2/ETIS] ✓ 门禁1 参数量 {p_m:.4f} M")

    # ---- 门禁 2：样本数 --------------------------------------------------- #
    ds = CrossDataset(ETIS_DIR, augment=False, strong_augment=False)
    print(f"[P3-2/ETIS] ETIS 样本数 = {len(ds)}")
    assert len(ds) == EXPECTED_N, f"ETIS 样本数 {len(ds)} ≠ 冻结值 {EXPECTED_N}"

    # ---- 推理（复用上游 run_one） ----------------------------------------- #
    res = run_one(CONFIG_NAME, device, ds)
    if res is None:
        print("[P3-2/ETIS] ✗ run_one 返回 None（权重缺失？）")
        return 1

    # ---- 门禁 3：独立重算 dice（不 import 上游度量函数） ------------------- #
    preds = np.load(os.path.join(OUT_ROOT, CONFIG_NAME, "predictions.npy"), allow_pickle=True)
    eps = 1e-8
    dice_re, dice_up = [], []
    for it in preds:
        P = np.asarray(it["pred"]).astype(bool).ravel()
        G = np.asarray(it["mask"]).astype(bool).ravel()
        inter = np.logical_and(P, G).sum()
        dice_re.append((2.0 * inter + eps) / (P.sum() + G.sum() + eps))
        dice_up.append(float(it["dice"]))
    dice_re = float(np.mean(dice_re) * 100)
    dice_up = float(np.mean(dice_up) * 100)
    delta = abs(dice_re - dice_up)
    print(f"[P3-2/ETIS] dice 独立重算 = {dice_re:.6f} | 上游存值 = {dice_up:.6f} | Δ = {delta:.2e}")
    assert delta < DICE_TOL, f"dice 独立重算不一致：Δ={delta:.3e} > {DICE_TOL}"
    assert abs(dice_re - res["dice"]) < DICE_TOL, "重算值与 results.json 不一致"
    print("[P3-2/ETIS] ✓ 门禁3 dice 独立重算一致")

    # ---- 门禁 4 后置：冻结汇总未被改动 ------------------------------------ #
    after_md5 = _md5(SUMMARY) if os.path.exists(SUMMARY) else None
    assert before_md5 == after_md5, "冻结的 ablation_etis_summary.json 被改动了！"
    print("[P3-2/ETIS] ✓ 门禁4 冻结汇总 md5 未变")

    # ---- 另存 8 格汇总（不动原文件） -------------------------------------- #
    base = []
    if os.path.exists(SUMMARY):
        with open(SUMMARY, "r", encoding="utf-8") as f:
            base = json.load(f)
    merged = list(base) + [res]
    with open(SUMMARY_FULL8, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[P3-2/ETIS] 8 格汇总另存 -> {SUMMARY_FULL8}（原 7 格文件未动）")

    print("=" * 78)
    print(f"[P3-2/ETIS] {CONFIG_NAME}: Dice {res['dice']:.2f} | HD95 {res['hd95']:.1f} | "
          f"崩溃(Dice<10) {res['collapse_rate_dice_lt_10'] * 100:.1f}%")
    print("=" * 78)

    # ---- 落 E 盘留痕 ------------------------------------------------------ #
    out_dir = os.path.join(E_ROOT, "03_results", "raw", "full_module")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "p3_2_etis_gate.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": CONFIG_NAME,
            "params_M": p_m,
            "n_etis": int(len(ds)),
            "dice_from_results_json": res["dice"],
            "dice_independent_recompute": dice_re,
            "dice_delta": delta,
            "summary_md5_before": before_md5,
            "summary_md5_after": after_md5,
            "summary_full8": SUMMARY_FULL8,
        }, f, indent=2, ensure_ascii=False)
    print(f"[P3-2/ETIS] 门禁留痕 -> {out_dir}/p3_2_etis_gate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
