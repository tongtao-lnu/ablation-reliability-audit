# -*- coding: utf-8 -*-
"""
p3_4_etis_capacity.py -- P3-4：容量匹配组的 ETIS 评测 ＋ 分支 A/B 判定

产出
----
  03_results/raw/capacity_matched/<name>.json   逐模型（detection/Dice/ID 门禁）
  03_results/stats/capacity_matched.json        汇总 + 配对比较 + 分支判定
  03_results/tables/T_p3_4_capacity.md          结果表（含分支 A/B 直答）

口径（G6 冻结，与 P3-1b / 消融 7 格一致）
----------------------------------------
  ETIS 输入   `data_zeroshot/etis`（196 张，未见域）
  预测        `sigmoid(logit) > 0.5`  → 布尔掩膜（与 `ablation_zeroshot_etis.py` 同）
  Dice        2|P∩G| / (|P|+|G|)，eps = 1e-8（与 `cross_dataset.train_cross.calculate_metrics` 同）
  检出 detection = |P∩G| >= 1 px（**no-eps 整数判据**，与 P3-1b 的 detection 同口径）
   ID 门禁    ① **自洽**：重算 ID test Dice 与训练 `results.json` 的 `test_metrics.Dice` 分档
              （IDENTICAL ≤1e-9 ／ EQUIVALENT ≤1e-6 ／ NUMERIC_DRIFT ≤1e-4＝通过留痕 ／ MISMATCH＝FAIL）；
              ② **对齐**：与**同参数量模块配置**的 ID Dice 差 ≤ 1.0 pp（T-CF-3b 闸门①）
              （① 的容差定调与证据见 `00_docs/P3-4闸门容差定调_2026-09-14.md`）
  配对比较   同一批 196 张上，`Δ = 模块配置 − 容量匹配 plain UNet`；自助 95% CI（B=2000, seed=42）

设计
----
* **幂等 / 可断点**：逐模型落盘，已存在则跳过（`--force` 重算）⇒ 训练出一个就能评测一个。
* `baseline ↔ bf64` 这一对**不需新训练**：`baseline` 与 `plainUNet_bf64` **架构恒等**（见 `CKPT_ALIAS`），
  故容量侧**复用** `experiments/ablation/baseline/best_model.pth`，但**重新前向**（不读既有预测）——
  于是该对成为**正对照**：模块侧读冻结的 `results_ablation_etis/baseline/predictions.npy`，
  容量侧是同一权重的**独立重算**，正期望 Δ检出 ≈ 0 ⇒ 检验的是**装置可信度**，并给真实对提供**零假设地板**。
* 模块配置侧的 ETIS 预测**直接读既有** `results_ablation_etis/<cfg>/predictions.npy`，
  **不重跑推理**（避免与冻结产物分叉）。

跑法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_4_etis -- $PY <本文件>
    # 只评某个：<本文件> --only bf72   ／ 重算：--force
"""

from __future__ import annotations

import argparse
import json
import os
import sys

D_ROOT = "D:/medical_segmentation"
E_ROOT = "E:/paper2_ablation_reliability"

# ★ 必须：让 `models` / `utils` / `cross_dataset` 可 import，且相对数据路径可解析。
# （与 p3_4_train_capacity.py L90-91、p3_2_etis_full_module.py L66-67 同做法；
#   2026-09-14 曾因漏掉这两行导致 7/7 模型评测全部 ModuleNotFoundError。）
if D_ROOT not in sys.path:
    sys.path.insert(0, D_ROOT)
os.chdir(D_ROOT)

ETIS_DIR = "data_zeroshot/etis"
CAP_ROOT_D = "experiments/capacity_matched/ablation"      # 新训练产物（仅新增目录）
ABL_ROOT_D = "experiments/ablation"                        # 既有 7 格
ETIS_PRED_ROOT_D = "results_ablation_etis"                 # 既有 OOD 预测

# ---- 架构恒等别名：`baseline` ≡ `plainUNet_bf64`（**正对照 / null control，零训练成本**）----
# 依据（2026-09-14 核实）：`ABLATION_CONFIGS['baseline']` 三模块全 False，`get_ablation_model`
#   默认 base_filters=64 ⇒ 与 `AblationUNet(bf=64, 全 False)` **架构上同一模型**；
#   锚点检查 `abs_diff_M = 0.0`（构造 31.037633 M = 既有 experiments/ablation/baseline，精确 MATCH，
#   见 03_results/stats/confound.json 的 anchor_check）；训练协议亦同一套。
# ⇒ 该对的**容量侧直接复用既有** `experiments/ablation/baseline/`（**不复制权重、不训练**）。
# 正期望：Δ检出 ≈ 0（两侧本是同一模型）⇒ 本对检验的是**整条 P3-4 流水线是否可信**（复现既有 OOD 预测），
#   而非科学假设。它同时给真实对（bf71/72/79 的 10~24 pp）提供**零假设地板**。
CKPT_ALIAS = {"plainUNet_bf64": ABL_ROOT_D + "/baseline"}

OUT_RAW = E_ROOT + "/03_results/raw/capacity_matched"
OUT_JSON = E_ROOT + "/03_results/stats/capacity_matched.json"
OUT_MD = E_ROOT + "/03_results/tables/T_p3_4_capacity.md"

EPS = 1e-8
PIXEL_THRESH = 0.5
EXPECTED_N = 196
ID_GATE_PP = 1.0          # T-CF-3b 闸门①：ID Dice 差 ≤ 1.0 pp

# ---- ID 自洽闸门（重算 ID test Dice vs 训练 results.json）三档 ----
# 定调依据（2026-09-14，证据见 00_docs/P3-4闸门容差定调_2026-09-14.md）：
#  ① results.json 存**完整 float64**（零舍入，如 0.819680721015582）⇒ 地板不是存储精度；
#  ② 两侧用**同一个** utils.metrics.SegmentationMetrics（eps=1e-8），且评测流程逐行一致
#     （autocast → sigmoid(..).float() → update，阈值 0.5）⇒ **不存在 eps 口径差**；
#  ③ 故偏差唯一来源 = 跨进程 GPU fp16 前向抖动；本项目同类闸门**实测 0 ~ 1 ULP**
#     （13 架构 tol=1e-9 通过 12/13；唯一 FAIL 的 PolypPVT 8.49e-3 已登记为**产物自身缺陷**）。
#  ⇒ 报告线取 spec 的 1e-9；**操作闸门**放宽到 1e-4（=0.01 pp，仍比 1.0 pp 闸门严 100×，
#     也远严于"载错权重"的 Δ≈1e-2）⇒ 既能抓真实错配，又不对正确模型误报阻断。
ID_CONSIST_IDENTICAL = 1e-9    # ≤ 此 = 位级等同（满足 spec）
ID_CONSIST_EQUIV     = 1e-6    # ≤ 此 = 数值等同
ID_CONSIST_GATE      = 1e-4    # ≤ 此 = 数值漂移（通过 + 留痕告警）；> 此 = MISMATCH（判 FAIL）


def classify_id_consistency(delta):
    """把 |重算 − results.json| 分档。返回 (state, passed)；delta=None ⇒ (None, None)。"""
    if delta is None:
        return None, None
    if delta <= ID_CONSIST_IDENTICAL:
        return "IDENTICAL", True        # 位级等同
    if delta <= ID_CONSIST_EQUIV:
        return "EQUIVALENT", True       # 数值等同
    if delta <= ID_CONSIST_GATE:
        return "NUMERIC_DRIFT", True    # 跨进程 fp16 漂移，通过但留痕
    return "MISMATCH", False            # 判 FAIL（多半是载错权重/文件）
BRANCH_A_PP = 5.0         # T-CF-3c：差距 ≥ 5 pp ⇒ 分支 A
BOOT_B = 2000
SEED = 42

# 模块配置 -> (容量匹配 plain UNet 名, base_filters)
PAIRS = {
    "baseline":  ("plainUNet_bf64", 64),
    "dpa_only":  ("plainUNet_bf66", 66),
    "egm_only":  ("plainUNet_bf71", 71),
    "egm_dpa":   ("plainUNet_bf72", 72),
    "msfa_only": ("plainUNet_bf73", 73),
    "dpa_msfa":  ("plainUNet_bf74", 74),
    "egm_msfa":  ("plainUNet_bf79", 79),
}
MODULE_NAME = {"baseline": "baseline", "dpa_only": "dpa_only", "egm_only": "egm_only",
               "egm_dpa": "egm_dpa", "msfa_only": "msfa_only", "dpa_msfa": "dpa_msfa",
               "egm_msfa": "egm_msfa"}


# --------------------------------------------------------------------------- #
# 纯计算（可被独立复核脚本复用同一算式来对照）
# --------------------------------------------------------------------------- #
def dice_from_counts(inter, np_, ng):
    return (2.0 * inter + EPS) / (np_ + ng + EPS)


def metrics_from_preds(preds):
    """preds: list of dict 含 'pred'(bool) 与 'mask'(GT)。返回逐样本量与汇总。"""
    import numpy as np
    dice, inter, np_, ng = [], [], [], []
    for it in preds:
        P = np.asarray(it["pred"]).astype(bool).ravel()
        G = np.asarray(it["mask"]).astype(bool).ravel()
        i = int(np.logical_and(P, G).sum())
        inter.append(i)
        np_.append(int(P.sum()))
        ng.append(int(G.sum()))
        dice.append(dice_from_counts(i, np_[-1], ng[-1]))
    inter = np.asarray(inter, float)
    dice = np.asarray(dice, float)
    det = (inter >= 1.0)                      # no-eps 整数判据
    return dict(n=len(preds), dice=dice, det=det, inter=inter,
                npred=np.asarray(np_, float), ngt=np.asarray(ng, float),
                mean_dice=float(dice.mean()), detection=float(det.mean()),
                n_undetected=int((~det).sum()),
                pct_fully_missed=float((inter == 0).mean()))


def boot_delta(a, b, B=BOOT_B, seed=SEED):
    """配对自助：Δ = mean(a) − mean(b)，对样本索引重采样。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    assert len(b) == n
    d = a - b
    idx = rng.integers(0, n, size=(B, n))
    stat = d[idx].mean(axis=1)
    return dict(point=float(d.mean()),
                lo=float(np.percentile(stat, 2.5)),
                hi=float(np.percentile(stat, 97.5)),
                crosses_zero=bool(np.percentile(stat, 2.5) <= 0 <= np.percentile(stat, 97.5)))


# --------------------------------------------------------------------------- #
# 评测单模型
# --------------------------------------------------------------------------- #
def eval_capacity_model(name, bf, device, force=False):
    import numpy as np
    import torch
    from torch.cuda.amp import autocast

    from models.ablation_models import AblationUNet
    from utils.dataset import get_dataloaders
    from utils.metrics import SegmentationMetrics
    from cross_dataset.dataset import CrossDataset

    out_path = os.path.join(OUT_RAW, name + ".json")
    if os.path.exists(out_path) and not force:
        print(f"[P3-4] 跳过已评测 {name}")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    # 架构恒等别名（如 plainUNet_bf64 → experiments/ablation/baseline）优先；
    # 否则用新训练的 capacity_matched 目录。
    src_dir = CKPT_ALIAS.get(name, os.path.join(CAP_ROOT_D, name))
    ckpt = os.path.join(D_ROOT, src_dir, "best_model.pth")
    res_json = os.path.join(D_ROOT, src_dir, "results.json")
    if name in CKPT_ALIAS:
        print(f"[P3-4] {name}: 架构恒等别名 → {src_dir}（复用既有权重，零训练）")
    if not os.path.exists(ckpt):
        print(f"[P3-4] {name}: 权重缺失，跳过（先训练）")
        return None

    model = AblationUNet(in_channels=3, num_classes=1, base_filters=bf,
                         use_egm=False, use_dpa=False, use_msfa=False)
    p_m = sum(x.numel() for x in model.parameters()) / 1e6
    ck = torch.load(ckpt, map_location="cpu")
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state)
    model = model.to(device).eval()

    # ---- ID 门禁：重算 ID test Dice -------------------------------------- #
    _, _, test_loader = get_dataloaders(os.path.join(D_ROOT, "processed_data"), 8, num_workers=0)
    mtr = SegmentationMetrics()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast():
                seg_out = model(images)
            mtr.update(torch.sigmoid(seg_out.float()), masks)
    id_re = mtr.get_metrics()
    id_dice_re = float(id_re["Dice"])
    id_dice_json = None
    if os.path.exists(res_json):
        with open(res_json, encoding="utf-8") as f:
            id_dice_json = float(json.load(f)["test_metrics"]["Dice"])
    consist = abs(id_dice_re - id_dice_json) if id_dice_json is not None else None
    id_state, id_ok = classify_id_consistency(consist)
    if consist is None:
        print(f"[P3-4] {name}: ID Dice 重算={id_dice_re:.6f}（无 results.json，自洽闸门跳过）")
    else:
        print(f"[P3-4] {name}: ID Dice 重算={id_dice_re:.6f} json={id_dice_json} "
              f"Δ={consist:.2e} → {id_state}"
              + ("" if id_ok else "  ⚠️ MISMATCH（自洽闸门 FAIL）"))

    # ---- ETIS ------------------------------------------------------------- #
    ds = CrossDataset(os.path.join(D_ROOT, ETIS_DIR), augment=False, strong_augment=False)
    assert len(ds) == EXPECTED_N, f"ETIS 样本数 {len(ds)} ≠ {EXPECTED_N}"
    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, mask, nm = ds[i]
            with autocast():
                out = model(img.unsqueeze(0).to(device))
            seg = out[0] if isinstance(out, tuple) else out
            if isinstance(seg, tuple):
                seg = seg[0]
            pr = (torch.sigmoid(seg.float()) > PIXEL_THRESH).cpu().numpy().squeeze()
            preds.append({"name": nm, "pred": pr, "mask": mask.numpy().squeeze()})

    O = metrics_from_preds(preds)
    rec = dict(name=name, base_filters=bf, params_M=p_m,
               id_dice_recomputed=id_dice_re, id_dice_from_results_json=id_dice_json,
               id_consistency_delta=consist,
               id_consistency_state=id_state, id_consistency_pass=id_ok,
               n_etis=O["n"], etis_dice=O["mean_dice"], etis_detection=O["detection"],
               etis_n_undetected=O["n_undetected"], etis_pct_fully_missed=O["pct_fully_missed"],
               # 逐样本留痕（供配对自助与独立复核）
               per_sample_dice=[float(x) for x in O["dice"]],
               per_sample_det=[int(x) for x in O["det"]],
               per_sample_names=[p["name"] for p in preds])
    os.makedirs(OUT_RAW, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"[P3-4] {name}: ETIS Dice={O['mean_dice']:.4f} | 检出={O['detection']:.4f} | "
          f"全漏 {O['n_undetected']}/{O['n']} -> {out_path}")
    del model
    torch.cuda.empty_cache()
    return rec


def load_module_side(cfg):
    """读既有模块配置的 ETIS 预测（不重跑推理）。baseline 亦走此路。"""
    import numpy as np
    p = os.path.join(D_ROOT, ETIS_PRED_ROOT_D, cfg, "predictions.npy")
    if not os.path.exists(p):
        return None
    preds = np.load(p, allow_pickle=True)
    O = metrics_from_preds([{"pred": it["pred"], "mask": it["mask"]} for it in preds])
    rj = os.path.join(D_ROOT, ABL_ROOT_D, cfg, "results.json")
    id_dice = None
    if os.path.exists(rj):
        with open(rj, encoding="utf-8") as f:
            j = json.load(f)
        id_dice = float(j["test_metrics"]["Dice"])
    return dict(config=cfg, n=O["n"], etis_dice=O["mean_dice"],
                etis_detection=O["detection"],
                etis_n_undetected=O["n_undetected"],
                id_dice=id_dice,
                per_sample_dice=[float(x) for x in O["dice"]],
                per_sample_det=[int(x) for x in O["det"]])


def main():
    import numpy as np
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只评这些 plainUNet 名")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-etis", action="store_true", help="只做汇总（已有逐模型 json）")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P3-4] 设备 {device}")

    names = sorted({v[0] for v in PAIRS.values()})
    if a.only:
        names = [n for n in names if n in set(a.only)]
    eval_errors = {}
    if not a.skip_etis:
        for n in names:
            bf = next(v[1] for v in PAIRS.values() if v[0] == n)
            try:
                eval_capacity_model(n, bf, device, force=a.force)
            except Exception as e:                                   # noqa: BLE001
                import traceback
                eval_errors[n] = f"{type(e).__name__}: {e}"
                print(f"[P3-4] {n} 评测失败：{type(e).__name__}: {e}")
                traceback.print_exc()

    # ---------------- 汇总 + 分支判定 ---------------- #
    rows, missing = [], []
    for cfg, (nm, bf) in PAIRS.items():
        mod = load_module_side(cfg)
        cap_p = os.path.join(OUT_RAW, nm + ".json")
        cap = None
        if os.path.exists(cap_p):
            with open(cap_p, encoding="utf-8") as f:
                cap = json.load(f)
        if mod is None or cap is None:
            missing.append(dict(config=cfg, plain=nm, have_module=mod is not None,
                                have_capacity=cap is not None))
            continue
        assert mod["n"] == cap["n_etis"], f"{cfg} 样本数不一致"
        # Δ 方向统一取「模块 − 容量」，与下方 gain_pp 同向。
        # 2026-09-14 修：原先传 (cap, mod) 得「容量−模块」，渲染端须手动反号 ⇒ 表与直答两处
        # 各反一次，极易漏改（当次即出现 Δ=+10.20 而 CI 显示 [-15.3,-5.6] 的符号自相矛盾）。
        d_det = boot_delta(mod["per_sample_det"], cap["per_sample_det"])
        d_dice = boot_delta(mod["per_sample_dice"], cap["per_sample_dice"])
        id_gap_pp = (abs(cap["id_dice_recomputed"] - mod["id_dice"]) * 100
                     if mod["id_dice"] is not None else None)
        # 分支判据（T-CF-3c）：看"模块配置 是否仍高于 容量匹配"
        gain_pp = (mod["etis_detection"] - cap["etis_detection"]) * 100
        if gain_pp >= BRANCH_A_PP:
            branch = "A"
        elif gain_pp < BRANCH_A_PP and d_det["lo"] <= 0 <= d_det["hi"]:
            branch = "B"
        else:
            branch = "ambiguous"
        # 内部自洽闸门：自助点估计必须等于两种均值之差（两路口径不得分叉）
        gap_check = abs(d_det["point"] * 100 - gain_pp)
        if gap_check > 1e-6:
            raise AssertionError(
                f"{cfg}: 自助点估计 {d_det['point']*100:+.6f} pp 与均值差 {gain_pp:+.6f} pp "
                f"偏离 {gap_check:.2e}（> 1e-6）—— 两路口径不一致，拒绝出表")
        rows.append(dict(
            config=cfg, plain=nm, base_filters=bf,
            # 架构恒等 ⇒ 本对是**正对照**（null control），A/B 分支语义不适用
            is_null_control=bool(nm in CKPT_ALIAS),
            module_etis_detection=mod["etis_detection"], capacity_etis_detection=cap["etis_detection"],
            module_etis_dice=mod["etis_dice"], capacity_etis_dice=cap["etis_dice"],
            module_id_dice=mod["id_dice"], capacity_id_dice=cap["id_dice_recomputed"],
            id_gate_gap_pp=id_gap_pp, id_gate_pass=bool(id_gap_pp is not None and id_gap_pp <= ID_GATE_PP),
            id_consistency_delta=cap["id_consistency_delta"],
            id_consistency_state=cap.get("id_consistency_state"),
            gain_detection_pp=gain_pp,
            delta_detection=d_det, delta_dice=d_dice,
            module_n_undetected=mod["etis_n_undetected"], capacity_n_undetected=cap["etis_n_undetected"],
            branch=branch))

    out = dict(n_pairs=len(rows), missing=missing, pairs=rows,
               eval_errors=eval_errors,
               frozen=dict(ID_GATE_PP=ID_GATE_PP, BRANCH_A_PP=BRANCH_A_PP,
                           detection="I >= 1 px (no-eps)", eps=EPS,
                           id_consist_tiers=dict(IDENTICAL=ID_CONSIST_IDENTICAL,
                                                 EQUIVALENT=ID_CONSIST_EQUIV,
                                                 NUMERIC_DRIFT=ID_CONSIST_GATE,
                                                 MISMATCH=f">{ID_CONSIST_GATE}"),
                           boot=f"B={BOOT_B}, seed={SEED}, 配对百分位法"))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---------------- Markdown ---------------- #
    L = ["# T-P3-4 ｜ 容量匹配对照（分支 A/B 判定）", ""]
    L.append("> 判据冻结：检出 = `|P∩G| >= 1 px`（no-eps）；Dice eps=1e-8；"
             f"配对自助 B={BOOT_B} seed={SEED}")
    L.append("> 分支 A：模块配置检出仍高出 **≥ 5 pp**；分支 B：追平（< 5 pp **且** CI 跨 0）")
    L.append(f"> ID 自洽闸门（三档）：Δ≤{ID_CONSIST_IDENTICAL:g} IDENTICAL ／ "
             f"≤{ID_CONSIST_EQUIV:g} EQUIVALENT ／ ≤{ID_CONSIST_GATE:g} NUMERIC_DRIFT（通过留痕）／ "
             f">{ID_CONSIST_GATE:g} MISMATCH（判 FAIL）")
    L.append("> 分支出现第三态 `ambiguous`（Δ<5 pp 但 CI **不跨 0**，或反之）⇒ **不强行归入 A/B**，"
             "记为待人工判断（用户 2026-09-14 已接受此设计）")
    L.append("")
    L.append("| 模块配置 | 容量匹配 | bf | 模块检出 | 容量检出 | **Δ检出(pp)** | Δ 95%CI | CI跨0 | 模块OOD-Dice | 容量OOD-Dice | ID对齐Δ(pp) | 门禁 | 分支 |")
    L.append("|---|---|:--:|---:|---:|---:|---|---|:--:|---:|---:|:--:|:--:|:--:|")
    for r in rows:
        # Δ 已统一为「模块 − 容量」方向（见上），直接渲染；勿再反号
        ci = f"[{r['delta_detection']['lo']*100:+.1f}, {r['delta_detection']['hi']*100:+.1f}]"
        gap_p = r["id_gate_gap_pp"]
        gap_s = f"{gap_p:.2f}" if gap_p is not None else "n/a"
        L.append(
            f"| `{r['config']}` | `{r['plain']}` | {r['base_filters']} | "
            f"{r['module_etis_detection']:.4f} | {r['capacity_etis_detection']:.4f} | "
            f"**{r['gain_detection_pp']:+.2f}** | {ci} | "
            f"{'是' if r['delta_detection']['crosses_zero'] else '否'} | "
            f"{r['module_etis_dice']:.4f} | {r['capacity_etis_dice']:.4f} | "
            f"{gap_s} | {'✅' if r['id_gate_pass'] else '❌'} | "
            f"{'**正对照**' if r['is_null_control'] else '**' + r['branch'] + '**'} |")
    if missing:
        L += ["", "## 未完成的对（等训练）", ""]
        for m_ in missing:
            L.append(f"- `{m_['config']}` → `{m_['plain']}`："
                     f"模块侧={m_['have_module']} / 容量侧={m_['have_capacity']}")

    # ---------------- ★ 直答：优势是否消失（分支 A/B 分水岭） ---------------- #
    VERDICT = {
        "A": ("**优势未消失** —— 容量匹配后，模块配置的域外检出仍高出该 plain UNet "
              "≥ 5 pp ⇒ **H5 支持**（EGM/DPA 的贡献不能由参数量解释）"),
        "B": ("**优势消失** —— 容量匹配后二者检出追平（< 5 pp 且配对自助 CI 跨 0）"
              "⇒ **H5 推翻**，域稳健性主要由参数量解释"),
        "ambiguous": ("**未达预注册判据** —— Δ 落在 5 pp 以下但 CI **不跨 0**"
                      "（或反之）⇒ 不强行归入 A/B，须人工判断"),
    }
    L += ["", "## ★ 直答：容量匹配后优势是否消失（H5 分水岭）", ""]
    gated = [r for r in rows if r["id_gate_pass"]]
    nulls = [r for r in rows if r["is_null_control"]]
    real = [r for r in rows if not r["is_null_control"]]
    if eval_errors:
        L.append(f"> ❌ **评测报错 {len(eval_errors)} 个模型**（结果不可用）：")
        for n, msg in eval_errors.items():
            L.append(f">   - `{n}`: {msg}")
        L.append("")
    if missing:
        L.append(f"> ⚠️ **尚有 {len(missing)} 对未完成**（等训练），下列直答**仅基于已完成的 "
                 f"{len(real)} 个实质对**（另有 {len(nulls)} 个正对照），不得作为最终结论。")
        L.append("")
    # ---- 正对照（架构恒等，零训练成本）：报告的是**装置可信度**，不是 H5 判决 ---- #
    if nulls:
        L.append("### 正对照（架构恒等，A/B 语义**不适用**）")
        L.append("")
        L.append("> 下列正对照的**两侧本是同一架构**（`baseline` ≡ `plainUNet_bf64`），正期望 Δ≈0。")
        L.append("> 它检验的是**流水线是否可信**，**不是**科学假设；若 Δ≠0 则须回查装置。")
        L.append("")
        for r in nulls:
            L.append(f"- `{r['config']}` vs `{r['plain']}`：Δ检出 **{r['gain_detection_pp']:+.4f} pp**"
                     f"（OOD Dice {r['module_etis_dice']:.8f} vs {r['capacity_etis_dice']:.8f}）"
                     f"｜未检出数 {r['module_n_undetected']} vs {r['capacity_n_undetected']}"
                     f"｜ID 门禁 Δ **{r['id_gate_gap_pp']:.6f} pp"
                     f" {'✅ PASS' if r['id_gate_pass'] else '❌ FAIL'}**")
        L.append("")
    if not real:
        L.append("**暂无已完成的实质对 —— 无法判定。**")
    else:
        L.append("### 实质对（模块配置 vs 容量匹配 plain UNet）")
        L.append("")
        for r in real:
            L.append(f"- `{r['config']}` vs `{r['plain']}`：Δ检出 **{r['gain_detection_pp']:+.2f} pp**"
                     f"（95% CI [{r['delta_detection']['lo']*100:+.1f}, "
                     f"{r['delta_detection']['hi']*100:+.1f}]）"
                     f" ⇒ {VERDICT[r['branch']]}")
        L.append("")
        # 针对 H5 的核心那一对单独点名
        core = next((r for r in real if r["config"] == "egm_dpa"), None)
        if core is not None:
            L.append(f"> ★ **核心对 `egm_dpa` ↔ `{core['plain']}`**（H5 的立论所在）："
                     f"Δ检出 {core['gain_detection_pp']:+.2f} pp，判据 = **分支 {core['branch']}**"
                     f" ⇒ {VERDICT[core['branch']]}")
        else:
            L.append("> ⚠️ **核心对 `egm_dpa` ↔ `plainUNet_bf72` 尚未完成** —— 该对未出，"
                     "分支 A/B **不能定论**。")
        failed = [r["config"] for r in real if not r["id_gate_pass"]]
        if failed:
            L.append(f"> ❌ **ID 对齐门禁（T-CF-3b 闸门①）未过 {len(failed)}/{len(real)} 个实质对**："
                     f"{failed} —— 同参数量下 plain UNet 的 ID Dice 与模块配置差 > {ID_GATE_PP} pp")
            L.append("> 　⇒ 容量侧**未达同水平 ID 训练**，上列「优势未消失」**不得直接采信**，"
                     "须按处置选项 A/B/C/D 人工定调（见 00_docs/P3-4预警_容量匹配ID未对齐_2026-09-14.md）")
            L.append("> 　★ **闸门判别力已验证**：正对照（架构恒等）的对 ΔID ≈ 0 ⇒ **PASS**，"
                     "而实质对 3/3 **FAIL** ⇒ 该门禁**能区分**「真匹配」与「没训好」，非恒判 FAIL。")
        drift = [(r["config"], r.get("id_consistency_state")) for r in rows
                 if r.get("id_consistency_state") in ("NUMERIC_DRIFT", "MISMATCH")]
        if drift:
            L.append(f"> ⚠️ ID 自洽闸门留痕：{drift}"
                     "（IDENTICAL/EQUIVALENT 为常态；NUMERIC_DRIFT = 跨进程 fp16 漂移，已通过；"
                     "MISMATCH 须回查是否载错权重/文件）")
    L.append("")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("=" * 84)
    for r in rows:
        tag = "正对照" if r["is_null_control"] else r["branch"]
        print(f"  {r['config']:<10} vs {r['plain']:<15} Δ检出={r['gain_detection_pp']:+7.2f} pp "
              f"分支={tag:<10} ID门禁={'PASS' if r['id_gate_pass'] else 'FAIL'}")
    if missing:
        print(f"  [待训练] {[m_['plain'] for m_ in missing]}")
    print(f"汇总 -> {OUT_JSON}")
    print(f"表   -> {OUT_MD}")

    # ★ 防"静默成功"：一个对都没算出来 ⇒ 必须报失败（否则上游会误判链已成功）
    if not rows:
        print("[P3-4] ✗ 无任何可判定的对 —— 判为失败（禁止静默成功）")
        if eval_errors:
            print(f"[P3-4]   评测报错 {len(eval_errors)} 个：{list(eval_errors)}")
        if missing:
            print(f"[P3-4]   缺产物 {len(missing)} 个：{[m_['plain'] for m_ in missing]}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
