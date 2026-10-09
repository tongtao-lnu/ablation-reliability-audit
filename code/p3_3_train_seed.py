# -*- coding: utf-8 -*-
"""
文件名: p3_3_train_seed.py
功能: 【论文二 P3-3】用**指定种子**重训**一个**消融配置，产物落 E 盘独立目录。

为什么要包一层（三个理由，都不是洁癖）
--------------------------------------
① **防覆盖**：`train_single.py` 没有种子子目录（`save_dir = <output_dir>/<type>/<model>`），
   多种子互跑会**互相覆盖**。本脚本把每个 (种子, 配置) 送进互不相同的 `output_dir`。
② **第 8 格无法直接训**：`egm_dpa_msfa` **不在** `models/ablation_models.py::ABLATION_CONFIGS`
   （该表只有 7 项）。直接 `train_single.py --model egm_dpa_msfa` 会
   `❌ 未知的消融配置` → `sys.exit(1)`（**本段冒烟校准已实测踩到**）。
   故本脚本**运行时注入**该配置（与 `p3_2_train_full_module.py` 同一手法），
   **不修改 D 盘任何源码**。
③ **零改动 train_single.py**：它是论文一的生产脚本，改它即毁可复现性。
   本脚本只做「注入 + 设 argv + 校验产物」，训练逻辑 100% 仍由 `train_single.py` 执行。
   `sys.argv` 注入对在其内部解析 arg 的脚本有效；`ABLATION_CONFIGS` 是**同一 dict 对象**，
   故运行时注入对其可见（`train_single.py` 第 18 行 `from ... import ABLATION_CONFIGS`）。

路径与产物（全部落 E 盘，G7）
---------------------------
    --root 默认 E:/paper2_ablation_reliability/03_results/raw/multiseed/exp
    本脚本的 output_dir = <root>/<seed>
    train_single.py 产 <root>/<seed>/ablation/<config>/best_model.pth + results.json
        （`ablation` 这一层是 train_single.py 的 `--type` 造成的，不是笔误）

⚠️ 除 `--seed` 外的一切超参**必须**保持 train_single.py 的默认值
   （epochs 100 / batch 8 / lr 1e-4 / val_interval 5 / early_stop 30），
   否则就不是「同协议多种子」，而是另一套训练 —— 种子效应会被协议差异污染。
   `--batch_size` 默认 **8** 有独立依据：`00_docs/P2-4执行报告_2026-09-14.md` §协议
   记载消融 7 格为「batch 8–16」。**注意**：第 8 格 `egm_dpa_msfa` 的种子 42 权重
   是用 **batch 4** 训的（见 `00_docs/P3-2执行报告_2026-09-14.md`）——
   这是一处**既有数据的协议不对称**，规格 §6.3 已预注册其处置（该格不参与副判据）。

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_3_train_seed.py \
        --config msfa_only --seed 123
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

D_ROOT = "D:/medical_segmentation"
DATA_DIR = os.path.join(D_ROOT, "processed_data")
DEFAULT_ROOT = "E:/paper2_ablation_reliability/03_results/raw/multiseed/exp"

EIGHTH_CONFIG = "egm_dpa_msfa"
FULL_CFG = {
    "use_egm": True,
    "use_dpa": True,
    "use_msfa": True,
    "description": "U-Net + EGM + DPA + MSFA (2^3 full)",
}
EXPECTED_PARAMS_M = 48.4154      # 与 p3_2_train_full_module.py 冻结值一致
PARAMS_TOL_M = 0.001


def artifact_paths(root: str, seed: int, config: str) -> tuple:
    """train_single.py 的实际落点（`ablation` 层来自 --type）。"""
    sd = os.path.join(root, str(seed), "ablation", config)
    return os.path.join(sd, "best_model.pth"), os.path.join(sd, "results.json")


def inject_eighth(verbose=True) -> float:
    """运行时注入第 8 格并自检参数量。返回参数量(M)。

    与 `p3_2_train_full_module.py` 的冻结断言一致 —— 若上游 `models/ablation_models.py`
    被改动，这里会**先失败**，而不是安静地训出一个参数结构不同的模型。
    """
    import models.ablation_models as am
    ref = am.ABLATION_CONFIGS["dpa_msfa"]
    assert set(FULL_CFG) == set(ref), "第 8 格字段与既有 7 格不一致（冻结失败）"
    assert FULL_CFG["use_egm"] and FULL_CFG["use_dpa"] and FULL_CFG["use_msfa"]
    am.ABLATION_CONFIGS[EIGHTH_CONFIG] = FULL_CFG
    net, _ = am.get_ablation_model(EIGHTH_CONFIG, in_channels=3, num_classes=1,
                                   base_filters=64)
    p = sum(x.numel() for x in net.parameters()) / 1e6
    del net
    assert abs(p - EXPECTED_PARAMS_M) < PARAMS_TOL_M, (
        "参数量 %.4f M 与冻结值 %.4f M 不符" % (p, EXPECTED_PARAMS_M))
    if verbose:
        print("[P3-3] 第 8 格运行时注入成功（未改动 D 盘源码）| 参数量 %.4f M" % p, flush=True)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-3: train one ablation config under one seed")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    # 冻结的超参（默认值 == train_single.py 默认值；改它们要显式写出来，好被审计）
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_interval", type=int, default=5)
    ap.add_argument("--early_stop", type=int, default=30)
    ap.add_argument("--skip-done", action="store_true",
                    help="若 results.json 与 best_model.pth 都在则跳过（幂等，断点续跑用）")
    a = ap.parse_args(argv)

    wpath, rpath = artifact_paths(a.root, a.seed, a.config)

    if a.skip_done and os.path.exists(rpath) and os.path.exists(wpath):
        r = json.load(open(rpath, encoding="utf-8"))
        print("[P3-3] SKIP 已存在: seed=%d cfg=%s  Dice=%.4f  用时=%.1f min"
              % (a.seed, a.config, r["test_metrics"]["Dice"], r["training_time_min"]),
              flush=True)
        return 0

    # train_single.py 依赖 `./processed_data` 之类相对路径 + 包导入 ⇒ 必须 chdir 到工程根
    os.chdir(D_ROOT)
    if D_ROOT not in sys.path:
        sys.path.insert(0, D_ROOT)

    if a.config == EIGHTH_CONFIG:
        inject_eighth()

    out_dir = os.path.join(a.root, str(a.seed))
    os.makedirs(out_dir, exist_ok=True)

    # ⚠️ 冻结协议：除 seed 外与 train_single.py 默认逐项一致
    sys.argv = [
        "train_single.py",
        "--model", a.config,
        "--type", "ablation",
        "--seed", str(a.seed),
        "--epochs", str(a.epochs),
        "--batch_size", str(a.batch_size),
        "--lr", repr(a.lr),
        "--val_interval", str(a.val_interval),
        "--early_stop", str(a.early_stop),
        "--data_dir", DATA_DIR,
        "--output_dir", out_dir,
    ]

    print("=" * 84, flush=True)
    print("[P3-3] 训练 seed=%d  config=%s" % (a.seed, a.config), flush=True)
    print("[P3-3] 冻结超参: epochs=%d batch=%d lr=%g val_interval=%d early_stop=%d"
          % (a.epochs, a.batch_size, a.lr, a.val_interval, a.early_stop), flush=True)
    print("[P3-3] 权重落点: " + wpath, flush=True)
    print("[P3-3] 生效 argv: " + " ".join(sys.argv[1:]), flush=True)
    print("=" * 84, flush=True)

    from train_single import main as train_main

    t0 = time.time()
    try:
        train_main()
    except SystemExit as e:                       # train_single 对未知配置会 sys.exit(1)
        if e.code not in (0, None):
            print("[P3-3] ✗ 训练中止 seed=%d cfg=%s SystemExit=%r" % (a.seed, a.config, e.code),
                  flush=True)
            return int(e.code) if isinstance(e.code, int) else 5
    except Exception as e:                        # noqa: BLE001 —— 必须把异常变成非零退出码
        import traceback
        traceback.print_exc()
        print("[P3-3] ✗ 训练异常 seed=%d cfg=%s: %r" % (a.seed, a.config, e), flush=True)
        return 6
    dt = (time.time() - t0) / 60.0

    # 产物校验：非空 + 能读 + Dice 是有限正数（防"跑完但没落盘"）
    for p in (wpath, rpath):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print("[P3-3] ✗ 训练返回但产物缺失/为空: " + p, flush=True)
            return 3
    r = json.load(open(rpath, encoding="utf-8"))
    dice = float(r["test_metrics"]["Dice"])
    if not (dice == dice) or dice <= 0.0:
        print("[P3-3] ✗ test Dice 异常: %r" % dice, flush=True)
        return 4

    print("[P3-3] ✓ seed=%d cfg=%s 完成 | 测试 Dice=%.4f | best_epoch=%d | %.1f min"
          % (a.seed, a.config, dice, int(r.get("best_epoch", -1)), dt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
