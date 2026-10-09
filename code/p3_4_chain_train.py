# -*- coding: utf-8 -*-
"""
p3_4_chain_train.py -- P3-4：串行训练多个容量匹配模型（一次启动，无人值守）

为什么需要它
------------
单个容量匹配模型要 **2–3.5 h**（实测：消融 7 格 101–439 min/模型，P3-2 48.42 M 实测
128.4 s/epoch ≈ 3.6 h）。若一个模型一个 `longrun` 任务手动接力，每段之间会出现
**人工等待的空档**（等通知 → 再启下一条），而 GPU 是唯一瓶颈。
本脚本把 N 个训练**串在一个进程里**，中间无空档：

    bf72 → bf79 → bf71        （★ 用户 2026-09-14 选定：先跑 3 个决定性的）

设计
----
* 每个模型用 **独立子进程** 调 `p3_4_train_capacity.py`（互不污染：各自 import 时才注入
  `ABLATION_CONFIGS`，故必须分开进程，不能在同一进程里连训）。
* **失败即止**：任一子训练非 0 退出 → 停止后续（避免在坏状态上继续烧 GPU），
  并**保留已完成模型的权重**（各自 `best_model.pth` 已落盘）。
* 全程走 `longrun.py`：一份日志 / 一个状态文件 / 断掉也留得下已跑部分。

跑法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_4_chain -- $PY E:/paper2_ablation_reliability/02_code/analysis/p3_4_chain_train.py --bfs 72 79 71

烟测（1 epoch 全部，验证链路；勿用于正式）：
    ... --bfs 72 79 71 --epochs 1 --output_dir ./experiments/_p34_smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "p3_4_train_capacity.py")

# 用户 2026-09-14 选定：3 个决定性对（H5 依赖的三处）
DECISIVE = [72, 79, 71]        # egm_dpa / egm_msfa / egm_only
REMAINDER = [66, 73, 74]       # dpa_only / msfa_only / dpa_msfa（后续补全表）

ROLE = {
    64: "↔ baseline（免费：复用 ablation/baseline，不需重训）",
    66: "↔ dpa_only（后续补全）",
    71: "↔ egm_only（已知反例）",
    72: "↔ egm_dpa（★ 优势所在，最决定性）",
    73: "↔ msfa_only（后续补全）",
    74: "↔ dpa_msfa（后续补全）",
    79: "↔ egm_msfa（最优配置）",
}


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--bfs", type=int, nargs="+", default=DECISIVE)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", type=str, default="./experiments/capacity_matched")
    a = ap.parse_args()

    py = sys.executable
    t0 = time.time()
    print("=" * 84)
    print(f"[P3-4] 串行训练 {len(a.bfs)} 个容量匹配模型：{a.bfs}")
    print(f"[P3-4] 训练器 {TRAIN}")
    print(f"[P3-4] epochs={a.epochs} batch={a.batch_size} out={a.output_dir}")
    print("[P3-4] 预算（实测校准 ~2–3.5 h/模型）：预计 "
          f"{len(a.bfs) * 2:.0f}–{len(a.bfs) * 3.5:.1f} h")
    print("=" * 84)

    done, failed = [], None
    for i, bf in enumerate(a.bfs, 1):
        print("\n" + "#" * 84)
        print(f"### [{i}/{len(a.bfs)}] bf={bf}  {ROLE.get(bf, '')}")
        print(f"### 已用时 {(time.time() - t0) / 60:.1f} min")
        print("#" * 84, flush=True)

        cmd = [py, TRAIN, "--bf", str(bf),
               "--epochs", str(a.epochs), "--batch_size", str(a.batch_size),
               "--output_dir", a.output_dir]
        print("[P3-4] 执行:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            failed = (bf, rc)
            print(f"\n[P3-4] ✗ bf={bf} 退出码 {rc} —— 停止后续（已完成 {done}）", flush=True)
            break
        done.append(bf)
        print(f"\n[P3-4] ✓ bf={bf} 完成（累计 {(time.time() - t0) / 60:.1f} min）", flush=True)

    print("\n" + "=" * 84)
    print(f"[P3-4] 结束 | 完成 {done} | 失败 {failed} | 总耗时 {(time.time() - t0) / 60:.1f} min")
    rest = [b for b in a.bfs if b not in done]
    if rest and not failed:
        print(f"[P3-4] 未跑 {rest}")
    if failed:
        return failed[1] or 1
    print("[P3-4] 下一步：跑 p3_4_etis_capacity.py（幂等，逐个评测 + 出分支 A/B 判定）")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
