# -*- coding: utf-8 -*-
"""
文件名: p3_3_calibrate.py
功能: 【论文二 P3-3】冒烟校准 —— 跑前实测每个配置的**真实**训练速率，冻结预算（G6b）。

为什么不能沿用历史数字
----------------------
`egm_dpa` 的历史 `results.json` 记 438.5 min，而参数量更大的 `egm_msfa` 只用 147.2 min。
已核查：**8 个配置的训练时间区间两两不重叠**，故 438.5 不是并发拖慢所致。
但"不是并发所致"≠"今天重跑还是 438.5"——原因未知（可能当时降频/占卡/温控）。
⇒ 预算必须**在当下这台机器上实测**，不能把可疑的历史值当承诺写进冻结文档。

做法
----
对每个配置跑 `--epochs <N>`（默认 1）的**真实训练**，量墙钟，则
    est_min_100ep = 墙钟min / N × 100
该估计**偏保守**：它把「1 次验证 + 1 次测试」的成本也按 ×100 放大，
而在正式 100 epoch 里验证只发生 20 次、测试只发生 1 次。偏保守是安全的。

顺带价值（重要）：这一步同时**逐配置验证了整条训练链路**
（dataloader / CombinedLoss / 模型前向 / 落盘），把 import 或路径类错误
在夜里暴露，而不是在第 9 小时暴露。

产物
    03_results/stats/p3_3_calibration.json   （供 p3_3_chain_train.py --eta-json 与规格文档引用）

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_3_calibrate.py --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "p3_3_train_seed.py")

KEY_CONFIGS = ["msfa_only", "dpa_msfa", "egm_msfa", "egm_dpa_msfa", "egm_dpa"]
HIST_MIN = {"msfa_only": 107.7, "dpa_msfa": 138.8, "egm_msfa": 147.2,
            "egm_dpa_msfa": 213.5, "egm_dpa": 438.5}

SMOKE_ROOT = "E:/paper2_ablation_reliability/03_results/raw/_p33_calib"
OUT_JSON = "E:/paper2_ablation_reliability/03_results/stats/p3_3_calibration.json"
SEED = 123          # 校准用种子（与正式新训种子一致，不额外发明）


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-3 pre-run throughput calibration")
    ap.add_argument("--configs", nargs="+", default=KEY_CONFIGS)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--root", default=SMOKE_ROOT)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--early_stop", type=int, default=30)
    a = ap.parse_args(argv)

    t0 = time.time()
    print("=" * 88, flush=True)
    print("[P3-3-calib] 配置 %s | 每配置 %d epoch | batch=%d" % (a.configs, a.epochs, a.batch_size), flush=True)
    print("[P3-3-calib] 冒烟根 %s （**非正式产物**，可整个删掉）" % a.root, flush=True)
    print("[P3-3-calib] 预计 ~%.1f min" % (sum(HIST_MIN.get(c, 150) for c in a.configs)
                                          * a.epochs / 100.0), flush=True)
    print("=" * 88, flush=True)

    per, order = {}, []
    for c in a.configs:
        print("\n" + "#" * 88, flush=True)
        print("### 校准 %s" % c, flush=True)
        print("#" * 88, flush=True)
        cmd = [sys.executable, TRAINER, "--config", c, "--seed", str(SEED),
               "--root", a.root, "--epochs", str(a.epochs),
               "--batch_size", str(a.batch_size), "--val_interval", "1",
               "--early_stop", str(a.early_stop)]
        ts = time.time()
        rc = subprocess.call(cmd)
        dt = (time.time() - ts) / 60.0
        est = dt / a.epochs * 100.0
        per[c] = dict(config=c, epochs=a.epochs, wall_min=dt, rc=rc,
                      sec_per_epoch=dt * 60.0 / a.epochs,
                      est_min_100ep=est, est_h_100ep=est / 60.0,
                      hist_min_100ep=HIST_MIN.get(c),
                      ratio_vs_hist=(est / HIST_MIN[c]) if HIST_MIN.get(c) else None)
        order.append(c)
        print("\n[P3-3-calib] %-14s 墙钟 %6.2f min / %d ep  ⇒  %.1f min/epoch  ⇒  100 ep ≈ %.1f min (%.2f h)"
              % (c, dt, a.epochs, dt / a.epochs, est, est / 60.0), flush=True)
        if HIST_MIN.get(c):
            print("[P3-3-calib] %-14s 历史值 %.1f min ⇒ 今日/历史 = ×%.2f"
                  % (c, HIST_MIN[c], est / HIST_MIN[c]), flush=True)
        if rc != 0:
            print("[P3-3-calib] ⚠ %s 退出码 %d —— 该配置链路有问题，正式链会失败，先修" % (c, rc), flush=True)

    total = sum(per[c]["est_min_100ep"] for c in per)
    ok = all(per[c]["rc"] == 0 for c in per)
    payload = dict(
        segment="P3-3", kind="throughput_calibration",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/p3_3_calibrate.py",
        method="est_min_100ep = wall_min / epochs * 100（偏保守：把 1 次验证+测试也按 ×100 放大）",
        seed_used=SEED, epochs_probed=a.epochs, batch_size=a.batch_size,
        smoke_root=a.root,
        per_config=per, order=order,
        all_ok=ok,
        per_seed_min_5cfg=total, per_seed_h_5cfg=total / 60.0,
        two_new_seeds_h_5cfg=total * 2 / 60.0,
        wall_total_min=(time.time() - t0) / 60.0,
        disclaimer=("冒烟基于 %d epoch 的墙钟外推，非 100 epoch 实测；"
                    "正式链每完成一格会用实测覆盖该格的 ETA（见 chain_progress.json）。" % a.epochs),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88, flush=True)
    print("[P3-3-calib] 全部配置链路 OK: %s" % ok, flush=True)
    print("[P3-3-calib] 关键 5 格每种子 ≈ %.1f min = %.2f h" % (total, total / 60.0), flush=True)
    print("[P3-3-calib] 2 个新种子（种子 42 复用）≈ %.0f min = %.1f h"
          % (total * 2, total * 2 / 60.0), flush=True)
    print("[P3-3-calib] 已写 " + a.out, flush=True)
    print("=" * 88, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
