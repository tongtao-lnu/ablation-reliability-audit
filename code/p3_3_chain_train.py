# -*- coding: utf-8 -*-
"""
文件名: p3_3_chain_train.py
功能: 【论文二 P3-3】夜间串行链 —— 关键 5 格 × 多种子，一次启动无人值守，可续跑。

为什么需要它
------------
本机只有 **一块 8 G 卡**（用户长期约定：单卡串行，禁并行）。P3-3 是
**训练**实验，不是推理 —— 若一个 (种子,配置) 一个 `longrun` 任务手动接力，
每段之间会出现「等通知 → 再启下一条」的人工空档，而 GPU 是唯一瓶颈。
本脚本把 N 个训练**串在一个进程里**，中间不留空档。

与 P3-4 接力脚本的关键差别（**故意的，且必须知情**）
----------------------------------------------------
`p3_4_chain_train.py` 的策略是「失败即止」。本脚本改为「**失败即记、继续下一个**」。
理由：本链是**无人值守的整夜**任务，一个配置因偶发原因（OOM / 掉电重启）失败
就丢掉后面 10 小时，代价远大于收益；而本链**完全可续跑**，失败格随时补跑。
⇒ 每个 (种子,配置) 独立成败，全部记入进度文件，早上可一眼看出谁成了谁没成。

优先级（为什么是这个顺序）
--------------------------
⚠️ 顺序**不是**按耗时，而是按「**能早完成几个可计数反转对**」（规格 §6.2 跑前修订版）：

    egm_dpa → msfa_only → dpa_msfa → egm_msfa → egm_dpa_msfa

理由：`egm_dpa` 与 `msfa_only` 构成**旗舰反转对 #1**（主判据），必须**最先**齐活；
`dpa_msfa` 第 3 个 ⇒ 同时凑齐 #2、#3；`egm_msfa`/`egm_dpa_msfa` 仅与彼此配对
（#4 因第 8 格协议混杂被排除，规格 §6.3）⇒ 排最后。
若按"便宜的先跑"（msfa_only 先），旗舰对会被推到很晚 ⇒ 一旦截止，#1 无判定，主判据落空。

代价：`egm_dpa` 是最贵的一格（今日实测 ~170.8 s/epoch × 100 ≈ 288 min），
放在首位意味着**首格就要 ~5 h**。这是**知情选择** —— 用"首格偏贵"换"主判据早锁定"。

种子顺序：**先填满一个种子的 5 格**，再进下一个种子。
理由：一套**完整的 5 格排名**（单种子）本身即可与种子 42 对读；而「5 格各跑了一半」
在任何种子下都不可解读。

断点续跑与心跳
--------------
* 判定「已完成」= 该 (种子,配置) 的 `results.json` 与 `best_model.pth` **都在盘上且非空**。
  重启本脚本会自动跳过已完成的格（幂等）。
* 每完成/失败一格即**原子写**一次进度文件（先写 .tmp 再 os.replace），
  含每格状态、耗时、累计耗时、ETA。**这就是独立心跳**，与日志互为佐证。

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_3_chain -- $PY E:/paper2_ablation_reliability/02_code/analysis/p3_3_chain_train.py \\
        --seeds 123 2024 --eta-json E:/paper2_ablation_reliability/03_results/stats/p3_3_calibration.json

烟测（不产生正式产物）：
    ... --seeds 123 --configs msfa_only --epochs 1 --root E:/paper2_ablation_reliability/03_results/raw/_p33_smoke
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

# 关键 5 格（由 p3_3_freeze_facts.py 从「参与 ID↔OOD 排名反转」客观推出）
# ⚠️ 顺序 = **跑前修订版**（规格 §6.2）：按"能早完成几个可计数反转对"排，而非按耗时。
#    egm_dpa + msfa_only = 旗舰对 #1（主判据）⇒ 必须最先
#    dpa_msfa 第 3 个 ⇒ 完成 #2、#3
#    egm_msfa / egm_dpa_msfa 仅与彼此配对（#4 因协议混杂被排除，规格 §6.3）⇒ 排最后
KEY_CONFIGS = ["egm_dpa", "msfa_only", "dpa_msfa", "egm_msfa", "egm_dpa_msfa"]

DEFAULT_ROOT = "E:/paper2_ablation_reliability/03_results/raw/multiseed/exp"
DEFAULT_PROGRESS = "E:/paper2_ablation_reliability/03_results/raw/multiseed/chain_progress.json"

# ETA 表（**默认用跑前冒烟实测值**，不是历史值）。
#
# 实测来源：`p3_3_calibrate.py`（每配置 1 epoch 真训，读 `[Epoch 1/1] (Xs)` 那行）：
#     msfa_only 87.7 s/epoch | dpa_msfa 108.1 | egm_msfa 125.9 | egm_dpa 170.8
# 换算：est_min_100ep = 100×T/60 + ~3 min（20 次验证 + 1 次测试 + 启动，摊到 100 epoch 极小）
#
# ⚠️ 历史值（`<配置>/results.json` 的 training_time_min）**只在注释里留档，不进 ETA**：
#     msfa_only 107.7 | dpa_msfa 138.8 | egm_msfa 147.2 | egm_dpa 438.5 | egm_dpa_msfa 213.5
#     `egm_dpa` 的历史 438.5 min 对应 263 s/epoch，而**今日实测 170.8 s/epoch**（×0.65）
#     ⇒ 该历史值被判定为**异常**（非并发所致，见 00_docs/P3-3规格冻结_2026-09-15.md §5.1），
#       沿用它会给出错误的预算，故弃用。
MEASURED_MIN = {
    "msfa_only": 149.0,
    "dpa_msfa": 183.0,
    "egm_msfa": 213.0,
    "egm_dpa": 288.0,
    # egm_dpa_msfa 冒烟时失败（不在 ABLATION_CONFIGS，`sys.exit(1)`）；
    # **修复已落地并实测**：`p3_3_train_seed.py::inject_eighth()` 运行时注入，
    #   实测 7→8 配置、参数量 48.4154 M == 冻结值（2026-09-15 复核，INJECT_OK）。
    # 预算仍用历史 213.5 min × 今日/历史比例 ~1.35 ≈ 288 min **外推**（尚未真训过完整 100 epoch）。
    "egm_dpa_msfa": 288.0,
}
HIST_MIN = {"msfa_only": 107.7, "dpa_msfa": 138.8, "egm_msfa": 147.2,
            "egm_dpa_msfa": 213.5, "egm_dpa": 438.5}     # 仅留档，不参与 ETA


def atomic_write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def artifacts_ok(root, seed, config) -> bool:
    sd = os.path.join(root, str(seed), "ablation", config)
    w = os.path.join(sd, "best_model.pth")
    r = os.path.join(sd, "results.json")
    return (os.path.exists(w) and os.path.getsize(w) > 0
            and os.path.exists(r) and os.path.getsize(r) > 0)


def fmt_h(minutes: float) -> str:
    return "%.2f h" % (minutes / 60.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-3 overnight serial training chain")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 2024],
                    help="要**新训**的种子（种子 42 已在盘上，不进本链）")
    ap.add_argument("--configs", nargs="+", default=KEY_CONFIGS)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--progress", default=DEFAULT_PROGRESS)
    ap.add_argument("--eta-json", default=None,
                    help="冒烟校准产物；给了就用实测 s/epoch 算 ETA，否则用历史值兜底")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_interval", type=int, default=5)
    ap.add_argument("--early_stop", type=int, default=30)
    ap.add_argument("--stop-after", type=int, default=0,
                    help="跑够 N 格就停（0=不限）。用于试跑/限时。")
    ap.add_argument("--order", choices=["seed", "config"], default="seed",
                    help="格子的遍历顺序：seed=先填满一个种子（原冻结序）；"
                         "config=按配置交错（先让**同一配置的所有种子**齐活）。"
                         "⚠️ 本机实测远慢于预算（内存抖动），故 config 序能让"
                         "**主判据（旗舰对 #1 的三种子）**提前到第 4 格即可判定。")
    ap.add_argument("--plan-only", action="store_true",
                    help="只打印排布并退出，不启动任何训练（安全阀；核对 --order 必用它）")
    a = ap.parse_args(argv)

    # ---- ETA 表（实测优先）----
    # 基准 = 跑前冒烟实测表（MEASURED_MIN），而非历史值：
    #   今日机器比 1 月慢 1.3–1.9×，历史值会系统性低估预算。
    eta = dict(MEASURED_MIN)
    eta_src = "跑前冒烟实测表 MEASURED_MIN（缺失项回退历史）"
    for k, hv in HIST_MIN.items():
        eta.setdefault(k, hv)
    if a.eta_json and os.path.exists(a.eta_json):
        cal = json.load(open(a.eta_json, encoding="utf-8"))
        got, rejected = {}, []
        for k, v in cal.get("per_config", {}).items():
            est = v.get("est_min_100ep")
            if not est:
                continue
            est = float(est)
            # ⚠️ 合理性闸门（**这是本链实测踩到的坑**）：冒烟校准若**中途失败**
            #   （如第 8 格在注入修复前踩 `sys.exit(1)`），其 s/epoch 被严重低估 ——
            #   实测 egm_dpa_msfa 记到 5.84 s/epoch（折算 9.7 min / 100 epoch），
            #   若照收会把最贵的一格排成最便宜。规则：校准值 < 历史值×0.5 即不可信。
            hist = HIST_MIN.get(k)
            if hist and est < 0.5 * hist:
                rejected.append((k, est, hist))
                continue
            got[k] = est
        if got:
            eta.update(got)
            eta_src = "冒烟校准 " + os.path.basename(a.eta_json) + "（已过合理性闸门）"
        for k, est, hist in rejected:
            print("[P3-3-chain] ⚠️ 弃用不可信校准 ETA: %s = %.1f min < 历史 %.1f min ×0.5 "
                  "⇒ 回退冒烟实测表 %.1f min" % (k, est, hist, eta.get(k, -1)), flush=True)

    if a.order == "config":
        # 交错：先让"同一配置的所有种子"齐活 ⇒ 旗舰对 #1 的第 3 个种子最早就位
        plan = [(s, c) for c in a.configs for s in a.seeds]
    else:
        plan = [(s, c) for s in a.seeds for c in a.configs]
    total_eta = sum(eta.get(c, 0.0) for s, c in plan)

    t0 = time.time()
    print("=" * 92, flush=True)
    print("[P3-3-chain] 启动 %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[P3-3-chain] 新训种子 %s | 关键 %d 格 %s" % (a.seeds, len(a.configs), a.configs), flush=True)
    print("[P3-3-chain] 共 %d 格（种子×配置）| 产物根 %s" % (len(plan), a.root), flush=True)
    print("[P3-3-chain] 冻结超参 epochs=%d batch=%d lr=%g val_interval=%d early_stop=%d"
          % (a.epochs, a.batch_size, a.lr, a.val_interval, a.early_stop), flush=True)
    print("[P3-3-chain] ETA 来源: %s" % eta_src, flush=True)
    for s, c in plan:
        print("             seed=%-5d cfg=%-14s 预计 %7.1f min" % (s, c, eta.get(c, -1)), flush=True)
    print("[P3-3-chain] 预计总时长 %.2f h（单卡串行；断点续跑已启用）" % (total_eta / 60.0), flush=True)

    # ⚠️ 安全阀：只打印计划并退出，**绝不启动任何训练**。
    # 立此阀的原因（实测教训）：为核对 `--order` 的排布，曾用 `--stop-after 0` 试跑
    #   —— 而 0 的语义是"不限"，于是**真的启动了一次竞争训练**（虽因外因未上 GPU，
    #   但已在产物树下建了 `_p33_orderprobe/123`）。**凡"想看看计划长什么样"，
    #   一律用 `--plan-only`。**
    if a.plan_only:
        print("=" * 92, flush=True)
        print("[P3-3-chain] --plan-only：仅打印排布，未启动任何训练", flush=True)
        for i, (s, c) in enumerate(plan, 1):
            print("   [%2d/%d] seed=%-5d cfg=%-14s 预计 %7.1f min" % (i, len(plan), s, c, eta.get(c, -1)),
                  flush=True)
        print("=" * 92, flush=True)
        return 0

    print("=" * 92, flush=True)

    def snapshot(steps, done, failed, skipped):
        el = (time.time() - t0) / 60.0
        remain = sum(eta.get(s["config"], 0.0) for s in steps if s["status"] == "pending")
        return dict(
            segment="P3-3", kind="chain_progress",
            updated_at=datetime.now().isoformat(timespec="seconds"),
            seeds=a.seeds, configs=a.configs, root=a.root,
            frozen_hparams=dict(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                                val_interval=a.val_interval, early_stop=a.early_stop),
            eta_source=eta_src,
            elapsed_min=el,
            n_total=len(steps), n_done=len(done), n_failed=len(failed), n_skipped=len(skipped),
            eta_remaining_min=remain, eta_remaining_h=remain / 60.0,
            steps=steps,
        )

    steps = [dict(seed=s, config=c, status="pending", minutes=None, started=None,
                  finished=None, note="") for s, c in plan]
    done, failed, skipped = [], [], []
    idx = 0
    ran = 0
    for st in steps:
        s, c = st["seed"], st["config"]

        if artifacts_ok(a.root, s, c):
            st.update(status="skipped", note="产物已存在（断点续跑）")
            skipped.append((s, c))
            print("\n[P3-3-chain] ↷ 跳过 seed=%d cfg=%s（产物已在盘上）" % (s, c), flush=True)
            atomic_write_json(a.progress, snapshot(steps, done, failed, skipped))
            continue

        if a.stop_after and ran >= a.stop_after:
            st.update(status="pending", note="--stop-after 达上限，未启动")
            continue

        idx += 1
        print("\n" + "#" * 92, flush=True)
        print("### [%d/%d] seed=%d  cfg=%s（预计 %.1f min）"
              % (idx, len(plan), s, c, eta.get(c, -1)), flush=True)
        print("### 链已用时 %.1f min（%.2f h）" % ((time.time() - t0) / 60.0,
                                                (time.time() - t0) / 3600.0), flush=True)
        print("#" * 92, flush=True)

        st["started"] = datetime.now().isoformat(timespec="seconds")
        st["status"] = "running"
        atomic_write_json(a.progress, snapshot(steps, done, failed, skipped))

        cmd = [sys.executable, TRAINER, "--config", c, "--seed", str(s),
               "--root", a.root, "--epochs", str(a.epochs),
               "--batch_size", str(a.batch_size), "--lr", repr(a.lr),
               "--val_interval", str(a.val_interval), "--early_stop", str(a.early_stop)]
        ts = time.time()
        rc = subprocess.call(cmd)
        dt = (time.time() - ts) / 60.0
        ran += 1
        st["minutes"] = dt
        st["finished"] = datetime.now().isoformat(timespec="seconds")

        if rc == 0:
            st["status"] = "done"
            st["note"] = "ok"
            done.append((s, c))
            print("\n[P3-3-chain] ✓ seed=%d cfg=%s 完成（%.1f min；超/欠预算 %+.1f min）"
                  % (s, c, dt, dt - eta.get(c, 0.0)), flush=True)
        else:
            st["status"] = "failed"
            st["note"] = "退出码 %d" % rc
            failed.append((s, c, rc))
            # 继续下一个：整夜无人值守时，单格失败不该吃掉后面所有小时
            print("\n[P3-3-chain] ✗ seed=%d cfg=%s 失败（退出码 %d，%.1f min）—— 记为 failed，继续下一格"
                  % (s, c, rc, dt), flush=True)

        # 用实测覆盖该配置的 ETA（后续种子更准）
        eta[c] = dt
        atomic_write_json(a.progress, snapshot(steps, done, failed, skipped))

    el = (time.time() - t0) / 60.0
    print("\n" + "=" * 92, flush=True)
    print("[P3-3-chain] 结束 %s | 总耗时 %.1f min (%.2f h)"
          % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), el, el / 60.0), flush=True)
    print("[P3-3-chain] 完成 %d | 跳过 %d | 失败 %d | 未跑 %d"
          % (len(done), len(skipped), len(failed),
             len([s for s in steps if s["status"] == "pending"])), flush=True)
    if done:
        print("[P3-3-chain] done: " + ", ".join("s%d/%s" % (s, c) for s, c in done), flush=True)
    if skipped:
        print("[P3-3-chain] skip: " + ", ".join("s%d/%s" % (s, c) for s, c in skipped), flush=True)
    if failed:
        print("[P3-3-chain] FAIL: " + ", ".join("s%d/%s(rc=%d)" % t for t in failed), flush=True)
    pend = [s for s in steps if s["status"] == "pending"]
    if pend:
        print("[P3-3-chain] 未跑（可直接重跑本脚本续跑）: "
              + ", ".join("s%d/%s" % (s["seed"], s["config"]) for s in pend), flush=True)
    print("[P3-3-chain] 进度文件: " + a.progress, flush=True)
    print("[P3-3-chain] 下一步: PI 跑 p3_3_multiseed_eval.py（逐种子 ETIS 评测 + H4 判定）", flush=True)
    print("=" * 92, flush=True)

    # 退出码：有失败 ⇒ 非 0（让 longrun 状态能一眼看出），但已完成格不受影响
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
