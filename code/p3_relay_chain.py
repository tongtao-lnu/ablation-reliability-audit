# -*- coding: utf-8 -*-
"""
p3_relay_chain.py -- 无人值守接力链：P3-2 收尾 → P3-4 训练 → P3-4 评测

为什么需要它
------------
P3-2 训练（已单独在跑）→ 跑 P3-2 的 ETIS 推理 → 写第 8 格结果；
再 → P3-4 串行训练 3 个容量匹配模型（6–11 h）→ P3-4 评测。
若每步都靠人工接力，段与段之间 GPU 会空转（等发现 → 再启下一条）。
本脚本把**机械步骤**串成一链，一次启动、中间无空档。

⚠️ 边界（诚实划界）
------------------
本链**只做机械步骤**：训练 / 推理 / 按冻结判据计算 / 落盘。
**判读仍然留给人**：分支 A/B 的结论表述、是否进正文、与 96.38% 差异怎么措辞 ——
链条只把**证据 + 已算好的判词**摆出来，不会自行下结论写进论文。

这是用户 2026-09-14 明确授权的**有界自动化**（"3-2 跑完你自己就继续 3-4"），
不是"全流程无人值守"（那仍不支持）。

链内步骤（顺序，失败即止）
------------------------
  0. 等 P3-2 训练结束（轮询 status.json；含**停滞检测**，防 status 卡在 running）
  1. `p3_2_etis_full_module.py`            → 第 8 格 ETIS 预测 + 4 道门禁
  2. `p3_4_chain_train.py --bfs 72 79 71`  → 3 个容量匹配模型串行训练（6–11 h）
  3. `p3_4_etis_capacity.py`               → 逐对评测 + 分支 A/B 直答

跑法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_relay -- $PY E:/paper2_ablation_reliability/02_code/analysis/p3_relay_chain.py

断链后恢复：链条对每步幂等（已存在的产物跳过），重跑本脚本即可从未完成处续上。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E_ROOT = "E:/paper2_ablation_reliability"
LOG_ROOT = E_ROOT + "/03_results/logs"
PROGRESS = E_ROOT + "/03_results/raw/p3_relay_progress.json"

P32_NAME = "p3_2_train_egm_dpa_msfa"
P32_STATUS = os.path.join(LOG_ROOT, P32_NAME + ".status.json")
P32_LOG = os.path.join(LOG_ROOT, P32_NAME + ".log")
P32_CKPT = "D:/medical_segmentation/experiments/ablation/egm_dpa_msfa/best_model.pth"
P32_RESULTS = "D:/medical_segmentation/experiments/ablation/egm_dpa_msfa/results.json"

STEPS = [
    ("p3_2_etis", "p3_2_etis_full_module.py", ["p3_2_etis_full_module.py"]),
    ("p3_4_train", "p3_4_chain_train.py", ["p3_4_chain_train.py", "--bfs", "72", "79", "71"]),
    ("p3_4_etis", "p3_4_etis_capacity.py", ["p3_4_etis_capacity.py"]),
]

STALL_MIN = 12          # status 卡 running 且日志 12 min 不增长 ⇒ 判停滞


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_progress(state, step=None, note=None, extra=None):
    d = dict(chain="p3_relay", state=state, current_step=step, note=note,
             updated_at=now(), steps=[])
    if os.path.exists(PROGRESS):
        try:
            with open(PROGRESS, encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict) and isinstance(old.get("steps"), list):
                d["steps"] = old["steps"]
        except Exception:                                            # noqa: BLE001
            pass
    if extra:
        d.update(extra)
    tmp = PROGRESS + ".tmp"
    try:
        os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        os.replace(tmp, PROGRESS)
    except OSError:
        pass
    return d


def mark_step(name, status, minutes=None):
    """把某步的结果并入 progress.steps（去重按 name）。"""
    d = {}
    if os.path.exists(PROGRESS):
        try:
            with open(PROGRESS, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:                                            # noqa: BLE001
            d = {}
    steps = [s for s in d.get("steps", []) if s.get("step") != name]
    steps.append(dict(step=name, status=status, minutes=minutes, at=now()))
    d["steps"] = steps
    tmp = PROGRESS + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        os.replace(tmp, PROGRESS)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 步骤 0：等 P3-2 训练结束
# --------------------------------------------------------------------------- #
def _p32_progress():
    """从训练日志里抓最后一次 'Epoch N/M'，用于心跳展示（只读，失败即静默）。"""
    import re
    try:
        with open(P32_LOG, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-400000:]
        m = re.findall(r"Epoch\s+(\d+)/(\d+)\]", tail)
        if m:
            return f"{m[-1][0]}/{m[-1][1]}"
        m2 = re.findall(r"Training:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", tail)
        if m2:
            return f"batch {m2[-1][0]}/{m2[-1][1]}"
    except Exception:                                                # noqa: BLE001
        pass
    return "?"


def wait_p3_2(timeout_h=6.0):
    print(f"[relay] 等待 {P32_NAME} 训练结束（上限 {timeout_h} h）…", flush=True)
    print("[relay] 注意：等待期本进程以 60 s 轮询、每 5 min 打一次心跳；"
          "日志长时间静默属正常（非停滞）。", flush=True)
    t0 = time.time()
    last_bytes, last_change = 0, time.time()
    last_beat = 0.0
    while True:
        el = time.time() - t0
        if el > timeout_h * 3600:
            print(f"[relay] ✗ 等待超时（{timeout_h} h）", flush=True)
            return "timeout"

        st = None
        if os.path.exists(P32_STATUS):
            try:
                with open(P32_STATUS, encoding="utf-8") as f:
                    st = json.load(f)
            except Exception:                                        # noqa: BLE001
                st = None

        if st:
            s = st.get("state")
            if s == "done":
                print(f"[relay] ✓ P3-2 训练完成（exit={st.get('exit_code')}，"
                      f"{st.get('elapsed_s', 0) / 60:.1f} min）", flush=True)
                return "done"
            if s == "failed":
                print(f"[relay] ✗ P3-2 训练失败（exit={st.get('exit_code')}）", flush=True)
                return "failed"

        # 停滞检测（防 status 永远卡 running）
        try:
            nb = os.path.getsize(P32_LOG) if os.path.exists(P32_LOG) else 0
        except OSError:
            nb = last_bytes
        if nb != last_bytes:
            last_bytes, last_change = nb, time.time()
        elif (time.time() - last_change) > STALL_MIN * 60:
            print(f"[relay] ✗ 停滞：日志 {STALL_MIN} min 无增长，判 P3-2 已死", flush=True)
            return "stalled"

        # 存活信号 A（独立于 stdout）：每次轮询刷新 progress json。
        # 即使 longrun 输出被断、stdout 丢失，也能用 updated_at 判活。
        pg = _p32_progress()
        write_progress("running", step="wait_p3_2",
                       extra={"p32_epoch": pg, "waited_min": round(el / 60, 1)})

        # 存活信号 B：每 5 min 一行心跳（含训练当前 epoch）
        if time.time() - last_beat >= 300:
            last_beat = time.time()
            print(f"[relay] 心跳 {now()} | 已等 {el / 60:.1f} min | "
                  f"P3-2 进度 {pg} | 日志 {nb / 1e6:.1f} MB", flush=True)

        time.sleep(60)


def run_step(name, argv, timeout_h=None):
    cmd = [sys.executable] + [os.path.join(HERE, argv[0])] + argv[1:]
    print("\n" + "#" * 84, flush=True)
    print(f"### [relay] 步骤 {name}: {' '.join(cmd)}", flush=True)
    print("#" * 84, flush=True)
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, timeout=(timeout_h * 3600 if timeout_h else None))
    except subprocess.TimeoutExpired:
        print(f"[relay] ✗ {name} 超时", flush=True)
        mark_step(name, "timeout", round((time.time() - t0) / 60, 1))
        return 2
    mins = round((time.time() - t0) / 60, 1)
    mark_step(name, "done" if rc == 0 else f"failed(rc={rc})", mins)
    print(f"[relay] {'✓' if rc == 0 else '✗'} {name} 结束 rc={rc}，{mins} min", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--wait-timeout-h", type=float, default=6.0)
    ap.add_argument("--skip-wait", action="store_true",
                    help="不等 P3-2（P3-2 已完成或不在本次范围）")
    a = ap.parse_args()

    t_all = time.time()
    write_progress("running", step="wait_p3_2")
    print("=" * 84)
    print(f"[relay] 启动 {now()}")
    print(f"[relay] 链：P3-2 训练等待 → P3-2 ETIS → P3-4 训练(72/79/71) → P3-4 评测")
    print("=" * 84)

    if not a.skip_wait:
        r = wait_p3_2(a.wait_timeout_h)
        if r != "done":
            write_progress("aborted", step="wait_p3_2", note=f"P3-2 未完成：{r}")
            print(f"[relay] 中止（P3-2 状态={r}）—— 未启动后续步骤", flush=True)
            return 1
    # 产物存在性门禁
    miss = [p for p in (P32_CKPT, P32_RESULTS) if not os.path.exists(p)]
    if miss:
        write_progress("aborted", step="wait_p3_2", note=f"P3-2 产物缺失：{miss}")
        print(f"[relay] 中止：P3-2 产物缺失 {miss}", flush=True)
        return 1
    mark_step("wait_p3_2", "done", None)
    print(f"[relay] P3-2 产物齐备：\n  {P32_CKPT}\n  {P32_RESULTS}", flush=True)

    for name, _mod, argv in STEPS:
        write_progress("running", step=name)
        rc = run_step(name, argv)
        if rc != 0:
            write_progress("aborted", step=name, note=f"{name} rc={rc}")
            print(f"[relay] ✗ 链在 {name} 中止（失败即止，已完成的产物保留）", flush=True)
            return rc

    total_h = (time.time() - t_all) / 3600
    write_progress("done", step=None,
                   note=f"全链完成，用时 {total_h:.2f} h")
    print("\n" + "=" * 84)
    print(f"[relay] ✓ 全链完成，用时 {total_h:.2f} h")
    print(f"[relay] 进度留痕 {PROGRESS}")
    print("[relay] 产物：")
    print("  03_results/raw/full_module/            （P3-2 第 8 格 + 门禁）")
    print("  03_results/raw/capacity_matched/       （P3-4 逐模型）")
    print("  03_results/stats/capacity_matched.json （P3-4 汇总 + 分支判定）")
    print("  03_results/tables/T_p3_4_capacity.md   （P3-4 直答表）")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
