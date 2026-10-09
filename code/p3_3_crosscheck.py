# -*- coding: utf-8 -*-
"""
文件名: p3_3_crosscheck.py
功能: 【论文二 P3-3 · G4 独立复核】不 import 主评测器，独立重算 P3-3 的全部关键数字。

独立到什么程度
--------------
1. **推理实现独立**：主评测器逐样本（batch=1）在 torch 张量上算布尔交并；
   本脚本**批量化**（batch=8）、把 logits 落到 **numpy float32** 再算交集。
   两条代码路径若都给同一逐样本 Dice，则 Dice 本身不是实现产物。
2. **名次算法独立**：主脚本用"排序后取位置"；本脚本用**逐对计数**（数有几个比自己大）。
3. **反转判定独立**：主脚本比较 rank 差；本脚本**直接比较数值差**的符号。
4. **ID 侧独立**：重新打开 `results.json` 读一次（不复用主脚本已解析的 dict）。
5. **H4 判定独立**：重新实现主/副判据与分侧失败模式。

逐项比对主产物 `03_results/stats/multiseed.json`，全 MATCH 才算过。

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_3_crosscheck.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

D_ROOT = "D:/medical_segmentation"
E_ROOT = "E:/paper2_ablation_reliability"
sys.path.insert(0, D_ROOT)
os.chdir(D_ROOT)

from cross_dataset.dataset import CrossDataset                       # noqa: E402
from models.ablation_models import get_ablation_model                # noqa: E402

ETIS_DIR = "data_zeroshot/etis"
ABL_DIR = os.path.join(D_ROOT, "experiments", "ablation")
SEED_ROOT = os.path.join(E_ROOT, "03_results", "raw", "multiseed", "exp")
MAIN_JSON = os.path.join(E_ROOT, "03_results", "stats", "multiseed.json")
OUT_JSON = os.path.join(E_ROOT, "03_results", "stats", "p3_3_crosscheck.json")
ETIS_SUMMARY = os.path.join(D_ROOT, "results_ablation_etis",
                            "ablation_etis_summary_full8.json")

THRESH = 0.5
EPS = 1e-6
TIE = 1e-6
FLAGSHIP = ("egm_dpa", "msfa_only")
TOL_DICE = 1e-9          # 两条实现路径的逐样本 Dice 容许差
TOL_AGG = 1e-12          # 聚合量容许差


def wpath(seed, cfg):
    if seed == 42:
        return os.path.join(ABL_DIR, cfg, "best_model.pth")
    return os.path.join(SEED_ROOT, str(seed), "ablation", cfg, "best_model.pth")


def rpath(seed, cfg):
    if seed == 42:
        return os.path.join(ABL_DIR, cfg, "results.json")
    return os.path.join(SEED_ROOT, str(seed), "ablation", cfg, "results.json")


@torch.no_grad()
def infer_probs(model, dataset, device, batch=8):
    """**批量**推理，返回 [(name, prob (H,W) float32)]。故意与主脚本(batch=1)不同。"""
    out = []
    n = len(dataset)
    idx = tqdm(range(0, n, batch), desc="  crosscheck", ncols=88, leave=False)
    for i in idx:
        items = [dataset[j] for j in range(i, min(i + batch, n))]
        x = torch.stack([it[0] for it in items]).to(device)
        with autocast():
            o = model(x)
            if isinstance(o, tuple):
                o = o[0]
            if isinstance(o, tuple):
                o = o[0]
        p = torch.sigmoid(o.float()).cpu().numpy()          # (B,H,W)
        for k, it in enumerate(items):
            out.append((str(it[2]), p[k].astype(np.float32)))
    return out


def load_gt(dataset):
    gts = []
    for j in range(len(dataset)):
        m = dataset[j][1].numpy()
        gts.append(m[0] if m.ndim == 3 else m)
    return gts


def build(cfg, path, device):
    model, acfg = get_ablation_model(cfg, in_channels=3, num_classes=1, base_filters=64)
    ck = torch.load(path, map_location=device)
    st = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(st, strict=True)
    return model.to(device).eval(), acfg


def rank_by_counting(vals, keys):
    """独立的名次算法：名次 = 1 + (严格大于自己的个数)。"""
    ks = sorted(keys)
    return {k: 1 + sum(1 for j in ks if vals[j] > vals[k] + 0.0) for k in ks}


def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-3 independent crosscheck (G4)")
    ap.add_argument("--main", default=MAIN_JSON)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)

    if not os.path.exists(a.main):
        print("✗ 主产物不存在: " + a.main)
        return 3
    M = json.load(open(a.main, encoding="utf-8"))
    seeds, configs = M["seeds"], M["configs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CrossDataset(ETIS_DIR, augment=False, strong_augment=False)
    gts = load_gt(dataset)
    print("ETIS %d 样本 | 复核种子 %s | 配置 %s" % (len(dataset), seeds, configs))

    checks, problems = [], []

    def chk(name, ok, detail=""):
        checks.append(dict(check=name, status="MATCH" if ok else "MISMATCH", detail=detail))
        if not ok:
            problems.append(name + " :: " + detail)

    mine = {}
    for s in seeds:
        mine[s] = {}
        for c in configs:
            model, acfg = build(c, wpath(s, c), device)
            pr = infer_probs(model, dataset, device, a.batch)
            del model
            torch.cuda.empty_cache()

            d_list = []
            for k, (nm, prob) in enumerate(pr):
                pb = (prob > THRESH)
                gb = (gts[k] > THRESH)
                inter = float(np.logical_and(pb, gb).sum())
                ps, gs = float(pb.sum()), float(gb.sum())
                d_list.append((2.0 * inter + EPS) / (ps + gs + EPS))
            d_arr = np.asarray(d_list, dtype=np.float64)
            rj = json.load(open(rpath(s, c), encoding="utf-8"))

            mine[s][c] = dict(
                mean_dice=float(d_arr.mean()),
                id_dice=float(rj["test_metrics"]["Dice"]),
                n=int(d_arr.size),
                dice=d_arr,
                per_sample_max_abs_diff=None,
            )
            # 逐样本层面对照主脚本缓存
            mc = M["per_config_raw"][str(s)][c]
            chk("agg_mean_dice seed=%d cfg=%s" % (s, c),
                abs(mine[s][c]["mean_dice"] - mc["mean_dice"]) <= TOL_AGG,
                "mine=%.12f main=%.12f" % (mine[s][c]["mean_dice"], mc["mean_dice"]))
            chk("agg_id_dice seed=%d cfg=%s" % (s, c),
                abs(mine[s][c]["id_dice"] - mc["id_dice"]) <= TOL_AGG,
                "mine=%.12f main=%.12f" % (mine[s][c]["id_dice"], mc["id_dice"]))
            print("  seed=%-5d cfg=%-14s OOD=%.8f%% ID=%.8f"
                  % (s, c, mine[s][c]["mean_dice"] * 100, mine[s][c]["id_dice"]))

    # 逐样本 Dice 对照（读主脚本缓存的逐样本数组）
    cache_path = os.path.join(E_ROOT, "03_results", "raw", "multiseed", "per_sample_dice.json")
    if os.path.exists(cache_path):
        C = json.load(open(cache_path, encoding="utf-8"))
        worst, wn = 0.0, ""
        for s in seeds:
            for c in configs:
                cd = C.get("%d|%s" % (s, c), {}).get("dice")
                if not cd or len(cd) != len(mine[s][c]["dice"]):
                    chk("per_sample_len seed=%d cfg=%s" % (s, c), False,
                        "缓存缺失或长度不符")
                    continue
                dd = float(np.max(np.abs(np.asarray(cd) - mine[s][c]["dice"])))
                if dd > worst:
                    worst, wn = dd, "seed=%d cfg=%s" % (s, c)
        chk("per_sample_dice_max_abs_diff", worst <= TOL_DICE,
            "max|Δ|=%.3e @ %s (tol %.0e)" % (worst, wn, TOL_DICE))

    # ---- 名次 / 反转 / H4 独立重算 ----
    per = {}
    for s in seeds:
        idv = {c: mine[s][c]["id_dice"] for c in configs}
        oodv = {c: mine[s][c]["mean_dice"] for c in configs}
        rid = rank_by_counting(idv, configs)
        rood = rank_by_counting(oodv, configs)
        rev = []
        ks = sorted(configs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                x, y = ks[i], ks[j]
                dx, dy = idv[x] - idv[y], oodv[x] - oodv[y]
                if abs(dx) <= TIE or abs(dy) <= TIE:
                    continue
                if (dx > 0) != (dy > 0):
                    rev.append((x, y))
        a_, b_ = FLAGSHIP
        d_id, d_od = idv[a_] - idv[b_], oodv[a_] - oodv[b_]
        flag_rev = (abs(d_id) > TIE and abs(d_od) > TIE and (d_id > 0) != (d_od > 0))
        per[s] = dict(id_rank=rid, ood_rank=rood, rev=rev, flag_rev=bool(flag_rev),
                      d_id=d_id, d_od=d_od)
        # 与主产物对照
        chk("id_rank seed=%d" % s, rid == M["per_seed"][str(s)]["id_rank"],
            "mine=%s main=%s" % (rid, M["per_seed"][str(s)]["id_rank"]))
        chk("ood_rank seed=%d" % s, rood == M["per_seed"][str(s)]["ood_rank"],
            "mine=%s main=%s" % (rood, M["per_seed"][str(s)]["ood_rank"]))
        mr = sorted((r["a"], r["b"]) for r in M["per_seed"][str(s)]["reversals"])
        chk("reversal_pairs seed=%d" % s, sorted(rev) == mr,
            "mine=%s main=%s" % (sorted(rev), mr))
        chk("flagship_reversed seed=%d" % s,
            flag_rev == bool(M["per_seed"][str(s)]["flagship"]["reversed"]),
            "mine=%s main=%s" % (flag_rev, M["per_seed"][str(s)]["flagship"]["reversed"]))

    main_ok = all(per[s]["flag_rev"] for s in seeds)
    pk = set()
    for s in seeds:
        pk |= set(per[s]["rev"])
    cons = []
    for (x, y) in sorted(pk):
        okall = True
        for s in seeds:
            dx = mine[s][x]["id_dice"] - mine[s][y]["id_dice"]
            dy = mine[s][x]["mean_dice"] - mine[s][y]["mean_dice"]
            if not (abs(dx) > TIE and abs(dy) > TIE and (dx > 0) != (dy > 0)):
                okall = False
        if okall:
            cons.append((x, y))
    sec_ok = len(cons) >= 3

    chk("h4_main_criterion", main_ok == bool(M["h4"]["main_criterion_pass"]),
        "mine=%s main=%s" % (main_ok, M["h4"]["main_criterion_pass"]))
    chk("h4_secondary_n", len(cons) == int(M["h4"]["secondary_n"]),
        "mine=%d main=%d" % (len(cons), M["h4"]["secondary_n"]))
    chk("h4_secondary_pass", sec_ok == bool(M["h4"]["secondary_criterion_pass"]),
        "mine=%s main=%s" % (sec_ok, M["h4"]["secondary_criterion_pass"]))
    verd = ("H4 支持（保留“致命反转”卖点）" if (main_ok and sec_ok)
            else "H4 被推翻（走分支 C：只留 L1 恒等式 + L3 协议）")
    chk("h4_verdict", verd == M["h4"]["verdict"], "mine=%s main=%s" % (verd, M["h4"]["verdict"]))

    # G2 复现闸的独立复算（种子 42）
    if 42 in seeds:
        ref = {d["config"]: d["dice"] for d in json.load(open(ETIS_SUMMARY, encoding="utf-8"))}
        worst, wn = 0.0, ""
        for c in configs:
            dd = abs(mine[42][c]["mean_dice"] * 100 - ref[c])
            if dd > worst:
                worst, wn = dd, c
        chk("gate_G2_reproduce_seed42", worst <= 1e-6,
            "max|Δ|=%.3e @ %s" % (worst, wn))

    payload = dict(
        segment="P3-3", kind="crosscheck_G4",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/p3_3_crosscheck.py",
        main_artifact=a.main,
        independence=["批量(batch=%d) numpy 路径 vs 主脚本逐样本 torch 路径" % a.batch,
                      "名次用逐对计数 vs 主脚本排序取位",
                      "反转用数值差符号 vs 主脚本名次差符号",
                      "ID 侧重新读盘 vs 主脚本已解析 dict",
                      "H4 判定独立重实现"],
        checks=checks,
        n_checks=len(checks), n_mismatch=len(problems),
        result="MATCH" if not problems else "MISMATCH",
        problems=problems,
        recomputed=dict(
            per_seed={str(s): dict(id_rank=per[s]["id_rank"], ood_rank=per[s]["ood_rank"],
                                   reversal_pairs=[list(t) for t in per[s]["rev"]],
                                   flagship_reversed=per[s]["flag_rev"],
                                   flagship_d_id=per[s]["d_id"],
                                   flagship_d_ood=per[s]["d_od"]) for s in seeds},
            h4=dict(main_pass=bool(main_ok), consistent_pairs=[list(t) for t in cons],
                    secondary_n=len(cons), secondary_pass=bool(sec_ok), verdict=verd),
        ),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("P3-3 独立复核 (G4)：%d 项检查，MISMATCH %d 项" % (len(checks), len(problems)))
    bad = [c for c in checks if c["status"] != "MATCH"]
    for c in bad:
        print("  ✗ %s | %s" % (c["check"], c["detail"]))
    print("⇒ " + payload["result"])
    print("已写 " + a.out)
    print("=" * 88)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
