# -*- coding: utf-8 -*-
"""
文件名: p3_3_freeze_facts.py
功能: 【论文二 P3-3】冻结事实计算器 —— 用**已有产物**推出"关键 5 格"是谁、以及真实预算。

为什么需要它
------------
总计划 P3-3 写"关键 5 格 × 3 种子"，但**任何冻结文档里都没有点名这 5 格是谁**。
（已对全仓 grep 确认。）若不先把"5 格"固定成**可复算的证据结论**，
P3-3 跑完后就有"事后挑格子"的空间 —— 违反 G6 与预注册原则。

本脚本的立场：
    关键格 = **参与 ID↔OOD 排名反转的那些配置**（反转是 H4 的对象）。
    这是从数据里推出来的，不是人挑的；任何人重跑本脚本都会得到同一份 5 格。

输入（只读，D 盘原位）
    experiments/ablation/<cfg>/results.json
        -> 分布内测试集 Dice（train_single.py 自产，协议权威值）+ training_time_min
    results_ablation_etis/ablation_etis_summary_full8.json
        -> 8 个配置的 ETIS 零样本 Dice（%）

输出
    03_results/stats/p3_3_freeze_facts.json     （机读，供规格文档与复核算引用）
    控制台可读摘要

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_3_freeze_facts.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

D_ROOT = "D:/medical_segmentation"
ABL_DIR = os.path.join(D_ROOT, "experiments", "ablation")
ETIS_SUMMARY = os.path.join(D_ROOT, "results_ablation_etis",
                            "ablation_etis_summary_full8.json")
OUT_JSON = ("E:/paper2_ablation_reliability/03_results/stats/"
            "p3_3_freeze_facts.json")

CFGS = ["baseline", "dpa_only", "egm_only", "msfa_only",
        "egm_dpa", "egm_msfa", "dpa_msfa", "egm_dpa_msfa"]


# --------------------------------------------------------------------------- #
def load_facts():
    """返回 {cfg: dict(id_dice, ood_dice, train_min, params_M, best_epoch)}。"""
    ood = {d["config"]: d for d in json.load(open(ETIS_SUMMARY, encoding="utf-8"))}
    out = {}
    for c in CFGS:
        rp = os.path.join(ABL_DIR, c, "results.json")
        if not os.path.exists(rp):
            raise FileNotFoundError(f"{c}: 缺 results.json ({rp})")
        r = json.load(open(rp, encoding="utf-8"))
        if ood.get(c) is None:
            raise KeyError(f"{c}: 缺 ETIS 汇总")
        out[c] = dict(
            id_dice=float(r["test_metrics"]["Dice"]),
            ood_dice=float(ood[c]["dice"]) / 100.0,      # 统一为分数（0-1）
            ood_dice_pct=float(ood[c]["dice"]),
            train_min=float(r["training_time_min"]),
            params_M=float(r.get("parameters_M", float("nan"))),
            best_epoch=int(r.get("best_epoch", -1)),
            ood_n=int(ood[c]["n"]),
            ood_std_pct=float(ood[c].get("dice_std", float("nan"))),
        )
    return out


def rank_desc(vals: dict) -> dict:
    """降序排名（1 = 最好）。先断言无并列，避免排名口径含糊。"""
    items = sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))
    seen = {}
    for i, (k, v) in enumerate(items, 1):
        if v in seen:
            raise ValueError(f"并列值 {v} 出现在 {seen[v]} 与 {k} —— 排名不唯一，须先定调")
        seen[v] = k
    return {k: i for i, (k, _) in enumerate(items, 1)}


def spearman(ra: dict, rb: dict, keys) -> float:
    """两个排名向量的 Spearman ρ（无并列时 = 1 - 6Σd²/(n(n²-1))）。"""
    n = len(keys)
    d2 = sum((ra[k] - rb[k]) ** 2 for k in keys)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def find_reversals(idr: dict, oodr: dict, keys):
    """反转对：ID 排 A>B 但 OOD 排 B>A（rank 越小越好）。"""
    out = []
    ks = sorted(keys)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            s_id = idr[a] - idr[b]          # <0 表示 ID 上 a 优于 b
            s_od = oodr[a] - oodr[b]
            if s_id * s_od < 0:
                out.append((a, b, s_id, s_od))
    return out


def main():
    f = load_facts()
    ks_all = list(CFGS)

    id_vals = {k: f[k]["id_dice"] for k in ks_all}
    ood_vals = {k: f[k]["ood_dice"] for k in ks_all}
    idr_all = rank_desc(id_vals)
    oodr_all = rank_desc(ood_vals)

    rev_all = find_reversals(idr_all, oodr_all, ks_all)
    key_set = sorted({c for a, b, _, _ in rev_all for c in (a, b)})
    non_key = [c for c in ks_all if c not in key_set]

    # ---- 在 5 格内部重算（排名口径 = 5 格内部互比）----
    idr_k = rank_desc({k: id_vals[k] for k in key_set})
    oodr_k = rank_desc({k: ood_vals[k] for k in key_set})
    rev_k = find_reversals(idr_k, oodr_k, key_set)
    rho_k = spearman(idr_k, oodr_k, key_set)

    # ---- 非关键格在两种排名下是否同序（用于说明"它们不可能反转"）----
    non_key_stable = all(
        (idr_all[a] - idr_all[b]) * (oodr_all[a] - oodr_all[b]) > 0
        for i, a in enumerate(non_key) for b in non_key[i + 1:]
    )

    # ---- 预算：便宜→贵；同步算"关掉了几对反转" ----
    order = sorted(key_set, key=lambda c: f[c]["train_min"])
    rows, cum = [], 0.0
    closed = set()
    for i, c in enumerate(order, 1):
        cum += f[c]["train_min"]
        for pi, (a, b, _, _) in enumerate(rev_k, 1):
            if a in order[:i] and b in order[:i]:
                closed.add(pi)
        rows.append(dict(step=i, config=c, train_min=f[c]["train_min"],
                         cum_min=cum, cum_h=cum / 60.0,
                         pairs_closed=sorted(closed),
                         n_pairs_closed=len(closed)))
    per_seed_min = cum
    per_seed_h = per_seed_min / 60.0

    payload = dict(
        segment="P3-3",
        kind="freeze_facts",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/p3_3_freeze_facts.py",
        sources=dict(id_dice=f"{ABL_DIR}/<cfg>/results.json::test_metrics.Dice",
                     ood_dice=ETIS_SUMMARY,
                     ood_n=f[CFGS[0]]["ood_n"]),
        all8={k: f[k] for k in ks_all},
        id_ranking_all8=idr_all,
        ood_ranking_all8=oodr_all,
        reversals_all8=[dict(a=a, b=b, sign_id="ID: a>b" if s < 0 else "ID: b>a",
                             id_rank_diff=s, ood_rank_diff=t)
                        for a, b, s, t in rev_all],
        key_set=key_set,
        key_set_size=len(key_set),
        non_key=non_key,
        non_key_never_reverses=bool(non_key_stable),
        id_ranking_key=idr_k,
        ood_ranking_key=oodr_k,
        reversals_key=[dict(a=a, b=b, id_rank_diff=s, ood_rank_diff=t,
                            id_val_diff=id_vals[a] - id_vals[b],
                            ood_val_diff=ood_vals[a] - ood_vals[b])
                       for a, b, s, t in rev_k],
        spearman_id_vs_ood_key=rho_k,
        budget=dict(order=order, rows=rows,
                    per_seed_min=per_seed_min, per_seed_h=per_seed_h,
                    n_new_seeds=2, total_min_2seeds=per_seed_min * 2,
                    total_h_2seeds=per_seed_min * 2 / 60.0),
        note=("关键格由『参与排名反转』这一客观准则推出，非人工挑选；"
              "非关键格在 ID/OOD 两种排名下同序，故不可能单独产生反转。"),
    )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---------------- 控制台 ----------------
    print("=" * 96)
    print("P3-3 冻结事实：8 配置 ID vs OOD")
    print("=" * 96)
    print(f"{'config':<14}{'ID(rank)':>16}{'OOD(rank)':>16}{'train_min':>11}{'params':>9}")
    for c in sorted(ks_all, key=lambda x: idr_all[x]):
        mark = "  <== 关键" if c in key_set else ""
        print(f"{c:<14}{id_vals[c]:>10.4f}({idr_all[c]}){ood_vals[c]:>10.4f}({oodr_all[c]})"
              f"{f[c]['train_min']:>11.1f}{f[c]['params_M']:>8.2f}M{mark}")
    print()
    print(f"反转对（全 8 格口径）: {len(rev_all)}")
    for a, b, s, t in rev_all:
        print(f"   ({a} , {b})   ID阶差={s:+d}  OOD阶差={t:+d}   "
              f"ID值差={id_vals[a]-id_vals[b]:+.4f}  OOD值差={ood_vals[a]-ood_vals[b]:+.4f}")
    print()
    print(f"★ 关键 {len(key_set)} 格 = {key_set}")
    print(f"  非关键 {len(non_key)} 格 = {non_key}（两种排名下同序: {non_key_stable}）")
    print(f"  5 格内部 ρ(ID,OOD) = {rho_k:+.4f}")
    print()
    print("预算（便宜→贵；每种子累计）:")
    for r in rows:
        print(f"  [{r['step']}] {r['config']:<14} +{r['train_min']:6.1f} min -> "
              f"累计 {r['cum_min']:7.1f} min ({r['cum_h']:5.2f} h) | "
              f"已关反转对 {r['n_pairs_closed']}/{len(rev_k)} {r['pairs_closed']}")
    print()
    print(f"每种子 {per_seed_min:.1f} min = {per_seed_h:.2f} h")
    print(f"2 个新种子 {per_seed_min*2:.0f} min = {per_seed_min*2/60:.1f} h "
          f"（种子 42 已有，直接复用）")
    print(f"已写 {OUT_JSON}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
