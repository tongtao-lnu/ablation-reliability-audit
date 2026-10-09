# -*- coding: utf-8 -*-
"""
文件名: p3_3_multiseed_eval.py
功能: 【论文二 P3-3】多种子 ETIS 评测 + H4（反转是否复现）判定。

做三件事
--------
1. **复现闸 G2**：在**种子 42** 的既有权重上跑本评测器，须复现
   `results_ablation_etis/ablation_etis_summary_full8.json` 的 OOD Dice（tol 1e-6）。
   不过 ⇒ 评测器协议与既有产物不同 ⇒ 全部读数作废，直接 STOP。
2. 对**每个 (种子, 配置)** 跑 ETIS 196 样本推理，逐样本 Dice 均值（eps=1e-6，
   与 `cross_dataset/train_cross.py::calculate_metrics` 逐字一致）。
3. 逐种子算 ID 名次 / OOD 名次 / **反转对**，按 `00_docs/P3-3规格冻结_2026-09-15.md` §四
   判 H4，并**分侧报告失败模式**（ID 侧翻转 / OOD 侧翻转）。

为什么 ID 侧不重算
------------------
ID Dice **直接读**该次训练的 `results.json::test_metrics.Dice`（242 样本，`utils/metrics.py`，
`eps=1e-8`）。重算会引入第二套口径，且没有必要。

可续跑
------
每个 (种子, 配置) 的逐样本 Dice 缓存进 `--cache`，已缓存则跳过推理。
故本脚本可反复跑：只补新增的格子。

用法
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_3_multiseed_eval.py --seeds 42 123 2024
    $PY ... --gate-only           # 只跑复现闸（跑训练前先验证评测器）
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

from cross_dataset.dataset import CrossDataset                      # noqa: E402
from models.ablation_models import get_ablation_model               # noqa: E402

# 修复(2026-09-20): 第 8 格 `egm_dpa_msfa` **不在** D 盘
#   `models/ablation_models.py::ABLATION_CONFIGS`（该表只有 7 项），训练链是靠
#   **运行时注入**进去的（见 p3_3_train_seed.py 文件头 ②）。
#   原评测脚本不知道这件事 ⇒ 跑到 egm_dpa_msfa 就
#   `ValueError: 未知配置: egm_dpa_msfa`，评测从未跑完。
#   此处**复用训练链的同一份注入函数**，绝不重复定义 —— 单一真源；
#   且 inject_eighth() 自带参数量自检（48.4154 M ± 0.001），
#   若 D 盘源码被改动会**先失败**，而不是安静地训/评一个结构不同的模型。
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)
from p3_3_train_seed import EIGHTH_CONFIG, inject_eighth             # noqa: E402

ETIS_DIR = "data_zeroshot/etis"
ABL_DIR = os.path.join(D_ROOT, "experiments", "ablation")           # 种子 42 原位（只读）
ETIS_SUMMARY = os.path.join(D_ROOT, "results_ablation_etis",
                            "ablation_etis_summary_full8.json")
SEED_ROOT = os.path.join(E_ROOT, "03_results", "raw", "multiseed", "exp")
OUT_JSON = os.path.join(E_ROOT, "03_results", "stats", "multiseed.json")
OUT_TABLE = os.path.join(E_ROOT, "03_results", "tables", "T_multiseed.md")
CACHE_JSON = os.path.join(E_ROOT, "03_results", "raw", "multiseed", "per_sample_dice.json")

PIXEL_THRESH = 0.5
EPS_OOD = 1e-6          # cross_dataset/train_cross.py::calculate_metrics
TIE_TOL = 1e-6          # 平局阈值（规格 §4.4）
GATE_TOL = 1e-6         # 复现闸容差（规格 §7.1）

# 关键 5 格（冻结，见 00_docs/P3-3规格冻结_2026-09-15.md §2.3）
KEY_CONFIGS = ["msfa_only", "dpa_msfa", "egm_msfa", "egm_dpa_msfa", "egm_dpa"]
# 旗舰反转对（冻结，主判据）
FLAGSHIP = ("egm_dpa", "msfa_only")
# 旗舰反转的**冻结参考方向**（规格 §4.5③，用户 2026-09-16 01:05 裁定）：
#   种子 42 上 Δ_ID < 0（egm_dpa 域内更差）而 Δ_OOD > 0（egm_dpa 域外更好）
#   ⇒ 方向元组 (sign(Δ_ID), sign(Δ_OOD)) = (-1, +1)。
#   主判据要求 3/3 种子的反转方向**与之一致** —— "复现"一词天然要求方向一致；
#   若某种子两侧仍相反但方向颠倒（(+1,-1)），§1 的"符号翻转"判推翻 ⇒ 必须同判。
# 修复(2026-09-20): 原实现只判了 §4.2 的 ①②（逐种子两侧符号相反 + |Δ|>1e-6），
#   漏了 §4.5③，且未输出该节明文要求的四列 ⇒ 已补齐（属实现缺陷修补，
#   判据本身 2026-09-16 已冻结、早于任何多种子 OOD 数据的产生，非"事后改判据"）。
REF_DIR = (-1, +1)
# ★ 副判据的**可计数**反转对（规格 §6.3）：#4 因跨协议混杂被排除
#   （第 8 格 egm_dpa_msfa 的种子 42 权重是 batch 4，其余 4 格是 batch 8）
COUNTABLE = [("egm_dpa", "msfa_only"), ("dpa_msfa", "egm_dpa"), ("dpa_msfa", "msfa_only")]
EXCLUDED = [("egm_dpa_msfa", "egm_msfa")]        # 跨协议混杂，仅参考
SEC_MIN = 2                                       # ≥2 / 3（规格 §4.2）
# 预期复现值（种子 42），仅用于 G2 闸
SEED42_EXPECT = {"msfa_only": 54.923838456478634,
                 "dpa_msfa": 56.839780358810785,
                 "egm_msfa": 66.72612952678554,
                 "egm_dpa_msfa": 65.5036571034353,
                 "egm_dpa": 63.913617849793305}


# --------------------------------------------------------------------------- #
def weight_path(seed: int, cfg: str) -> str:
    if seed == 42:
        return os.path.join(ABL_DIR, cfg, "best_model.pth")
    return os.path.join(SEED_ROOT, str(seed), "ablation", cfg, "best_model.pth")


def results_path(seed: int, cfg: str) -> str:
    if seed == 42:
        return os.path.join(ABL_DIR, cfg, "results.json")
    return os.path.join(SEED_ROOT, str(seed), "ablation", cfg, "results.json")


def build_and_load(cfg: str, wpath: str, device):
    """构造 + 载权重 + 硬拦键不匹配。

    ⚠️ 用 `strict=True`（默认）而**不是** `strict=False`：键不匹配时若静默通过，
    模型会带着**随机初始化的那部分参数**跑推理，产出"看起来正常"的全零/噪声预测，
    而且不会报错。论文一/二的既有铁律：键不匹配必须硬拦。
    """
    model, acfg = get_ablation_model(cfg, in_channels=3, num_classes=1, base_filters=64)
    ckpt = torch.load(wpath, map_location=device)
    state = (ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt
             else ckpt)
    model.load_state_dict(state, strict=True)     # 不匹配即抛异常 ⇒ 硬拦
    return model.to(device).eval(), acfg


@torch.no_grad()
def eval_etis(model, use_egm: bool, dataset, device) -> dict:
    """返回 {'names': [...], 'dice': [...], 'mean_dice': float}。逐样本 eps=1e-6。"""
    ds, gs, ps = [], [], []
    names = []
    for i in tqdm(range(len(dataset)), desc="  ETIS", ncols=88, leave=False):
        img, mask, name = dataset[i]
        xt = img.unsqueeze(0).to(device)
        with autocast():
            out = model(xt)
            seg = out[0] if (use_egm and isinstance(out, tuple)) else out
            if isinstance(seg, tuple):
                seg = seg[0]
        pred = (torch.sigmoid(seg.float()) > PIXEL_THRESH)
        # 修复(2026-09-20): 原写法漏了 .to(device) ⇒ `pred & gt` 抛
        #   RuntimeError: Expected all tensors to be on the same device (cuda:0 and cpu)
        #   种子 42 第一格即崩，评测从未产出过。
        # 等价性证明（对齐 ablation_zeroshot_etis.py::run_one → train_cross.py::calculate_metrics）:
        #   参考实现 pred=(sigmoid>0.5).float(); inter=(pred*target).sum();
        #             dice=(2*inter+1e-6)/(pred.sum()+target.sum()+1e-6)
        #   本脚本        pred=(sigmoid>0.5) bool; inter=(pred&gt).sum()
        #   CrossDataset(augment=False) 不做任何 transform ⇒ mask 为原样 npy，
        #   实测取值严格 ∈ {0.0,1.0}（float32）⇒ target 无需阈值化，
        #   (pred*target).sum() ≡ (pred&gt).sum()，target.sum() ≡ gt.sum()，
        #   eps 同为 1e-6 ⇒ 两式逐位等价。仅补 device，不改协议。
        gt = (mask.unsqueeze(0).to(device) > PIXEL_THRESH)
        inter = float((pred & gt).sum())
        ds.append((2.0 * inter + EPS_OOD) / (float(pred.sum()) + float(gt.sum()) + EPS_OOD))
        names.append(str(name))
    d = np.asarray(ds, dtype=float)
    return dict(names=names, dice=d.tolist(), mean_dice=float(d.mean()),
                n=int(d.size), std=float(d.std()))


def rank_desc(vals: dict) -> dict:
    """降序名次（1=最好）。值并列 ⇒ 名次不唯一，返回 None 标记由调用者报平局。"""
    items = sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))
    return {k: i for i, (k, _) in enumerate(items, 1)}


def reversal_pairs(idv: dict, oodv: dict, keys):
    """反转对 + 平局对（规格 §1.1 定义）。"""
    rev, tie = [], []
    ks = sorted(keys)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            d_id, d_od = idv[a] - idv[b], oodv[a] - oodv[b]
            if abs(d_id) <= TIE_TOL or abs(d_od) <= TIE_TOL:
                tie.append(dict(a=a, b=b, d_id=d_id, d_ood=d_od,
                                reason="ID 侧平局" if abs(d_id) <= TIE_TOL else "OOD 侧平局"))
                continue
            if (d_id > 0) != (d_od > 0):
                rev.append(dict(a=a, b=b, d_id=d_id, d_ood=d_od,
                                winner_id=a if d_id > 0 else b,
                                winner_ood=a if d_od > 0 else b,
                                amp=abs(d_od) / abs(d_id)))
    return rev, tie


def pair_sign(idv, oodv, pair):
    a, b = pair
    d_id, d_od = idv[a] - idv[b], oodv[a] - oodv[b]
    return dict(a=a, b=b, d_id=d_id, d_ood=d_od,
                id_tie=abs(d_id) <= TIE_TOL, ood_tie=abs(d_od) <= TIE_TOL,
                reversed=((d_id > 0) != (d_od > 0)) if (abs(d_id) > TIE_TOL and abs(d_od) > TIE_TOL) else None,
                winner_id=(a if d_id > 0 else b), winner_ood=(a if d_od > 0 else b))


def dir_of(pv):
    """旗舰反转的**方向元组** `(sign(Δ_ID), sign(Δ_OOD))`；任一侧平局 ⇒ 该位取 0。

    规格 §4.5③ 用它与 REF_DIR 比较判定"方向是否一致"。
    """
    sx = 0 if pv["id_tie"] else (1 if pv["d_id"] > 0 else -1)
    sy = 0 if pv["ood_tie"] else (1 if pv["d_ood"] > 0 else -1)
    return (sx, sy)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="P3-3 multiseed ETIS eval + H4 verdict")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--configs", nargs="+", default=KEY_CONFIGS)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--table", default=OUT_TABLE)
    ap.add_argument("--cache", default=CACHE_JSON)
    ap.add_argument("--gate-only", action="store_true",
                    help="只跑种子 42 的复现闸（G2），不写主产物")
    a = ap.parse_args(argv)

    # 必须早于任何 build_and_load：把第 8 格注入 ABLATION_CONFIGS
    if EIGHTH_CONFIG in a.configs:
        inject_eighth(verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 92)
    print("P3-3 多种子 ETIS 评测 | 设备 %s" % device)
    print("=" * 92)

    dataset = CrossDataset(ETIS_DIR, augment=False, strong_augment=False)
    if len(dataset) != 196:
        print("✗ ETIS 样本数 %d != 196 —— STOP" % len(dataset))
        return 3

    # ---------------- 权重存在性（G0 的一部分） ----------------
    missing = []
    for s in a.seeds:
        for c in a.configs:
            w, r = weight_path(s, c), results_path(s, c)
            if not (os.path.exists(w) and os.path.getsize(w) > 0):
                missing.append("缺失权重 seed=%d cfg=%s -> %s" % (s, c, w))
            if not os.path.exists(r):
                missing.append("缺失结果 seed=%d cfg=%s -> %s" % (s, c, r))
    if missing:
        print("\n".join("  ✗ " + m for m in missing))
        print("\n⇒ 有格子未训练完。训练链可续跑：p3_3_chain_train.py（幂等，自动跳过已完成格）")
        return 4

    cache = json.load(open(a.cache, encoding="utf-8")) if os.path.exists(a.cache) else {}

    # ---------------- 逐 (种子,配置) 推理 ----------------
    per = {}
    for s in a.seeds:
        per[s] = {}
        for c in a.configs:
            ck = "%d|%s" % (s, c)
            if ck in cache:
                per[s][c] = cache[ck]
                print("[cache] seed=%-5d cfg=%-14s OOD=%.6f%%" % (s, c, cache[ck]["mean_dice"] * 100))
                continue
            model, acfg = build_and_load(c, weight_path(s, c), device)
            use_egm = bool(acfg["use_egm"])
            print("[run  ] seed=%-5d cfg=%-14s ..." % (s, c))
            r = eval_etis(model, use_egm, dataset, device)
            r["config"], r["seed"] = c, s
            r["use_egm"] = use_egm
            r["modules"] = {"EGM": bool(acfg["use_egm"]), "DPA": bool(acfg["use_dpa"]),
                            "MSFA": bool(acfg["use_msfa"])}
            rj = json.load(open(results_path(s, c), encoding="utf-8"))
            r["id_dice"] = float(rj["test_metrics"]["Dice"])       # ID 侧直读，不重算
            r["train_min"] = float(rj.get("training_time_min", float("nan")))
            r["best_epoch"] = int(rj.get("best_epoch", -1))
            per[s][c] = r
            cache[ck] = r
            del model
            torch.cuda.empty_cache()
            print("      -> OOD=%.6f%%  ID=%.6f" % (r["mean_dice"] * 100, r["id_dice"]))
            os.makedirs(os.path.dirname(a.cache), exist_ok=True)
            with open(a.cache, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False)

    # ---------------- 复现闸 G2 ----------------
    gate_rows, gate_ok = [], True
    ref = {d["config"]: d for d in json.load(open(ETIS_SUMMARY, encoding="utf-8"))}
    if 42 in per:
        for c in a.configs:
            got = per[42][c]["mean_dice"] * 100.0
            exp = ref[c]["dice"]
            d = abs(got - exp)
            ok = d <= GATE_TOL
            gate_ok &= ok
            gate_rows.append(dict(config=c, got=got, expected=exp, delta=d, ok=bool(ok)))
        print("\n" + "-" * 92)
        print("G2 复现闸（种子 42 本评测器 vs 既有 ETIS 汇总）")
        for g in gate_rows:
            print("   %-14s got=%.6f  expected=%.6f  |Δ|=%.3e  %s"
                  % (g["config"], g["got"], g["expected"], g["delta"], "OK" if g["ok"] else "FAIL"))
        print("   ⇒ " + ("PASS" if gate_ok else "FAIL —— 评测器协议不一致，本段读数作废，STOP"))
        print("-" * 92)
    if a.gate_only:
        return 0 if gate_ok else 2
    if 42 in per and not gate_ok:
        print("✗ G2 未过 ⇒ 不产出主结果")
        return 2

    # ---------------- 逐种子名次 + 反转 ----------------
    seeds = [s for s in a.seeds if s in per]
    per_seed = {}
    for s in seeds:
        idv = {c: per[s][c]["id_dice"] for c in a.configs}
        oodv = {c: per[s][c]["mean_dice"] for c in a.configs}
        rev, tie = reversal_pairs(idv, oodv, a.configs)
        per_seed[s] = dict(
            id_dice=idv, ood_dice=oodv,
            id_rank=rank_desc(idv), ood_rank=rank_desc(oodv),
            reversals=rev, ties=tie,
            n_reversals=len(rev),
            flagship=pair_sign(idv, oodv, FLAGSHIP),
        )

    # ---------------- H4 判定 ----------------
    flags = {s: per_seed[s]["flagship"] for s in seeds}
    # 主判据 = §4.2 ①② + §4.5③（方向与种子 42 一致）
    main_ok = bool(flags) and all(
        (v["reversed"] is True) and (dir_of(v) == REF_DIR) for v in flags.values())
    # 自检：参考方向必须由种子 42 自身满足，否则 REF_DIR 无意义
    if 42 in flags:
        assert flags[42]["reversed"] is True and dir_of(flags[42]) == REF_DIR, (
            "种子 42 的旗舰对不满足 REF_DIR=%s ⇒ 参考方向定义失效，STOP" % (REF_DIR,))
    # 副判据：**可计数**的反转对（规格 §6.3 排除跨协议混杂的 #4）在**全部种子**上都反转
    consistent = []
    for (x, y) in COUNTABLE:
        okall = True
        for s in seeds:
            ps = pair_sign(per_seed[s]["id_dice"], per_seed[s]["ood_dice"], (x, y))
            if ps["reversed"] is not True:
                okall = False
        if okall:
            consistent.append((x, y))
    sec_n = len(consistent)
    sec_ok = sec_n >= SEC_MIN
    # 被排除的对单独报（仅参考，不进判据）
    excluded_read = {("%s|%s" % p): [pair_sign(per_seed[s]["id_dice"],
                                                per_seed[s]["ood_dice"], p)["reversed"]
                                     for s in seeds] for p in EXCLUDED}

    # 失败模式分侧
    fail_mode = None
    if not main_ok:
        fm = {}
        for s, v in flags.items():
            d = dir_of(v)
            if v["reversed"] is True:
                if d == REF_DIR:
                    fm[s] = "反转成立"
                else:
                    fm[s] = ("符号翻转（两侧仍相反，但方向 %s 与种子 42 的 %s 相反）"
                             " —— 规格 §4.5③ 判不过，§1「符号翻转」同判推翻" % (d, REF_DIR))
            elif v["id_tie"] or v["ood_tie"]:
                fm[s] = "平局（%s）" % ("ID 侧" if v["id_tie"] else "OOD 侧")
            elif v["d_id"] > 0:
                # 两侧同向且 Δ_ID > 0：ID 上 egm_dpa 反超 ⇒ 域外排序不变，域内分不开两者
                fm[s] = "ID 侧翻转（域外排序不变，域内已无法分开两者）"
            else:
                fm[s] = "OOD 侧翻转（域外排名本身不稳健）"
        fail_mode = fm

    verdict = "H4 支持（保留“致命反转”卖点）" if (main_ok and sec_ok) else \
              "H4 被推翻（走分支 C：只留 L1 恒等式 + L3 协议）"

    payload = dict(
        segment="P3-3", kind="multiseed_h4",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/p3_3_multiseed_eval.py",
        spec="00_docs/P3-3规格冻结_2026-09-15.md",
        seeds=seeds, configs=a.configs,
        protocol=dict(ood_data=ETIS_DIR, n_ood=len(dataset), eps_ood=EPS_OOD,
                      id_source="results.json::test_metrics.Dice (eps=1e-8)",
                      binarize="sigmoid>0.5", autocast=True,
                      tie_tol=TIE_TOL, gate_tol=GATE_TOL),
        gate_g2=dict(ok=bool(gate_ok), rows=gate_rows),
        per_seed=per_seed,
        per_config_raw={str(s): {c: {k: per[s][c][k] for k in
                                     ("mean_dice", "std", "id_dice", "n",
                                      "train_min", "best_epoch", "modules")}
                                 for c in a.configs} for s in seeds},
        h4=dict(flagship_pair=list(FLAGSHIP), flagship_per_seed=flags,
                flagship_ref_dir=list(REF_DIR),
                flagship_dir_per_seed={s: list(dir_of(flags[s])) for s in seeds},
                flagship_same_dir_as_ref={s: bool(dir_of(flags[s]) == REF_DIR)
                                          for s in seeds},
                main_criterion_spec=("§4.2 ①②（逐种子两侧符号相反 且 两路 |Δ|>1e-6）"
                                     " + §4.5③（反转方向与种子 42 的 %s 一致）" % (REF_DIR,)),
                main_criterion_pass=bool(main_ok),
                countable_pairs=[list(p) for p in COUNTABLE],
                excluded_pairs=[list(p) for p in EXCLUDED],
                excluded_pairs_read=excluded_read,
                excluded_reason=("第 8 格 egm_dpa_msfa 的种子 42 权重为 batch 4，"
                                 "其余 4 格为 batch 8 ⇒ 跨协议混杂，不计入副判据（规格 §6.3）"),
                consistent_reversal_pairs=[list(p) for p in consistent],
                secondary_n=sec_n, secondary_min_required=SEC_MIN,
                secondary_criterion_pass=bool(sec_ok),
                fail_mode_by_seed=fail_mode,
                verdict=verdict),
        boundaries=["n=3 种子，只报同向计数，不做显著性检验",
                    "5 格 n=5，ρ 仅弱陈述",
                    "ID eps=1e-8 vs OOD eps=1e-6，跨集合相减须声明",
                    "单元非独立，任何区间仅描述性",
                    "反转对 #4（egm_dpa_msfa, egm_msfa）跨协议混杂，仅参考，不计入副判据"],
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---------------- 表 ----------------
    md = [f"# T_multiseed — P3-3 关键 5 格 × {len(seeds)} 种子：ID/OOD 排名与反转", "",
          f"> 生成：{payload['generated_at']} ｜ 规格：`{payload['spec']}`", "",
          "## 逐种子读数（ID 直读 results.json；OOD 为 ETIS 196 样本均值）", "",
          "| 种子 | 配置 | EGM | DPA | MSFA | ID Dice | ID 名次 | OOD Dice | OOD 名次 | 反转对数 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        for c in sorted(a.configs, key=lambda x: per_seed[s]["id_rank"][x]):
            m = per[s][c]["modules"]
            md.append("| %d | %s | %d | %d | %d | %.4f | %d | %.4f | %d | %d |"
                      % (s, c, m["EGM"], m["DPA"], m["MSFA"],
                         per_seed[s]["id_dice"][c], per_seed[s]["id_rank"][c],
                         per_seed[s]["ood_dice"][c], per_seed[s]["ood_rank"][c],
                         per_seed[s]["n_reversals"]))
    md += ["", "## 逐种子反转对", "",
           "| 种子 | a | b | ID 值差 | OOD 值差 | ID 优者 | OOD 优者 | 放大倍数 |",
           "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        for r in per_seed[s]["reversals"]:
            md.append("| %d | %s | %s | %+.4f | %+.4f | %s | %s | %.1f× |"
                      % (s, r["a"], r["b"], r["d_id"], r["d_ood"],
                         r["winner_id"], r["winner_ood"], r["amp"]))
        if not per_seed[s]["reversals"]:
            md.append("| %d | — | — | — | — | — | — | 无反转 |" % s)
    md += ["", "## 旗舰对逐种子判定（规格 §4.5 明文要求的四列）", "",
           "> `Δ = egm_dpa − msfa_only`。两侧符号相反 ⇒ 反转；方向还须与种子 42 的 `%s` 一致才计入主判据。"
           % (REF_DIR,), "",
           "| 种子 | Δ_ID | Δ_OOD | sign(Δ_ID) | sign(Δ_OOD) | 两侧符号相反 | 与种子42同向 | 主判据该种子通过 |",
           "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        v = flags[s]
        dv = dir_of(v)
        md.append("| %d | %+.4f | %+.4f | %+d | %+d | %s | %s | %s |"
                  % (s, v["d_id"], v["d_ood"], dv[0], dv[1],
                     "是" if v["reversed"] is True else ("平局" if v["reversed"] is None else "否"),
                     "是" if dv == REF_DIR else "否",
                     "**是**" if (v["reversed"] is True and dv == REF_DIR) else "否"))
    md += ["", "## H4 判定", "",
           "| 项 | 值 |", "|---|---|",
           "| 旗舰对 | (%s, %s) |" % FLAGSHIP,
           "| 主判据（3/3 种子反转 **且** 方向与种子 42 的 `%s` 一致，规格 §4.2①②+§4.5③） | %s |"
           % (REF_DIR, "通过" if main_ok else "**未通过**"),
           "| 副判据（全种子同号反转的对数 ≥ %d） | %d 对 %s |"
           % (SEC_MIN, sec_n, "通过" if sec_ok else "**未通过**"),
           "| **结论** | **%s** |" % verdict, ""]
    if fail_mode:
        md += ["### 失败模式（分侧）", "", "| 种子 | 失败模式 |", "|---|---|"]
        for s, v in fail_mode.items():
            md.append("| %d | %s |" % (s, v))
        md.append("")
    if any(per_seed[s]["ties"] for s in seeds):
        md += ["### 平局对（不计入反转）", "", "| 种子 | a | b | ID 差 | OOD 差 | 原因 |",
               "|---|---|---|---|---|---|"]
        for s in seeds:
            for t in per_seed[s]["ties"]:
                md.append("| %d | %s | %s | %+.2e | %+.2e | %s |"
                          % (s, t["a"], t["b"], t["d_id"], t["d_ood"], t["reason"]))
        md.append("")
    md += ["## 边界（须随结论出现）", ""] + ["- " + b for b in payload["boundaries"]]
    os.makedirs(os.path.dirname(a.table), exist_ok=True)
    with open(a.table, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    # ---------------- 控制台 ----------------
    print()
    print("=" * 92)
    for s in seeds:
        print("种子 %-5d | ID 序: %s" % (s, " > ".join(
            sorted(a.configs, key=lambda x: per_seed[s]["id_rank"][x]))))
        print("           | OOD 序: %s" % (" > ".join(
            sorted(a.configs, key=lambda x: per_seed[s]["ood_rank"][x]))))
        print("           | 反转 %d 对 | 旗舰对反转=%s"
              % (per_seed[s]["n_reversals"], per_seed[s]["flagship"]["reversed"]))
        v = flags[s]
        print("           | 旗舰 Δ_ID=%+.4f Δ_OOD=%+.4f 符号=%s 与种子42同向=%s"
              % (v["d_id"], v["d_ood"], dir_of(v), dir_of(v) == REF_DIR))
    print("-" * 92)
    print("主判据(旗舰对 3/3): %s | 副判据(同号反转对 %d ≥ %d): %s"
          % (main_ok, sec_n, SEC_MIN, sec_ok))
    print("★ H4 判定: " + verdict)
    if fail_mode:
        for s, v in fail_mode.items():
            print("   种子 %d 失败模式: %s" % (s, v))
    print("已写 " + a.out)
    print("已写 " + a.table)
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
