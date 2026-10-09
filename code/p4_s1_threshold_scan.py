# -*- coding: utf-8 -*-
"""
文件名: p4_s1_threshold_scan.py
功能: 【论文二 P4-3 / Fig S1】边界条件 B2 · 检出-勾画分解的阈值敏感性扫描

规格: 00_docs/B2阈值扫描规格冻结_2026-09-20.md  (G6 跑前冻结, 不得事后改判据)

要回答的问题
------------
B2 声明 "检出因子依赖二值化阈值 τ, 勾画因子对 τ 的依赖较弱;
必须报告 τ 并给阈值敏感性扫描"。全篇现有读数**只有一个 τ=0.50**,
本脚本把它扩成 21 点 τ 网格, 并回答 "主结论是否只是 τ 选择的产物"。

扫描范围 (冻结)
---------------
架构 n=14 = decompose.ID_BASELINE_ARCHS (13 baseline) + EGAUNet
权重口径 = 路线 A (分布内联合训练权重), 即 p31a_ood_idtrain.build_jobs()
域       = ID: processed_data/test (242 张)  |  OOD: data_zeroshot/etis (196 张)
           两侧读数**来自同一个模型** ⇒ τ 效应不与训练过程差异混淆
τ 网格   = {0.01} ∪ {0.05..0.95 步长 0.05} ∪ {0.99}  共 21 点 (含 0.50 锚点)

逐样本量 (全部在 float32 概率图上算, 严格大于)
---------------------------------------------
    gs          = |G|                        (τ 无关; B3 要求 gs > 0)
    ps(τ)       = |{prob > τ}|
    inter(τ)    = |{prob > τ} ∩ G|
    det_honest  = 1[ inter >= 1 ]            <- |P_τ ∩ G| > 0 的忠实实现 (本次判据)
    det_frozen  = 1[ (inter+eps)/(gs+eps) > 0 ]
                  ID  侧 eps=1e-8 ⇒ 恒真 (B6 的 eps 构造, 与 τ 无关)
                  OOD 侧 recall_gt = inter/gs 无 eps ⇒ 与 det_honest 重合
    dice(τ)     = (2*inter + eps) / (ps + gs + eps)
                  eps = 1e-8 (ID, utils/metrics.py) / 1e-6 (OOD, train_cross.calculate_metrics)

⚠️ 口径警告: p31a_units_n14.csv 的 id_del **不是** ID 侧勾画因子 ——
   冻结判据下它代数恒等于 id_dice (alignment_test.py §17 已断言)。
   本脚本的 Delineation(τ) 是**实质判据下的条件均值**, 与 id_del 不同口径。

设计 (单一真源, 不重写协议)
--------------------------
    模型工厂 / 权重载入 / eabs  复用 p31_cross_eval
    架构清单 + 权重目录         复用 p31a_ood_idtrain.build_jobs
    ID 数据加载 + 张量构造      复用 p3_7_stress_test.load_id_test
                                (张量构造与 utils/dataset.py::MedicalDataset test 档逐字对齐)
    OOD 数据加载                复用 cross_dataset.dataset.CrossDataset
    判据常数                    复用 decompose (EPS_ID/EPS_OOD/FROZEN_THRESH/...)
    双路复核聚合                复用 decompose.factorize
    必须 with autocast():      既有全部产物均在该设置下前向 (漏掉会破坏逐位一致)

闸门 (跑前冻结)
--------------
    G2  τ=0.50 锚点对 p31a_units_n14.csv:
          Detection      vs id_det_subst / ood_det_subst   容差 T1=1e-9
          Delineation    vs ood_del                        容差 T1=1e-9
          E[Dice]        vs id_dice / ood_dice             容差 T2=1e-6(ID) / 1e-4(OOD)
          Del_frozen     vs id_del                         容差 T1  (验口径: 应恒等 id_dice)
        T1 超差 ⇒ ANCHOR_FAIL 阻断; T2 超差 ⇒ 记 ANCHOR_PROVISIONAL 并列出架构名
    G4  τ=0.50 用 decompose.factorize 走第二条聚合路, max|Δ| <= 1e-12
    C4  max|residual(τ)| < 1e-6 对 14x2x21 全体成立

结论判据 C1–C3 见规格 §6 (双向解释, 任一结果都合格, 只有"不报告"不合格)。

断点续跑
-------
每 (arch, domain) 跑完立刻写 partial/<arch>__<domain>.csv; 重跑默认跳过 (--force 覆盖)。

用法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    # 冒烟 (1 架构 16 张, 不落盘, 打印结构自检)
    $PY E:/paper2_ablation_reliability/02_code/analysis/p4_s1_threshold_scan.py --gate-only
    # 全量 14 架构 x (242+196)
    $PY E:/paper2_ablation_reliability/02_code/analysis/p4_s1_threshold_scan.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p31_cross_eval as X                       # noqa: E402  (副作用: chdir 到 D:/medical_segmentation)
import p31a_ood_idtrain as A                     # noqa: E402
import p3_7_stress_test as S                     # noqa: E402
from decompose import (                          # noqa: E402
    EPS_ID, EPS_OOD, FROZEN_THRESH, RESIDUAL_TOL, factorize,
)
from cross_dataset.dataset import CrossDataset   # noqa: E402

E_ROOT = X.E_ROOT

# --------------------------------------------------------------------------- #
# 冻结常数 (规格 §2/§3)
# --------------------------------------------------------------------------- #
TAU = tuple([0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99])
TAU_ANCHOR = 0.50
assert TAU_ANCHOR in TAU, 'τ=0.50 必须在网格内 (G2 锚点)'
N_TAU = len(TAU)

ID_DOMAIN, OOD_DOMAIN = 'ID', 'OOD'
TARGET_OOD = A.TARGET_DIR                        # 'data_zeroshot/etis'
OOD_X = A.DIRECTION_LABEL                        # 'idtrain_to_etis'

# 锚点容差档 (规格 §5; 分档经 "判据修订 M1/M2" 修正 —— 见
# 00_docs/B2阈值扫描_判据修订_2026-09-20.md。修订只动容差分档, 不动结论判据 C1–C4)
TOL_T1 = 1e-9     # T1  整数计数类 (detection): 与锚应逐位相等
TOL_T1B = 1e-7    # T1b 浮点均值类 (delineation / mean_dice): 均值最后两位
TOL_T2 = 1e-4     # T2  上游已登记的第二源差异 (PolypPVT ID, REGISTERED_DIFF)
TOL_DUAL_PATH = 1e-12

# 结论判据 (规格 §6)
BAND = (0.30, 0.70)
C2_MIN_GAP = 0.05
C3_MIN_RHO = 0.90
N_BOOT = 10000
BOOT_SEED = 20260920

ANCHOR_CSV = f'{E_ROOT}/03_results/stats/p31a_units_n14.csv'    # detection + OOD
ANCHOR_ID_CSV = f'{E_ROOT}/03_results/stats/id_units_n14.csv'   # ID: id_dice / id_dice_recalc / cons_status

OUT_ROOT = f'{E_ROOT}/03_results/raw/s1_threshold_scan'
OUT_PARTIAL = f'{OUT_ROOT}/partial'
OUT_PER_SAMPLE = f'{OUT_ROOT}/per_sample.csv'
OUT_UNITS = f'{OUT_ROOT}/units.csv'
OUT_JSON = f'{E_ROOT}/03_results/stats/s1_threshold_scan.json'
OUT_MD = f'{E_ROOT}/03_results/tables/T_s1_threshold.md'


# --------------------------------------------------------------------------- #
# 逐样本: 21 个 τ 下的 (ps, inter, dice, det)
# --------------------------------------------------------------------------- #
def sweep_sample(prob: np.ndarray, gt_bool: np.ndarray, eps: float):
    """prob (H,W) f32 in [0,1]; gt_bool (H,W) bool。

    返回 (gs, ps[N_TAU], inter[N_TAU], dice[N_TAU], det_honest[N_TAU], det_frozen[N_TAU])
    全部按规格 §3 逐字实现 (严格大于)。
    """
    gs = int(gt_bool.sum())
    pf = prob.ravel()
    gf = gt_bool.ravel()

    ps = np.zeros(N_TAU, dtype=np.int64)
    inter = np.zeros(N_TAU, dtype=np.int64)
    for k, t in enumerate(TAU):
        m = pf > t                       # 严格大于
        ps[k] = int(np.count_nonzero(m))
        inter[k] = int(np.count_nonzero(m & gf))

    dice = (2.0 * inter.astype(np.float64) + eps) / (
        ps.astype(np.float64) + gs + eps)
    det_honest = (inter >= 1)            # |P_τ ∩ G| > 0
    if eps == EPS_ID:
        # ID 侧: recall_metric = (inter+eps)/(gs+eps) > 0 恒真 (B6 的 eps 构造)
        det_frozen = np.ones(N_TAU, dtype=bool)
    else:
        # OOD 侧: recall_gt = inter/gs 无 eps ⇒ 与实质判据重合
        det_frozen = det_honest.copy()
    return gs, ps, inter, dice, det_honest, det_frozen


# --------------------------------------------------------------------------- #
# 单 (arch, domain) 推理
# --------------------------------------------------------------------------- #
def run_cell(arch, ckpt_dir, domain, device, id_data, ood_ds, limit=None):
    """返回 (df_cell, meta)。df_cell 列见 spec; 每行 = 一个样本 x 一个 τ。"""
    wpath = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(wpath):
        return None, dict(arch=arch, domain=domain, status='NO_WEIGHT')

    model = X.build_model(arch)
    fmt, miss, unexp = X.load_weights(model, wpath, device)
    if len(miss) or len(unexp):
        # 与 P3-1/P3-7 同铁律: 键不匹配会静默全零预测
        raise RuntimeError(
            f'{arch}: 权重键不匹配 (missing={len(miss)}, unexpected={len(unexp)}) '
            f'-> 拒绝运行。 path={wpath} fmt={fmt}')
    model = model.to(device).eval()

    eps = EPS_ID if domain == ID_DOMAIN else EPS_OOD
    rows = []
    n_gs0 = 0

    with torch.no_grad():
        if domain == ID_DOMAIN:
            iterable = id_data if limit is None else id_data[:limit]
            for name, img, gt in tqdm(iterable, desc=f'{arch}/{domain}', ncols=88,
                                      leave=False):
                # 张量构造与 p3_7_stress_test.eval_arch 逐字一致
                xt = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() \
                    .unsqueeze(0).to(device)
                with autocast():
                    out = model(xt)
                    if isinstance(out, tuple):
                        out = out[0]
                prob = torch.sigmoid(out.float())[0, 0].cpu().numpy()
                gs, ps, inter, dice, dh, dfz = sweep_sample(prob, gt, eps)
                if gs == 0:
                    n_gs0 += 1
                    continue
                for k, t in enumerate(TAU):
                    rows.append((arch, domain, name, t, gs, int(ps[k]),
                                 int(inter[k]), float(dice[k]),
                                 int(dh[k]), int(dfz[k])))
        else:
            n = len(ood_ds) if limit is None else min(limit, len(ood_ds))
            for i in tqdm(range(n), desc=f'{arch}/{domain}', ncols=88, leave=False):
                img, mask, name = ood_ds[i]
                img_t = img.unsqueeze(0).to(device)
                with autocast():
                    out = model(img_t)
                    if isinstance(out, tuple):
                        out = out[0]
                prob = torch.sigmoid(out.float())[0, 0].cpu().numpy()
                g = mask.numpy().squeeze()
                uq = np.unique(g)
                if not np.all(np.isin(uq, (0.0, 1.0))):
                    raise RuntimeError(f'{arch}/{name}: OOD 掩膜非 {{0,1}}: {uq[:8]}')
                gt = (g > 0.5)
                gs, ps, inter, dice, dh, dfz = sweep_sample(prob, gt, eps)
                if gs == 0:
                    n_gs0 += 1
                    continue
                for k, t in enumerate(TAU):
                    rows.append((arch, domain, name, t, gs, int(ps[k]),
                                 int(inter[k]), float(dice[k]),
                                 int(dh[k]), int(dfz[k])))

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows, columns=['arch', 'domain', 'name', 'tau', 'gt_area',
                                     'pred_area', 'inter', 'dice',
                                     'det_honest', 'det_frozen'])
    meta = dict(arch=arch, domain=domain, status='OK', weight_format=fmt,
                n_samples=int(df['name'].nunique()) if len(df) else 0,
                n_gs0_skipped=n_gs0, eps=eps)
    return df, meta


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """逐 (arch, domain, tau) 聚合出 Detection / Detection_frozen / E[Dice] /
    Delineation / residual。"""
    out = []
    for (arch, dom, tau), g in df.groupby(['arch', 'domain', 'tau'], sort=True):
        det = float(g['det_honest'].mean())
        detf = float(g['det_frozen'].mean())
        md = float(g['dice'].mean())
        sub = g[g['det_honest'] == 1]
        delin = float(sub['dice'].mean()) if len(sub) else float('nan')
        out.append(dict(arch=arch, domain=dom, tau=float(tau), n=len(g),
                        detection=det, detection_frozen=detf,
                        mean_dice=md, delineation=delin,
                        product=det * delin, residual=md - det * delin,
                        n_undetected=int((g['det_honest'] == 0).sum())))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# G2 锚点闸
# --------------------------------------------------------------------------- #
def gate_g2(units: pd.DataFrame) -> dict:
    """τ=0.50 对锚表逐架构比对。

    分档 (判据修订 M1): T1 = 整数计数类 (应逐位相等) / T1b = 浮点均值类
    ID 侧同源锚取 `id_units_n14.csv::id_dice_recalc`; `id_dice` 作为第二源单列
    (PolypPVT 上游已登记 REGISTERED_DIFF, 见判据修订 M2)。
    """
    if not os.path.exists(ANCHOR_CSV):
        return dict(status='NO_ANCHOR_FILE', path=ANCHOR_CSV)
    ref = pd.read_csv(ANCHOR_CSV).set_index('arch')
    refid = (pd.read_csv(ANCHOR_ID_CSV).set_index('arch')
             if os.path.exists(ANCHOR_ID_CSV) else None)
    a = units[np.isclose(units['tau'], TAU_ANCHOR)].set_index(['arch', 'domain'])

    rows = []
    for arch in sorted(ref.index):
        for dom, suf in ((ID_DOMAIN, 'id'), (OOD_DOMAIN, 'ood')):
            if (arch, dom) not in a.index:
                continue
            r = a.loc[(arch, dom)]
            checks = []

            def _chk(name, got, exp, tol, tier, note=''):
                if exp is None or (isinstance(exp, float) and exp != exp):
                    return
                d = abs(float(got) - float(exp))
                checks.append(dict(quantity=name, ours=float(got), ref=float(exp),
                                   abs_diff=d, tol=tol, tier=tier, note=note,
                                   status='PASS' if d <= tol else 'FAIL'))

            _chk('detection', r['detection'], ref.loc[arch, f'{suf}_det_subst'],
                 TOL_T1, 'T1')
            if suf == 'ood':
                _chk('delineation', r['delineation'], ref.loc[arch, 'ood_del'],
                     TOL_T1B, 'T1b')
                _chk('mean_dice', r['mean_dice'], ref.loc[arch, 'ood_dice'],
                     TOL_T1B, 'T1b')
            elif refid is not None and arch in refid.index:
                st = str(refid.loc[arch, 'cons_status'])
                registered = (st == 'REGISTERED_DIFF')
                _chk('detection_vs_id_units', r['detection'],
                     refid.loc[arch, 'id_det_subst'], TOL_T1, 'T1')
                # 同源锚 = id_dice_recalc (判据修订 M2) —— 与本次扫描同源, 恒为 T1b
                _chk('mean_dice', r['mean_dice'], refid.loc[arch, 'id_dice_recalc'],
                     TOL_T1B, 'T1b')
                # 以下两项的**参照列**派生自 `id_dice` 源列:
                # 该列在 REGISTERED_DIFF 架构上已知与同源复算不一致 (判据修订 M2 列级规则),
                # 故这些架构一律降为 T2 并在报告表格中逐条列出。
                _chk('mean_dice_vs_id_del', r['mean_dice'],
                     refid.loc[arch, 'id_del_frozen'],
                     TOL_T2 if registered else TOL_T1B,
                     'T2' if registered else 'T1b', note=st)
                _chk('mean_dice_vs_id_dice_2ndsrc', r['mean_dice'],
                     refid.loc[arch, 'id_dice'],
                     TOL_T2 if registered else TOL_T1B,
                     'T2' if registered else 'T1b', note=st)
            else:
                _chk('mean_dice', r['mean_dice'], ref.loc[arch, 'id_dice'],
                     TOL_T1B, 'T1b')
            rows.append(dict(arch=arch, domain=dom, checks=checks))

    flat = [c | dict(arch=r['arch'], domain=r['domain'])
            for r in rows for c in r['checks']]
    strict = [c for c in flat if c['tier'] in ('T1', 'T1b')]
    loose = [c for c in flat if c['tier'] == 'T2']
    strict_fail = [c for c in strict if c['status'] == 'FAIL']
    loose_fail = [c for c in loose if c['status'] == 'FAIL']

    def _mx(cs):
        return max((c['abs_diff'] for c in cs), default=None)

    return dict(
        status='ANCHOR_FAIL' if strict_fail else
               ('ANCHOR_PROVISIONAL' if loose_fail else 'PASS'),
        anchor_file=ANCHOR_CSV, anchor_id_file=ANCHOR_ID_CSV, tau=TAU_ANCHOR,
        revision='判据修订 M1/M2 (00_docs/B2阈值扫描_判据修订_2026-09-20.md)',
        n_checks=len(flat),
        n_t1=len([c for c in flat if c['tier'] == 'T1']),
        n_t1b=len([c for c in flat if c['tier'] == 'T1b']),
        n_t2=len(loose),
        n_strict_fail=len(strict_fail), n_loose_fail=len(loose_fail),
        max_abs_diff_t1=_mx([c for c in flat if c['tier'] == 'T1']),
        max_abs_diff_t1b=_mx([c for c in flat if c['tier'] == 'T1b']),
        max_abs_diff_t2=_mx(loose),
        strict_failures=strict_fail, loose_failures=loose_fail,
        t2_rows=loose,
        rows=flat,
    )


# --------------------------------------------------------------------------- #
# G4 双路复核 (decompose.factorize 走第二条聚合路)
# --------------------------------------------------------------------------- #
def gate_g4(df: pd.DataFrame, units: pd.DataFrame) -> dict:
    rows = []
    for (arch, dom), g in df.groupby(['arch', 'domain'], sort=True):
        g0 = g[np.isclose(g['tau'], TAU_ANCHOR)]
        if not len(g0):
            continue
        fz = factorize(g0['dice'].values, g0['inter'].values, thr=FROZEN_THRESH)
        u = units[(units['arch'] == arch) & (units['domain'] == dom)
                  & np.isclose(units['tau'], TAU_ANCHOR)].iloc[0]
        d_det = abs(fz['detection'] - float(u['detection']))
        d_del = abs(fz['delineation'] - float(u['delineation']))
        d_dice = abs(fz['mean_dice'] - float(u['mean_dice']))
        rows.append(dict(arch=arch, domain=dom, path_a_detection=float(u['detection']),
                         path_b_detection=fz['detection'],
                         path_a_delineation=float(u['delineation']),
                         path_b_delineation=fz['delineation'],
                         path_a_mean_dice=float(u['mean_dice']),
                         path_b_mean_dice=fz['mean_dice'],
                         max_abs_diff=max(d_det, d_del, d_dice)))
    mx = max((r['max_abs_diff'] for r in rows), default=None)
    return dict(status='PASS' if (mx is not None and mx <= TOL_DUAL_PATH) else 'FAIL',
                tol=TOL_DUAL_PATH, max_abs_diff=mx, n_cells=len(rows), rows=rows)


# --------------------------------------------------------------------------- #
# 结论判据 C1–C3
# --------------------------------------------------------------------------- #
def _band(units, dom, col):
    b = units[(units['domain'] == dom) & (units['tau'] >= BAND[0])
              & (units['tau'] <= BAND[1])]
    return b


def conclusions(units: pd.DataFrame) -> dict:
    from scipy import stats as sps

    # --- C1: S_det vs S_del (架构均值曲线在操作带内的极差) --------------------
    c1 = {}
    for dom in (ID_DOMAIN, OOD_DOMAIN):
        b = _band(units, dom, None)
        det_curve = b.groupby('tau')['detection'].mean()
        del_curve = b.groupby('tau')['delineation'].mean()
        s_det = float(det_curve.max() - det_curve.min())
        s_del = float(del_curve.max() - del_curve.min())
        c1[dom] = dict(s_detection=s_det, s_delineation=s_del,
                       ratio=(s_det / s_del) if s_del > 0 else float('inf'),
                       detection_range=[float(det_curve.min()), float(det_curve.max())],
                       delineation_range=[float(del_curve.min()), float(del_curve.max())],
                       support_b2=(s_det > s_del))
    c1['support_b2_all'] = all(c1[d]['support_b2'] for d in (ID_DOMAIN, OOD_DOMAIN))

    # --- C2: ID-OOD 检出差在操作带内每点 > 0.05 (架构级配对 bootstrap) --------
    archs = sorted(units['arch'].unique())
    taus = sorted(t for t in units['tau'].unique() if BAND[0] <= t <= BAND[1])
    rng = np.random.default_rng(BOOT_SEED)
    bidx = rng.integers(0, len(archs), size=(N_BOOT, len(archs)))
    c2_rows = []
    for t in taus:
        a = units[np.isclose(units['tau'], t)]
        piv = a.pivot_table(index='arch', columns='domain', values='detection')
        piv = piv.reindex(archs)
        g = (piv[ID_DOMAIN] - piv[OOD_DOMAIN]).values
        boot = g[bidx].mean(axis=1)
        c2_rows.append(dict(tau=float(t), gap=float(g.mean()),
                            ci95=[float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
                            ci95_excludes_zero=bool(np.percentile(boot, 2.5) > 0),
                            above_min_gap=bool(g.mean() > C2_MIN_GAP)))
    c2 = dict(min_gap_observed=float(min(r['gap'] for r in c2_rows)),
              criterion=C2_MIN_GAP,
              all_above=all(r['above_min_gap'] for r in c2_rows),
              all_ci_exclude_zero=all(r['ci95_excludes_zero'] for r in c2_rows),
              rows=c2_rows)

    # --- C3: 各域内 E[Dice] 架构排名相对 τ=0.50 的 Spearman -------------------
    c3 = {}
    for dom in (ID_DOMAIN, OOD_DOMAIN):
        d = units[units['domain'] == dom]
        piv = d.pivot_table(index='arch', columns='tau', values='mean_dice')
        base = piv[TAU_ANCHOR]
        rows = []
        for t in sorted(piv.columns):
            rho = float(sps.spearmanr(piv[t], base).statistic)
            rows.append(dict(tau=float(t), spearman_vs_anchor=rho,
                             in_band=bool(BAND[0] <= t <= BAND[1]),
                             passes=bool(rho >= C3_MIN_RHO)))
        band_rows = [r for r in rows if r['in_band']]
        crosses = [r['tau'] for r in rows if r['spearman_vs_anchor'] < C3_MIN_RHO]
        c3[dom] = dict(min_rho_in_band=min((r['spearman_vs_anchor'] for r in band_rows),
                                          default=None),
                       criterion=C3_MIN_RHO,
                       all_pass_in_band=all(r['passes'] for r in band_rows),
                       tau_below_criterion=crosses, rows=rows)
    return dict(C1=c1, C2=c2, C3=c3)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def write_report(df, units, g2, g4, c4, cc, meta, elapsed):
    os.makedirs(OUT_ROOT, exist_ok=True)
    df.to_csv(OUT_PER_SAMPLE, index=False)
    units.to_csv(OUT_UNITS, index=False)

    verdict = dict(
        G2_anchor=g2['status'], G4_dual_path=g4['status'],
        C4_identity='PASS' if c4['pass'] else 'FAIL',
        C1_b2_supported=cc['C1']['support_b2_all'],
        C2_gap_robust=cc['C2']['all_above'],
        C3_ranking_stable=all(cc['C3'][d]['all_pass_in_band']
                              for d in (ID_DOMAIN, OOD_DOMAIN)),
    )
    payload = dict(
        generated_at=datetime.now().isoformat(timespec='seconds'),
        script='02_code/analysis/p4_s1_threshold_scan.py',
        spec='00_docs/B2阈值扫描规格冻结_2026-09-20.md',
        elapsed_sec=round(elapsed, 1),
        protocol=dict(
            archs=meta['archs'], n_archs=len(meta['archs']),
            route='A (_ood_idtrain: 同一个模型评 ID 与 OOD)',
            id_source='processed_data/test (242)',
            ood_source=f'{TARGET_OOD} (196) direction={OOD_X}',
            tau=list(TAU), tau_anchor=TAU_ANCHOR,
            eps=dict(id=EPS_ID, ood=EPS_OOD),
            detection_criterion='det_honest = 1[|P_tau ∩ G| >= 1]  (+ det_frozen for B6)',
            reuse=['p31_cross_eval.build_model/load_weights',
                   'p31a_ood_idtrain.build_jobs', 'p3_7_stress_test.load_id_test',
                   'CrossDataset', 'autocast', 'decompose.factorize'],
        ),
        verdict=verdict,
        gates=dict(G2=g2, G4=g4, C4=c4),
        conclusions=cc,
        cells=meta['cells'],
    )
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_jsonable)

    L = []
    L.append('# Fig S1 / B2 阈值敏感性扫描 — 机器生成报告')
    L.append('')
    L.append(f'- **Generated:** {payload["generated_at"]}')
    L.append('- **Script:** `02_code/analysis/p4_s1_threshold_scan.py`')
    L.append(f'- **Spec (G6 冻结):** `{payload["spec"]}`')
    L.append(f'- **Elapsed:** {elapsed:.1f} s')
    L.append(f'- **Scope:** n={len(meta["archs"])} 架构 × '
             f'{len(TAU)} 个 τ × (ID 242 + OOD 196)')
    L.append('- **τ 网格:** ' + ', '.join(f'{t:.2f}' for t in TAU))
    L.append('')
    L.append('## 闸门')
    L.append('')
    L.append('| 闸门 | 状态 | 关键读数 |')
    L.append('|---|---|---|')
    L.append(f'| G2 锚点 (τ=0.50) | **{g2["status"]}** | '
             f'T1(整数计数) n={g2.get("n_t1")} fail={g2.get("n_strict_fail")} '
             f'max|Δ|={g2.get("max_abs_diff_t1")}; '
             f'T1b(浮点均值) n={g2.get("n_t1b")} '
             f'max|Δ|={g2.get("max_abs_diff_t1b")}; '
             f'T2(已登记第二源) n={g2.get("n_t2")} '
             f'fail={g2.get("n_loose_fail")} '
             f'max|Δ|={g2.get("max_abs_diff_t2")} |')
    L.append(f'| G4 双路复核 | **{g4["status"]}** | '
             f'max|Δ|={g4["max_abs_diff"]} (tol {g4["tol"]:.0e}), '
             f'{g4["n_cells"]} 个 arch×domain 格 |')
    L.append(f'| C4 恒等式 | **{"PASS" if c4["pass"] else "FAIL"}** | '
             f'max|residual| = {c4["max_abs_residual"]:.3e} '
             f'(tol {c4["tol"]:.0e}), 覆盖 {c4["n_cells"]} 格 |')
    L.append('')
    if g2.get('t2_rows'):
        L.append('**G2 T2 档（上游已登记的第二源/派生列差异；列级规则，不阻断）**')
        L.append('')
        L.append('| arch | domain | quantity | ours | ref | |Δ| | 上游登记 |')
        L.append('|---|---|---|---:|---:|---:|---|')
        for c in g2['t2_rows']:
            L.append(f'| {c["arch"]} | {c["domain"]} | {c["quantity"]} | '
                     f'{c["ours"]:.6f} | {c["ref"]:.6f} | {c["abs_diff"]:.3e} | '
                     f'{c.get("note", "")} |')
        L.append('')
    if g2.get('strict_failures'):
        L.append('**G2 严格档超差（阻断级，须查因）**')
        L.append('')
        L.append('| arch | domain | quantity | ours | ref | |Δ| | tol |')
        L.append('|---|---|---|---:|---:|---:|---:|')
        for c in g2['strict_failures']:
            L.append(f'| {c["arch"]} | {c["domain"]} | {c["quantity"]} | '
                     f'{c["ours"]:.6f} | {c["ref"]:.6f} | {c["abs_diff"]:.3e} | '
                     f'{c["tol"]:.0e} |')
        L.append('')
    L.append('## 结论判据（C1–C3，规格 §6 冻结）')
    L.append('')
    L.append('| 判据 | 结果 | 读数 |')
    L.append('|---|---|---|')
    for dom in (ID_DOMAIN, OOD_DOMAIN):
        d = cc['C1'][dom]
        L.append(f'| C1 {dom}：S_det > S_del | '
                 f'**{"支持 B2" if d["support_b2"] else "不支持(须弱化 B2 措辞)"}** | '
                 f'S_det={d["s_detection"]:.4f} vs S_del={d["s_delineation"]:.4f} '
                 f'(比值 {d["ratio"]:.2f}×) |')
    L.append(f'| C2：ID−OOD 检出差在带内每点 > {C2_MIN_GAP} | '
             f'**{"成立" if cc["C2"]["all_above"] else "不成立"}** | '
             f'最小差 {cc["C2"]["min_gap_observed"]:.4f}；'
             f'CI 全不含 0: {cc["C2"]["all_ci_exclude_zero"]} |')
    for dom in (ID_DOMAIN, OOD_DOMAIN):
        d = cc['C3'][dom]
        L.append(f'| C3 {dom}：E[Dice] 排名对 τ 稳定 (ρ≥{C3_MIN_RHO}) | '
                 f'**{"成立" if d["all_pass_in_band"] else "不成立"}** | '
                 f'带内最小 ρ={d["min_rho_in_band"]:.4f}；'
                 f'低于判据的 τ={d["tau_below_criterion"] or "无"} |')
    L.append('')
    L.append('## 曲线数值（架构均值；操作带内的代表点）')
    L.append('')
    reps = [t for t in (0.30, 0.40, 0.50, 0.60, 0.70) if t in TAU]
    for dom in (ID_DOMAIN, OOD_DOMAIN):
        L.append(f'### {dom}')
        L.append('')
        L.append('| τ | Detection | Detection_frozen | Delineation | E[Dice] | residual |')
        L.append('|---:|---:|---:|---:|---:|---:|')
        for t in reps:
            g = units[(units['domain'] == dom) & np.isclose(units['tau'], t)]
            L.append(f'| {t:.2f} | {g["detection"].mean():.4f} | '
                     f'{g["detection_frozen"].mean():.4f} | '
                     f'{g["delineation"].mean():.4f} | '
                     f'{g["mean_dice"].mean():.4f} | '
                     f'{g["residual"].abs().max():.2e} |')
        L.append('')
    L.append('## 复现')
    L.append('')
    L.append('```bash')
    L.append('D:/miniconda/aniconda/envs/medical_seg/python.exe '
             'E:/paper2_ablation_reliability/02_code/analysis/p4_s1_threshold_scan.py')
    L.append('```')
    L.append('')
    with open(OUT_MD, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    return payload


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serializable: {type(o)}')


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--archs', nargs='+', default=None)
    ap.add_argument('--limit', type=int, default=None, help='每域样本数上限 (冒烟用)')
    ap.add_argument('--gate-only', action='store_true',
                    help='单架构小样本冒烟: 只做结构自检, 不落盘')
    ap.add_argument('--force', action='store_true', help='覆盖 partial 单元格')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args(argv)

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}')
    print(f'tau grid ({N_TAU}): ' + ', '.join(f'{t:.2f}' for t in TAU))

    jobs = A.build_jobs(args.archs)
    if args.gate_only and args.archs is None:
        jobs = [j for j in jobs if j[0] == 'EGAUNet']
        args.limit = args.limit or 16
    print(f'jobs={len(jobs)}: ' + ', '.join(a for a, _ in jobs))

    print('loading ID test ...')
    id_data = S.load_id_test()
    print(f'  ID n={len(id_data)}')
    print(f'loading OOD {TARGET_OOD} ...')
    ood_ds = CrossDataset(TARGET_OOD, augment=False, strong_augment=False)
    print(f'  OOD n={len(ood_ds)}')

    os.makedirs(OUT_PARTIAL, exist_ok=True)
    frames, cells = [], []
    for arch, ckpt in jobs:
        for dom in (ID_DOMAIN, OOD_DOMAIN):
            part = os.path.join(OUT_PARTIAL, f'{arch}__{dom}.csv')
            if os.path.exists(part) and not args.force:
                d = pd.read_csv(part)
                frames.append(d)
                cells.append(dict(arch=arch, domain=dom, status='RESUMED',
                                  n_samples=int(d['name'].nunique())))
                print(f'>>> {arch}/{dom}: RESUMED ({d["name"].nunique()} samples)')
                continue
            print(f'\n>>> {arch}/{dom}')
            df, meta = run_cell(arch, ckpt, dom, device, id_data, ood_ds,
                                limit=args.limit)
            if df is None:
                print('    [跳过] 权重缺失')
                cells.append(meta)
                continue
            print(f"    n={meta['n_samples']}  fmt={meta['weight_format']}  "
                  f"gs==0 跳过={meta['n_gs0_skipped']}")
            frames.append(df)
            cells.append(meta)
            if not args.gate_only and not args.no_write:
                df.to_csv(part, index=False)

    if not frames:
        print('\n无可聚合数据。'); return 1

    df = pd.concat(frames, ignore_index=True)
    units = aggregate(df)
    print(f'\n聚合完成: {len(df)} 逐样本×τ 行 -> {len(units)} 格')

    # ---- 结构自检 (冒烟用, 恒跑) ----
    chk = {}
    b = units[np.isclose(units['tau'], TAU_ANCHOR)]
    chk['tau_monotone_pred_area'] = bool(
        df.sort_values('tau').groupby(['arch', 'domain', 'name'])['pred_area']
        .apply(lambda s: bool((np.diff(s.values) <= 0).all())).all())
    chk['det_monotone_in_tau'] = bool(
        df.sort_values('tau').groupby(['arch', 'domain', 'name'])['det_honest']
        .apply(lambda s: bool((np.diff(s.values) <= 0).all())).all())
    chk['det_frozen_id_all_one'] = bool(
        (units[units['domain'] == ID_DOMAIN]['detection_frozen'] == 1.0).all())
    chk['det_frozen_ood_equals_honest'] = bool(np.allclose(
        units[units['domain'] == OOD_DOMAIN]['detection_frozen'],
        units[units['domain'] == OOD_DOMAIN]['detection']))
    chk['gs_positive'] = bool((df['gt_area'] > 0).all())
    print('结构自检: ' + ', '.join(f'{k}={"OK" if v else "FAIL"}' for k, v in chk.items()))

    # ---- C4 恒等式闸 ----
    mx = float(units['residual'].abs().max())
    c4 = dict(pass_=bool(mx < RESIDUAL_TOL), tol=RESIDUAL_TOL,
              n_cells=int(len(units)), max_abs_residual=mx,
              worst_cell=(units.loc[units['residual'].abs().idxmax(),
                                    ['arch', 'domain', 'tau']].to_dict()))
    c4['pass'] = c4.pop('pass_')
    print(f'C4: max|residual| = {mx:.3e}  (tol {RESIDUAL_TOL:.0e})  -> '
          f'{"PASS" if c4["pass"] else "FAIL"}')

    if args.gate_only:
        print('\n[gate-only] 不落盘。τ=0.50 对照锚点 (仅展示, n 太小不作为闸门):')
        if os.path.exists(ANCHOR_CSV):
            ref = pd.read_csv(ANCHOR_CSV).set_index('arch')
            for dom, col in ((ID_DOMAIN, 'id_det_subst'), (OOD_DOMAIN, 'ood_det_subst')):
                g = units[np.isclose(units['tau'], TAU_ANCHOR)
                          & (units['domain'] == dom)]
                for _, r in g.iterrows():
                    e = float(ref.loc[r['arch'], col]) if r['arch'] in ref.index else float('nan')
                    print(f"  {dom:3s} {r['arch']:14s} ours={r['detection']:.4f} "
                          f"ref={e:.4f}")
        print(f'\n耗时 {time.time()-t0:.1f}s'); return 0

    # ---- 正式闸门 ----
    g2 = gate_g2(units)
    g4 = gate_g4(df, units)
    print(f'G2 锚点: {g2["status"]}  '
          f'T1 n={g2.get("n_t1")} fail={g2.get("n_strict_fail")} '
          f'max|Δ|={g2.get("max_abs_diff_t1")}  '
          f'T1b n={g2.get("n_t1b")} max|Δ|={g2.get("max_abs_diff_t1b")}  '
          f'T2 n={g2.get("n_t2")} fail={g2.get("n_loose_fail")} '
          f'max|Δ|={g2.get("max_abs_diff_t2")}')
    print(f'G4 双路: {g4["status"]}  max|Δ|={g4["max_abs_diff"]}')

    cc = conclusions(units)
    print(f'C1 (S_det>S_del): ID {cc["C1"][ID_DOMAIN]["support_b2"]} '
          f'({cc["C1"][ID_DOMAIN]["ratio"]:.2f}x)  '
          f'OOD {cc["C1"][OOD_DOMAIN]["support_b2"]} '
          f'({cc["C1"][OOD_DOMAIN]["ratio"]:.2f}x)')
    print(f'C2 (ID-OOD 检出差 > {C2_MIN_GAP}): {cc["C2"]["all_above"]}  '
          f'min gap={cc["C2"]["min_gap_observed"]:.4f}')
    for dom in (ID_DOMAIN, OOD_DOMAIN):
        print(f'C3 {dom}: min ρ in band={cc["C3"][dom]["min_rho_in_band"]:.4f}  '
              f'all_pass={cc["C3"][dom]["all_pass_in_band"]}')

    meta = dict(archs=[a for a, _ in jobs], cells=cells)
    if not args.no_write:
        payload = write_report(df, units, g2, g4, c4, cc, meta, time.time() - t0)
        print(f'\n已写: {OUT_PER_SAMPLE}\n已写: {OUT_UNITS}\n已写: {OUT_JSON}\n已写: {OUT_MD}')
        print('VERDICT: ' + json.dumps(payload['verdict'], ensure_ascii=False))
    print(f'耗时 {time.time()-t0:.1f}s')
    return 0 if (g2['status'] != 'ANCHOR_FAIL' and g4['status'] == 'PASS'
                 and c4['pass']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
