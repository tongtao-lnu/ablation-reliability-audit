# -*- coding: utf-8 -*-
"""
文件名: p3_1b_ablation_etis.py
功能: 【论文二 P3-1b】给 7 个消融配置的 **ETIS 跨域预测** 算检出/勾画分解（消融维度上的 H2）

数据（只读，D 盘原位）:
    D:/medical_segmentation/results_ablation_etis/<config>/predictions.npy
        -> object ndarray, 196 个 dict, keys = ['name','image','mask','pred','dice','iou','hd95']
           mask: (352,352) float32 {0,1}  GT
           pred: (352,352) bool           二值预测
           dice: float, 该样本 Dice（**含平滑常数 eps=1e-6**，空预测也 ~1e-10 而非 0）
    D:/medical_segmentation/results_ablation_etis/ablation_etis_summary.json  (架构级均值对照)

口径（与 decompose.py 完全一致，不另立标准）:
    - OOD 侧检出度量取 **recall_gt = I / |G|（no-eps）** -> 0 是精确 0
      （与 P2-1 的 OOD 侧约定一致；**不使用** predictions.npy 里带 eps 的 dice 作检出判据）
    - 分解恒等式  E[Dice] = Detection × Delineation，残差 = Σ_{未检出} Dice / n（可解析归因）
    - 双判据并报（B6）:
        冻结 frozen      : recall_gt > 0      (== I >= 1)
        实质 substantive : I >= 1 px          (== recall_gt > 1e-6)
      ⚠️ OOD 侧 recall_gt 无 eps，故 **两判据在本段严格等价**，必须显式说明（与 ID 侧不同！）

闸门 (G6, 跑前冻结):
    mean(每样本 Dice) 必须复现 ablation_etis_summary.json 的 dice（百分比 / 100），tol 1e-12
    （预验证：7/7 配置 Δ = 0）

输出:
    03_results/stats/decomp_ablation_etis.csv   (7 行, 主产物)
    03_results/stats/decomp_ablation_etis.json

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_1b_ablation_etis.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from decompose import (EPS_OOD, FROZEN_THRESH, SUBSTANTIVE_THRESH, factorize)

ETIS_ROOT_D = 'D:/medical_segmentation/results_ablation_etis'
SUMMARY_JSON_D = os.path.join(ETIS_ROOT_D, 'ablation_etis_summary.json')

CFGS = ['baseline', 'egm_only', 'dpa_only', 'msfa_only',
        'egm_dpa', 'egm_msfa', 'dpa_msfa']
EXPECTED_N = 196
PIXEL_THRESH = 0.5

OUT_CSV = 'E:/paper2_ablation_reliability/03_results/stats/decomp_ablation_etis.csv'
OUT_JSON = 'E:/paper2_ablation_reliability/03_results/stats/decomp_ablation_etis.json'

CSV_COLS = ['config', 'EGM', 'DPA', 'MSFA', 'n', 'mean_dice', 'detection',
            'delineation', 'product', 'residual', 'n_undetected',
            'detection_sub', 'delineation_sub', 'n_undetected_sub',
            'mean_recall_gt', 'gate_delta', 'gate_status']


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #
def load_config(cfg: str) -> dict:
    """读一个配置的 predictions.npy，返回逐样本数组。"""
    p = os.path.join(ETIS_ROOT_D, cfg, 'predictions.npy')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{cfg}: 预测缺失 {p}')
    arr = np.load(p, allow_pickle=True)
    if len(arr) != EXPECTED_N:
        raise ValueError(f'{cfg}: {len(arr)} 样本, 期望 {EXPECTED_N}')

    names, dice_stored, inter, gs, ps = [], [], [], [], []
    for d in arr:
        g = (np.asarray(d['mask']) > PIXEL_THRESH)
        pr = np.asarray(d['pred']).astype(bool)
        if g.shape != pr.shape:
            raise ValueError(f"{cfg}/{d['name']}: mask {g.shape} vs pred {pr.shape}")
        names.append(str(d['name']))
        dice_stored.append(float(d['dice']))
        inter.append(int(np.logical_and(pr, g).sum()))
        gs.append(int(g.sum()))
        ps.append(int(pr.sum()))

    inter = np.asarray(inter, dtype=float)
    gs = np.asarray(gs, dtype=float)
    ps = np.asarray(ps, dtype=float)

    # OOD 侧检出度量: no-eps, 0 是精确 0
    with np.errstate(divide='ignore', invalid='ignore'):
        recall_gt = np.where(gs > 0, inter / gs, np.nan)
    if np.isnan(recall_gt).any():
        raise ValueError(f'{cfg}: 出现 |G| = 0 的样本（违反 B3），需人工处理')

    # 复核 predictions.npy 内 dice 的 eps 口径
    dice_recomputed = (2.0 * inter + EPS_OOD) / (ps + gs + EPS_OOD)

    return dict(cfg=cfg, names=names,
                dice=np.asarray(dice_stored, dtype=float),
                dice_recomputed=dice_recomputed,
                recall_gt=recall_gt, inter=inter, gs=gs, ps=ps)


# --------------------------------------------------------------------------- #
# 消融维度的方差归属（补充产物；与 P2-2 的三段分解同式）
# --------------------------------------------------------------------------- #
def variance_shares(det: np.ndarray, delin: np.ndarray) -> dict:
    """var(log D) = var(log det) + var(log delin) + 2 cov(log det, log delin)。

    注意：n = 7 个配置，**非独立**（共享训练协议），只作描述性归属，不作推断。
    """
    ok = (det > 0) & (delin > 0) & np.isfinite(det) & np.isfinite(delin)
    ld, le = np.log(det[ok]), np.log(delin[ok])
    n = int(ok.sum())
    if n < 3:
        return dict(n=n, note='有效点数不足')
    v_d = float(np.var(ld, ddof=0))
    v_e = float(np.var(le, ddof=0))
    cov = float(np.cov(ld, le, ddof=0)[0, 1])
    v_tot = v_d + v_e + 2.0 * cov
    return dict(n=n, var_log_detection=v_d, var_log_delineation=v_e,
                cov2=2.0 * cov, var_log_dice=v_tot,
                share_detection=(v_d / v_tot if v_tot else float('nan')),
                share_delineation=(v_e / v_tot if v_tot else float('nan')),
                share_covariance=(2.0 * cov / v_tot if v_tot else float('nan')),
                note='n=7 配置非独立，描述性归属，不作推断')


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description='P3-1b: ETIS ablation detection/delineation decomposition')
    ap.add_argument('--out-csv', default=OUT_CSV)
    ap.add_argument('--out-json', default=OUT_JSON)
    args = ap.parse_args(argv)

    ref = {d['config']: d for d in json.load(open(SUMMARY_JSON_D, encoding='utf-8'))}

    rows, details = [], []
    for cfg in CFGS:
        L = load_config(cfg)
        fz = factorize(L['dice'], L['recall_gt'], FROZEN_THRESH)
        fs = factorize(L['dice'], L['recall_gt'], SUBSTANTIVE_THRESH)

        # 闸门: 每样本 Dice 的均值 vs 架构级 results
        r = ref.get(cfg)
        gate_delta = float(abs(fz['mean_dice'] - r['dice'] / 100.0)) if r else float('nan')
        gate_status = 'MATCH' if (r and gate_delta <= 1e-12) else 'DIFF'
        mods = r['modules'] if r else {}

        rows.append(dict(
            config=cfg, EGM=mods.get('EGM'), DPA=mods.get('DPA'), MSFA=mods.get('MSFA'),
            n=fz['n'], mean_dice=fz['mean_dice'],
            detection=fz['detection'], delineation=fz['delineation'],
            product=fz['product'], residual=fz['residual'],
            n_undetected=fz['n_undetected'],
            detection_sub=fs['detection'], delineation_sub=fs['delineation'],
            n_undetected_sub=fs['n_undetected'],
            mean_recall_gt=float(np.mean(L['recall_gt'])),
            gate_delta=gate_delta, gate_status=gate_status))

        details.append(dict(
            config=cfg, modules=mods, n=fz['n'],
            mean_dice=fz['mean_dice'], mean_dice_ref=(r['dice'] / 100.0 if r else None),
            gate_delta=gate_delta, gate_status=gate_status,
            dice_eps_check_max_abs_diff=float(np.max(np.abs(L['dice'] - L['dice_recomputed']))),
            detection=fz['detection'], delineation=fz['delineation'],
            product=fz['product'], residual=fz['residual'],
            predicted_residual=fz['predicted_residual'],
            n_undetected=fz['n_undetected'],
            detection_sub=fs['detection'], delineation_sub=fs['delineation'],
            product_sub=fs['product'], residual_sub=fs['residual'],
            n_undetected_sub=fs['n_undetected_sub'] if 'n_undetected_sub' in fs else fs['n_undetected'],
            criteria_equivalent=bool(abs(fz['detection'] - fs['detection']) < 1e-15),
            mean_recall_gt=float(np.mean(L['recall_gt'])),
            pct_fully_missed=float((L['inter'] == 0).mean()),
        ))

    df = pd.DataFrame(rows)[CSV_COLS]

    # 消融维度的方差归属（冻结判据 = 实质判据，二者等价）
    det = df['detection'].values.astype(float)
    delin = df['delineation'].values.astype(float)
    vs = variance_shares(det, delin)

    gate_ok = bool((df['gate_status'] == 'MATCH').all())
    payload = dict(
        segment='P3-1b', mode='ablation_dimension_decomposition_on_etis',
        generated_at=datetime.now().isoformat(timespec='seconds'),
        script='02_code/analysis/p3_1b_ablation_etis.py',
        protocol=dict(
            source=os.path.join(ETIS_ROOT_D, '<config>', 'predictions.npy'),
            n_per_config=EXPECTED_N, configs=CFGS,
            pixel_threshold=PIXEL_THRESH,
            detection_metric='recall_gt = I / |G| (no eps) — OOD 侧约定',
            eps_ood=EPS_OOD,
            criterion_frozen=f'recall_gt > {FROZEN_THRESH}  (== I >= 1)',
            criterion_substantive=f'I >= 1 px  (== recall_gt > {SUBSTANTIVE_THRESH})',
            criteria_equivalence_note='OOD 侧 recall_gt 无 eps -> 两判据严格等价（与 ID 侧相反）',
            identity='E[Dice] = Detection x Delineation; residual = sum_{undetected} Dice / n',
            gate='mean(per-sample Dice) == ablation_etis_summary.json dice/100 (tol 1e-12)',
            reuse=['decompose.factorize', 'decompose.FROZEN_THRESH/SUBSTANTIVE_THRESH'],
        ),
        gate=dict(all_match=gate_ok,
                  n_match=int((df['gate_status'] == 'MATCH').sum()), n_total=len(df),
                  max_delta=float(df['gate_delta'].max())),
        rows=details,
        h2_ablation_dimension=vs,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_json, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---------------- 控制台速览 ----------------
    pd.set_option('display.width', 220)
    show = df.copy()
    for c in ['mean_dice', 'detection', 'delineation', 'product']:
        show[c] = show[c].map(lambda v: f'{v:.4f}')
    print('=' * 110)
    print('P3-1b  ETIS 消融维度分解 (检出判据: recall_gt > 0, no-eps)')
    print('=' * 110)
    print(show[['config', 'EGM', 'DPA', 'MSFA', 'n', 'mean_dice', 'detection',
                'delineation', 'product', 'n_undetected', 'detection_sub']].to_string(index=False))
    print()
    print('[闸门] mean(Dice) vs ablation_etis_summary.json：'
          f"{int((df['gate_status'] == 'MATCH').sum())}/{len(df)} MATCH，"
          f"max|Δ| = {df['gate_delta'].max():.3e}")
    print(f"[等价性] 两判据在 OOD 侧等价: "
          f"{all(d['criteria_equivalent'] for d in details)}")
    print()
    print('消融维度方差归属 var(log Dice)（n=7 配置，非独立，描述性）:')
    print(f"  检出 {vs.get('share_detection', float('nan')):.3f}  "
          f"勾画 {vs.get('share_delineation', float('nan')):.3f}  "
          f"协方差 {vs.get('share_covariance', float('nan')):.3f}")
    print()
    print(f'已写: {args.out_csv}')
    print(f'已写: {args.out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
