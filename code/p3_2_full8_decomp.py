# -*- coding: utf-8 -*-
"""
p3_2_full8_decomp.py -- 【论文二 P3-2】2^3 析因第 8 格并入 ETIS 检出/勾画分解表（8 行）

为什么单独一个脚本
------------------
`p3_1b_ablation_etis.py` 的 7 行产物（`decomp_ablation_etis.csv/.json`）是**冻结产物**，
不得改写。本脚本**逐行复用它的函数与常量**（`load_config`、`decompose.factorize`、
`FROZEN_THRESH`、`SUBSTANTIVE_THRESH`），只把配置清单从 7 扩到 8、读**新**汇总
`ablation_etis_summary_full8.json`、写到**新**路径 `*_full8.*`。
⇒ 口径零分叉：任何差异只可能来自数据（多了一格），不来自算法。

口径（与 P3-1b 完全一致，不另立标准）
------------------------------------
    OOD 侧检出 = recall_gt = I / |G|（**no-eps**，0 是精确 0）
    分解恒等式  E[Dice] = Detection x Delineation，残差 = sum_{undetected} Dice / n
    双判据并报（B6）：冻结 recall_gt > 0  ／ 实质 I >= 1 px（**OOD 侧严格等价**）
    闸门  mean(每样本 Dice) == summary dice/100，tol 1e-12

产物（全部新建，不动旧件）
------------------------
    03_results/stats/decomp_ablation_etis_full8.csv   (8 行)
    03_results/stats/decomp_ablation_etis_full8.json

用法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_2_full8_decomp.py
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

# 与 P3-1b 同源：函数与常量全部复用，保证口径零分叉
from p3_1b_ablation_etis import (          # noqa: E402
    CFGS as CFGS7, EXPECTED_N, PIXEL_THRESH, CSV_COLS, ETIS_ROOT_D, load_config)
from decompose import (EPS_OOD, FROZEN_THRESH, SUBSTANTIVE_THRESH, factorize)  # noqa: E402

# 第 8 格 = 三模块全开（P3-2 本次训练补出）
CFG8 = 'egm_dpa_msfa'
CFGS8 = list(CFGS7) + [CFG8]

SUMMARY_FULL8_D = os.path.join(ETIS_ROOT_D, 'ablation_etis_summary_full8.json')

E_ROOT = 'E:/paper2_ablation_reliability'
OUT_CSV = E_ROOT + '/03_results/stats/decomp_ablation_etis_full8.csv'
OUT_JSON = E_ROOT + '/03_results/stats/decomp_ablation_etis_full8.json'


def variance_shares(det: np.ndarray, delin: np.ndarray) -> dict:
    """同 P3-1b：var(log D) 三段分解。n=8 配置**非独立**，只作描述性归属。"""
    ok = (det > 0) & (delin > 0) & np.isfinite(det) & np.isfinite(delin)
    ld, le = np.log(det[ok]), np.log(delin[ok])
    n = int(ok.sum())
    if n < 3:
        return dict(n=n, note='有效点数不足')
    v_d, v_e = float(np.var(ld, ddof=0)), float(np.var(le, ddof=0))
    cov = float(np.cov(ld, le, ddof=0)[0, 1])
    v_tot = v_d + v_e + 2.0 * cov
    return dict(n=n, var_log_detection=v_d, var_log_delineation=v_e,
                cov2=2.0 * cov, var_log_dice=v_tot,
                share_detection=(v_d / v_tot if v_tot else float('nan')),
                share_delineation=(v_e / v_tot if v_tot else float('nan')),
                share_covariance=(2.0 * cov / v_tot if v_tot else float('nan')),
                note='n=8 配置非独立，描述性归属，不作推断')


def main(argv=None):
    ap = argparse.ArgumentParser(description='P3-2: 8-cell ETIS decomposition (2^3 full)')
    ap.add_argument('--out-csv', default=OUT_CSV)
    ap.add_argument('--out-json', default=OUT_JSON)
    args = ap.parse_args(argv)

    ref = {d['config']: d for d in json.load(open(SUMMARY_FULL8_D, encoding='utf-8'))}
    missing = [c for c in CFGS8 if c not in ref]
    if missing:
        raise SystemExit(f'full8 汇总缺配置: {missing}')

    rows, details = [], []
    for cfg in CFGS8:
        L = load_config(cfg)
        fz = factorize(L['dice'], L['recall_gt'], FROZEN_THRESH)
        fs = factorize(L['dice'], L['recall_gt'], SUBSTANTIVE_THRESH)

        r = ref.get(cfg)
        gate_delta = float(abs(fz['mean_dice'] - r['dice'] / 100.0))
        gate_status = 'MATCH' if gate_delta <= 1e-12 else 'DIFF'
        mods = r['modules']

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
            mean_dice=fz['mean_dice'], mean_dice_ref=r['dice'] / 100.0,
            gate_delta=gate_delta, gate_status=gate_status,
            dice_eps_check_max_abs_diff=float(np.max(np.abs(L['dice'] - L['dice_recomputed']))),
            detection=fz['detection'], delineation=fz['delineation'],
            product=fz['product'], residual=fz['residual'],
            predicted_residual=fz['predicted_residual'],
            n_undetected=fz['n_undetected'],
            detection_sub=fs['detection'], delineation_sub=fs['delineation'],
            product_sub=fs['product'], residual_sub=fs['residual'],
            n_undetected_sub=fs['n_undetected'],
            criteria_equivalent=bool(abs(fz['detection'] - fs['detection']) < 1e-15),
            mean_recall_gt=float(np.mean(L['recall_gt'])),
            pct_fully_missed=float((L['inter'] == 0).mean()),
        ))

    df = pd.DataFrame(rows)[CSV_COLS]
    vs = variance_shares(df['detection'].values.astype(float),
                         df['delineation'].values.astype(float))
    gate_ok = bool((df['gate_status'] == 'MATCH').all())

    # ---- 第 8 格 vs 既有 7 格（描述性对照，不作检验）----
    d8 = df[df['config'] == CFG8].iloc[0]
    d7 = df[df['config'] != CFG8]
    best7 = d7.loc[d7['detection'].idxmax()]
    best7_dice = d7.loc[d7['mean_dice'].idxmax()]
    compare8 = dict(
        detection_full8=float(d8['detection']),
        detection_best7=float(best7['detection']),
        detection_best7_config=str(best7['config']),
        delta_detection_vs_best7=float(d8['detection'] - best7['detection']),
        mean_dice_full8=float(d8['mean_dice']),
        mean_dice_best7=float(best7_dice['mean_dice']),
        mean_dice_best7_config=str(best7_dice['config']),
        delta_dice_vs_best7=float(d8['mean_dice'] - best7_dice['mean_dice']),
        residual_max_abs=float(df['residual'].abs().max()),
        note='描述性对照：n=8 配置非独立（共享协议），不作显著性推断',
    )

    payload = dict(
        segment='P3-2', mode='full_2^3_factorial_decomposition_on_etis',
        generated_at=datetime.now().isoformat(timespec='seconds'),
        script='02_code/analysis/p3_2_full8_decomp.py',
        protocol=dict(
            source=os.path.join(ETIS_ROOT_D, '<config>', 'predictions.npy'),
            n_per_config=EXPECTED_N, configs=CFGS8, eighth_cell=CFG8,
            pixel_threshold=PIXEL_THRESH,
            detection_metric='recall_gt = I / |G| (no eps) — OOD 侧约定',
            eps_ood=EPS_OOD,
            criterion_frozen=f'recall_gt > {FROZEN_THRESH}  (== I >= 1)',
            criterion_substantive=f'I >= 1 px  (== recall_gt > {SUBSTANTIVE_THRESH})',
            criteria_equivalence_note='OOD 侧 recall_gt 无 eps -> 两判据严格等价（与 ID 侧相反）',
            identity='E[Dice] = Detection x Delineation; residual = sum_{undetected} Dice / n',
            gate='mean(per-sample Dice) == ablation_etis_summary_full8.json dice/100 (tol 1e-12)',
            reuse='p3_1b_ablation_etis.load_config + decompose.factorize（口径零分叉）',
        ),
        gate=dict(all_match=gate_ok,
                  n_match=int((df['gate_status'] == 'MATCH').sum()), n_total=len(df),
                  max_delta=float(df['gate_delta'].max())),
        rows=details,
        h2_ablation_dimension=vs,
        full8_vs_7=compare8,
    )
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print('=' * 90)
    print(f"{'config':16s} {'EGM':>5s} {'DPA':>5s} {'MSFA':>5s} {'meanDice':>9s} "
          f"{'detect':>8s} {'delin':>8s} {'检出(实质)':>10s} {'全漏%':>7s} gate")
    for r in details:
        m = r['modules']
        print(f"{r['config']:16s} {str(m.get('EGM')):>5s} {str(m.get('DPA')):>5s} "
              f"{str(m.get('MSFA')):>5s} {r['mean_dice']:9.4f} {r['detection']:8.4f} "
              f"{r['delineation']:8.4f} {r['detection_sub']:10.4f} "
              f"{100*r['pct_fully_missed']:7.2f} {r['gate_status']}")
    print('-' * 90)
    print(f"闸门: {payload['gate']['n_match']}/{payload['gate']['n_total']} MATCH，"
          f"maxΔ={payload['gate']['max_delta']:.2e}")
    print(f"残差 max|residual| = {compare8['residual_max_abs']:.3e}")
    print(f"★ 第 8 格 vs 最佳 7 格：检出 {compare8['detection_full8']:.4f} vs "
          f"{compare8['detection_best7']:.4f} ({compare8['detection_best7_config']}) "
          f"Δ={compare8['delta_detection_vs_best7']:+.4f}")
    print(f"  mean Dice {compare8['mean_dice_full8']:.4f} vs "
          f"{compare8['mean_dice_best7']:.4f} ({compare8['mean_dice_best7_config']}) "
          f"Δ={compare8['delta_dice_vs_best7']:+.4f}")
    print(f"  方差归属(n=8，描述性): 检出 {vs.get('share_detection',float('nan')):.3%} / "
          f"勾画 {vs.get('share_delineation',float('nan')):.3%} / "
          f"2cov {vs.get('share_covariance',float('nan')):+.3%}")
    print(f"CSV  -> {args.out_csv}")
    print(f"JSON -> {args.out_json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
