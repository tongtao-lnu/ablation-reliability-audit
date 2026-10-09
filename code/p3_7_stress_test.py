# -*- coding: utf-8 -*-
"""
文件名: p3_7_stress_test.py
功能: 【论文二 P3-7】域内检出"关闸"压力测试 (in-domain gate-closing stress test)

口径 (v4.2.2 冻结, 边界条件 B6):
    对**分布内测试集** (D:/medical_segmentation/processed_data/test, 242 张, 352x352)
    施加受控劣化: 模糊 blur / 高斯噪声 noise / JPEG 压缩 jpeg / 低亮度 dark / 镜面高光 specular。
    每个劣化 3 个严重档 (level 1..3), 另加 clean (level 0) 对照。
    **模型与权重完全不动**, 只改输入 -> 重算检出率。纯推理, 零训练。

    检出判据 (必须并报两条):
      冻结 frozen      : recall_metric > 0
                         recall_metric = (I + 1e-8) / (|G| + 1e-8)   (ID 侧度量, utils/metrics.py)
                         -> 因平滑常数 eps = 1e-8 > 0, 分子恒 > 0, 故**恒为 1.000, 与输入无关**
      实质 substantive : I >= 1 px   (等价 recall_metric > 1e-6)

    两种结果都合格, 只有"不报告"不合格:
      ① 实质检出随劣化单调下降 -> 检出因子是真实可变量, 强化 L1;
      ② 始终 ~= 1.000           -> 如实报告, L1 改为"该基准下检出不可分辨", 范围缩小但不推翻。

设计 (单一真源, 不重写协议):
    - 模型工厂 / 权重载入   复用 p31_cross_eval.build_model / load_weights
    - 架构清单 + 权重目录   复用 p31a_ood_idtrain.build_jobs (13 baseline + EGAUNet = 14)
    - 必须 `with autocast():` —— 既有 ID/OOD 产物均在该设置下产生
    - 张量构造与 utils/dataset.py::MedicalDataset(split='test', augment=False) 逐字对齐
      (img (H,W,3) float32 -> transpose(2,0,1); mask (H,W) -> [None,:,:]); 测试集无增强

G4 复现闸门 (硬要求):
    clean 档的**实质检出**必须复现 03_results/stats/p31a_units_n14.csv 的 `id_det_subst`
    (同一批 242 张, 同一权重, 同一判据); 差异逐架构报出, 不吻合即标 GATE_FAIL 供人工判断。

断点续跑:
    每架构跑完立刻写 out_root/partial/<arch>.csv; 重跑时默认跳过已存在者 (--force 覆盖)。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    # 冒烟 (1 架构, 8 张)
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_7_stress_test.py \
        --archs EGAUNet --limit 8 --out-root 03_results/raw/stress_test_smoke --no-hd95
    # 全量 14 架构 x 242 张 x 16 条件 (不写 partial 汇总 -> stats)
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_7_stress_test.py \
        --out-root 03_results/raw/stress_test --out 03_results/stats/stress_test.json
"""
import argparse
import json
import os
import sys
import zlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# import 副作用: chdir 到 D:/medical_segmentation (PROJECT_ROOT); 故相对路径按 D 盘解析,
# E 盘一律走 X.eabs
import p31_cross_eval as X
import p31a_ood_idtrain as A
from decompose import FROZEN_THRESH, SUBSTANTIVE_THRESH, EPS_ID

# --------------------------------------------------------------------------- #
# 冻结路径与常数
# --------------------------------------------------------------------------- #
ID_TEST_DIR = 'processed_data/test'                 # 相对 PROJECT_ROOT(D:)
ID_IMG_DIR = os.path.join(ID_TEST_DIR, 'images')
ID_MSK_DIR = os.path.join(ID_TEST_DIR, 'masks')
PIXEL_THRESH = 0.5                                  # 二值化阈值 (与 utils/metrics.py 一致)
UNITS_CSV = '03_results/stats/p31a_units_n14.csv'   # G4 对照 (E 盘相对)

# 劣化类型 -> 每档参数（**跑前冻结**, 见执行报告 §冻结劣化阶梯）
DEGRADATIONS = {
    'blur':     {1: 1.5, 2: 3.0, 3: 6.0},           # 高斯 sigma (px), 离焦/运动模糊
    'noise':    {1: 0.05, 2: 0.10, 3: 0.20},        # 加性高斯噪声 std (像素域 [0,1])
    'jpeg':     {1: 30, 2: 15, 3: 8},               # JPEG 质量 (越低越差)
    'dark':     {1: 0.60, 2: 0.40, 3: 0.25},        # 亮度线性缩放 (低照度)
    'specular': {1: 1, 2: 3, 3: 6},                 # 镜面高光斑块数 (饱和白, 局部遮挡)
}
# 镜面高光斑块几何 (固定)
SPEC_RADIUS = 30.0
SPEC_SIGMA = 10.0
SPEC_AMP = 0.9
LEVELS = (1, 2, 3)
CONDITIONS = [('clean', 0)] + [(t, l) for t in DEGRADATIONS for l in LEVELS]


# --------------------------------------------------------------------------- #
# 受控劣化 (全部在 [0,1] float32 上做, 出口 clip; 由 (name,type,level) 稳定播种)
# --------------------------------------------------------------------------- #
def _rng(name, dtype, level):
    seed = zlib.crc32(f'{name}|{dtype}|{level}'.encode('utf-8')) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def degrade(img, dtype, level, name):
    """img: (H,W,3) float32 in [0,1] -> 劣化后同形同域。clean(level=0) 原样返回。"""
    if dtype == 'clean' or level == 0:
        return img

    p = DEGRADATIONS[dtype][level]

    if dtype == 'blur':
        sigma = float(p)
        k = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT101)

    if dtype == 'noise':
        rng = _rng(name, dtype, level)
        return img + rng.normal(0.0, float(p), img.shape).astype(np.float32)

    if dtype == 'jpeg':
        q = int(p)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode('.jpg', u8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError(f'JPEG encode failed: {name} q={q}')
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec.astype(np.float32) / 255.0

    if dtype == 'dark':
        return img * float(p)

    if dtype == 'specular':
        rng = _rng(name, dtype, level)
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        out = img.copy()
        rad, sig, amp = SPEC_RADIUS, SPEC_SIGMA, SPEC_AMP
        for _ in range(int(p)):
            cy = int(rng.integers(int(rad) + 1, H - int(rad) - 1))
            cx = int(rng.integers(int(rad) + 1, W - int(rad) - 1))
            blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                / (2.0 * sig ** 2)).astype(np.float32)
            out = out + blob[:, :, None]
        return out

    raise ValueError(f'未知劣化类型: {dtype}')


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_id_test(limit=None):
    """返回 [(name, img (H,W,3) f32 [0,1], gt_bool (H,W)), ...]，按文件名排序。"""
    files = sorted(f for f in os.listdir(ID_IMG_DIR) if f.endswith('.npy'))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        img = np.load(os.path.join(ID_IMG_DIR, f)).astype(np.float32)
        msk = np.load(os.path.join(ID_MSK_DIR, f)).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        out.append((f, img, (msk > PIXEL_THRESH)))
    return out


# --------------------------------------------------------------------------- #
# 指标 (逐字对齐 utils/metrics.py::SegmentationMetrics.update)
# --------------------------------------------------------------------------- #
def sample_metrics(prob, gt_bool):
    """prob: (H,W) float sigmoid 概率; gt_bool: (H,W) bool。
    返回 (inter, gt_area, pred_area, recall_metric, dice, det_frozen, det_subst)。"""
    p = (prob > PIXEL_THRESH)
    inter = int(np.logical_and(p, gt_bool).sum())
    gs = int(gt_bool.sum())
    ps = int(p.sum())
    recall_metric = (inter + EPS_ID) / (gs + EPS_ID)      # ID 侧度量实现 (eps=1e-8)
    dice = (2.0 * inter + EPS_ID) / (ps + gs + EPS_ID)
    det_frozen = bool(recall_metric > FROZEN_THRESH)      # 恒 True (eps 构造)
    det_subst = bool(inter >= 1)                          # 实质: I >= 1 px
    return inter, gs, ps, float(recall_metric), float(dice), det_frozen, det_subst


# --------------------------------------------------------------------------- #
# 单架构评估
# --------------------------------------------------------------------------- #
def eval_arch(arch, ckpt_dir, device, data, conditions):
    wpath = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(wpath):
        return None, None

    model = X.build_model(arch)
    fmt, miss, unexp = X.load_weights(model, wpath, device)
    if len(miss) or len(unexp):
        # 与 P3-1 同铁律: 键不匹配会静默全零预测, 必须硬拦
        raise RuntimeError(
            f'{arch}: 权重键不匹配 (missing={len(miss)}, unexpected={len(unexp)}) -> 拒绝运行。'
            f' path={wpath}  fmt={fmt}')
    model = model.to(device).eval()

    rows = []
    with torch.no_grad():
        for name, img, gt in tqdm(data, desc=f'{arch}', ncols=88, leave=False):
            gt_t = torch.from_numpy(gt[None, None].astype(np.float32)).to(device)  # 仅占位, 下面用 numpy gt
            for dtype, level in conditions:
                x = clip01(degrade(img, dtype, level, name))
                xt = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
                with autocast():
                    out = model(xt)
                    if isinstance(out, tuple):
                        out = out[0]
                prob = torch.sigmoid(out.float())[0, 0].cpu().numpy()
                inter, gs, ps, rec, dice, df, ds = sample_metrics(prob, gt)
                rows.append(dict(arch=arch, name=name, degradation=dtype, level=level,
                                 gt_area=gs, pred_area=ps, inter=inter,
                                 recall_metric=rec, dice=dice,
                                 det_frozen=int(df), det_subst=int(ds),
                                 weight_format=fmt))
            del gt_t

    del model
    torch.cuda.empty_cache()
    return rows, fmt


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summarize(df):
    g = (df.groupby(['arch', 'degradation', 'level'], as_index=False)
           .agg(n=('name', 'size'),
                k_frozen=('det_frozen', lambda s: int((s == 0).sum())),
                k_subst=('det_subst', lambda s: int((s == 0).sum())),
                det_frozen=('det_frozen', 'mean'),
                det_subst=('det_subst', 'mean'),
                mean_dice=('dice', 'mean'),
                mean_recall=('recall_metric', 'mean'),
                mean_inter=('inter', 'mean')))
    return g


def clean_gate(df, units_csv):
    """G4: clean 档实质检出 vs p31a_units_n14.csv 的 id_det_subst。"""
    path = units_csv if os.path.isabs(units_csv) else X.eabs(units_csv)
    if not os.path.exists(path):
        return None, None
    ref = pd.read_csv(path)
    ref_map = dict(zip(ref['arch'], ref['id_det_subst']))
    clean = df[(df.degradation == 'clean')]
    got = clean.groupby('arch')['det_subst'].mean().to_dict()
    rec = []
    for arch in sorted(got):
        exp = ref_map.get(arch, float('nan'))
        rec.append(dict(arch=arch, ours=got[arch], ref=float(exp),
                        abs_diff=abs(got[arch] - float(exp)) if exp == exp else float('nan'),
                        ours_k=int(clean[(clean.arch == arch) & (clean.det_subst == 0)].shape[0]),
                        status=('MATCH' if (exp == exp and abs(got[arch] - exp) <= 1e-9)
                                else 'DIFF')))
    return rec, dict(n_match=sum(1 for r in rec if r['status'] == 'MATCH'), n_total=len(rec))


def monotonicity(df):
    """按 (arch, 劣化) 看实质检出是否随档位单调不增。

    曲线 = [clean, L1, L2, L3]；clean 一律取 degradation=='clean' 行作基线
    （非 clean 劣化本身没有 level=0 行，勿用同类型 level 0 求均值）。
    """
    out = []
    for arch in sorted(df.arch.unique()):
        base = float(df[(df.arch == arch) & (df.degradation == 'clean')]['det_subst'].mean())
        for dtype in DEGRADATIONS:
            sub = df[(df.arch == arch) & (df.degradation == dtype)]
            if sub.empty:
                continue
            v = [base] + [float(sub[sub.level == l]['det_subst'].mean())
                          for l in LEVELS if not sub[sub.level == l].empty]
            if len(v) != 1 + len(LEVELS):
                continue
            drops = [v[i + 1] - v[i] for i in range(len(v) - 1)]
            out.append(dict(arch=arch, degradation=dtype,
                            curve=v, clean=v[0], total_drop=v[0] - v[-1],
                            monotone_nonincreasing=bool(all(d <= 1e-12 for d in drops)),
                            any_drop=bool(any(d < -1e-12 for d in drops))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', nargs='+', default=None, help='子集，默认全部 14 个')
    ap.add_argument('--limit', type=int, default=None, help='只用前 N 张图（冒烟）')
    ap.add_argument('--types', nargs='+', default=None,
                    choices=['clean'] + list(DEGRADATIONS), help='只用这些劣化类型')
    ap.add_argument('--out-root', default='03_results/raw/stress_test')
    ap.add_argument('--out', default=None, help='汇总 JSON（相对路径按 E 盘解析）')
    ap.add_argument('--force', action='store_true', help='忽略 partial 缓存，重跑')
    ap.add_argument('--no-hd95', action='store_true', help='占位（本脚本不算 HD95）')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}')

    conditions = CONDITIONS
    if args.types:
        want = set(args.types)
        conditions = [(t, l) for (t, l) in CONDITIONS if t in want]

    jobs = A.build_jobs(args.archs)
    data = load_id_test(args.limit)
    print(f'archs={len(jobs)}  images={len(data)}  conditions={len(conditions)}')
    print('conditions=' + ', '.join(f'{t}@{l}' for t, l in conditions))

    out_root = X.eabs(args.out_root)
    part_dir = os.path.join(out_root, 'partial')
    os.makedirs(part_dir, exist_ok=True)

    all_rows, summaries = [], []
    for arch, ckpt_dir in jobs:
        pp = os.path.join(part_dir, f'{arch}.csv')
        if os.path.exists(pp) and not args.force:
            print(f'>>> {arch}  [跳过] 命中 partial: {pp}')
            d = pd.read_csv(pp)
            all_rows.append(d)
            summaries.append(dict(arch=arch, ckpt_dir=ckpt_dir, status='CACHED',
                                  n_rows=len(d)))
            continue

        print(f'\n>>> {arch}   <- {ckpt_dir}/best_model.pth')
        rows, fmt = eval_arch(arch, ckpt_dir, device, data, conditions)
        if rows is None:
            print('    [跳过] 权重缺失')
            summaries.append(dict(arch=arch, ckpt_dir=ckpt_dir, status='NO_WEIGHT'))
            continue

        d = pd.DataFrame(rows)
        d.to_csv(pp, index=False)
        all_rows.append(d)
        c = d[d.degradation == 'clean']
        s = dict(arch=arch, ckpt_dir=ckpt_dir, status='OK', weight_format=fmt,
                 n_rows=len(d), n_conditions=len(conditions),
                 clean_det_frozen=float(c.det_frozen.mean()),
                 clean_det_subst=float(c.det_subst.mean()),
                 clean_k_subst=int((c.det_subst == 0).sum()),
                 clean_dice=float(c.dice.mean()))
        summaries.append(s)
        print(f"    rows={len(d)}  clean: det_frozen={s['clean_det_frozen']:.4f}  "
              f"det_subst={s['clean_det_subst']:.6f}  k_subst={s['clean_k_subst']}  "
              f"keys(miss/unexp)=0/0")

    df = pd.concat(all_rows, ignore_index=True)

    # 全量落盘
    csv_all = os.path.join(out_root, 'per_sample.csv')
    df.to_csv(csv_all, index=False)
    print(f'\n已写: {csv_all}  ({len(df)} 行)')

    summary = summarize(df)
    summary.to_csv(os.path.join(out_root, 'summary_by_condition.csv'), index=False)

    gate, gate_meta = clean_gate(df, UNITS_CSV)
    mono = monotonicity(df)

    payload = dict(
        segment='P3-7', mode='in_domain_gate_closing_stress_test',
        protocol=dict(
            data=ID_TEST_DIR, n_images=int(df.name.nunique()),
            n_archs=int(df.arch.nunique()), archs=sorted(df.arch.unique()),
            conditions=[f'{t}@{l}' for t, l in CONDITIONS],
            degradations={k: v for k, v in DEGRADATIONS.items()},
            weights='ID 联合训练 (experiments/baseline/<arch> | outputs/checkpoints/ega_unet)，冻结不动',
            pixel_threshold=PIXEL_THRESH, eps_id=EPS_ID,
            criterion_frozen=f'recall_metric > {FROZEN_THRESH}   (因 eps>0 恒为 1.000, 与输入无关)',
            criterion_substantive=f'I >= 1 px  (等价 recall_metric > {SUBSTANTIVE_THRESH})',
            reuse=['p31_cross_eval.build_model', 'p31_cross_eval.load_weights',
                   'p31a_ood_idtrain.build_jobs', 'autocast', 'utils/metrics.py eps=1e-8'],
        ),
        summaries=summaries,
        clean_gate=dict(meta=gate_meta, rows=gate),
        monotonicity=mono,
        curves=summary.to_dict(orient='records'),
    )
    if args.out:
        outp = X.eabs(args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'已写: {outp}')

    # ---------------- 控制台速览 ----------------
    print('\n' + '=' * 96)
    print('[G4] clean 档实质检出 vs p31a_units_n14.csv')
    if gate:
        for r in gate:
            print(f"  {r['arch']:<16} ours={r['ours']:.6f}  ref={r['ref']:.6f}  "
                  f"|d|={r['abs_diff']:.2e}  k={r['ours_k']}  {r['status']}")
        print(f"  -> {gate_meta['n_match']}/{gate_meta['n_total']} MATCH")
    else:
        print('  (对照表缺失)')

    print('\n' + '=' * 96)
    print('冻结判据 (recall_metric > 0), 全部条件:')
    print(df.groupby(['degradation', 'level'])['det_frozen'].mean().to_string())
    print('\n实质判据 (I >= 1 px), 各劣化 x 档位 (跨架构均值):')
    piv = summary.pivot_table(index='degradation', columns='level', values='det_subst',
                              aggfunc='mean')
    print(piv.to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n按架构 x 劣化的实质检出曲线 (clean -> L1 -> L2 -> L3):')
    for m in sorted(mono, key=lambda z: -z['total_drop']):
        c = m['curve']
        print(f"  {m['arch']:<16} {m['degradation']:<9} "
              f"{c[0]:.4f} -> {c[1]:.4f} -> {c[2]:.4f} -> {c[3]:.4f}   "
              f"drop={m['total_drop']:+.4f}  {'monotone' if m['monotone_nonincreasing'] else 'NOT-monotone'}")


if __name__ == '__main__':
    main()
