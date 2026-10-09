import pandas as pd, numpy as np, os
from scipy import stats

pd.set_option('display.width', 220)
ROOT = 'D:/medical_segmentation'
CFG = ['baseline', 'egm_only', 'dpa_only', 'msfa_only', 'egm_dpa', 'egm_msfa', 'dpa_msfa']
ARCH = ['UNet', 'AttentionUNet', 'TransUNet', 'EGAUNet']

d = pd.read_csv(r'C:/Users/Lenovo/WorkBuddy/2026-09-12-19-51-30/verify/out/paper2_per_sample.csv')

# ---- OOD 分解表 ----
rec = []
for (dr, ar), g in d.groupby(['dir', 'arch']):
    det = (g['recall_gt'] > 0).mean()
    dl  = g.loc[g['recall_gt'] > 0, 'dice'].mean()
    rec.append(dict(dir=dr, arch=ar, dice=g['dice'].mean(), det=det, dl=dl))
O = pd.DataFrame(rec)

print('=' * 100)
print('D. OOD 的方差归属 (log 尺度):  log Dice = log Detection + log Delineation')
print('=' * 100)
ld, lt, ll = np.log(O['dice']), np.log(O['det']), np.log(O['dl'])
vt, vl = lt.var(ddof=1), ll.var(ddof=1)
cv = np.cov(lt, ll, ddof=1)[0, 1]
tot = vt + vl + 2 * cv
print(f'var(logDice)={ld.var(ddof=1):.4f}  var(logDet)={vt:.4f} ({vt/tot*100:5.1f}%)  '
      f'var(logDel)={vl:.4f} ({vl/tot*100:5.1f}%)  2cov={2*cv:.4f} ({2*cv/tot*100:5.1f}%)')
print(f'Detection  范围: {O["det"].min():.3f} - {O["det"].max():.3f}  (极差 {O["det"].max()-O["det"].min():.3f})')
print(f'Delineation范围: {O["dl"].min():.3f} - {O["dl"].max():.3f}  (极差 {O["dl"].max()-O["dl"].min():.3f})')
print(f'按 log 尺度极差: Detection {np.log(O["det"].max())-np.log(O["det"].min()):.3f} | '
      f'Delineation {np.log(O["dl"].max())-np.log(O["dl"].min()):.3f}')

# ---- ID 分解 (7 配置 + 4 架构) ----
print()
print('=' * 100)
print('E. 分布内 (ID) 的检出率 —— 是否退化为 1.000')
print('=' * 100)
print(f"{'set':<28}{'n':>5}{'Detect':>10}{'Deline':>10}")
for c in CFG:
    f = f'{ROOT}/experiments/ablation/{c}/sample_metrics.csv'
    g = pd.read_csv(f)
    print(f"{'ablation/' + c:<28}{len(g):>5}{(g['Recall'] > 0).mean():>10.3f}{g.loc[g['Recall'] > 0, 'Dice'].mean()*100:>10.2f}")
IDPATH = {a: f'{ROOT}/experiments/baseline/{a}/sample_metrics.csv' for a in ARCH}
IDPATH['EGAUNet'] = f'{ROOT}/results/test_results/sample_metrics.csv'
for a in ARCH:
    f = IDPATH[a]
    if not os.path.exists(f):
        print(f"{'baseline/' + a:<28}  MISSING"); continue
    g = pd.read_csv(f)
    print(f"{'baseline/' + a:<28}{len(g):>5}{(g['Recall'] > 0).mean():>10.3f}{g.loc[g['Recall'] > 0, 'Dice'].mean()*100:>10.2f}")

# 13 个 baseline 架构的 ID 检出率 —— 是否全部退化为 1.000
print()
print('--- 全部 13 个已训练 baseline 的 ID 检出率 ---')
alld = []
for fn in sorted(os.listdir(f'{ROOT}/experiments/baseline')):
    f = f'{ROOT}/experiments/baseline/{fn}/sample_metrics.csv'
    if not os.path.exists(f):
        continue
    g = pd.read_csv(f)
    rc = next((c for c in g.columns if c.lower().startswith('recall') or c.lower() == 'recall'), None)
    if rc is None:
        print(f'  [skip {fn}] cols={list(g.columns)}')
        continue
    alld.append(((g[rc] > 0).mean(), len(g)))
print(f'共 {len(alld)} 个架构, 检出率 min={min(a[0] for a in alld):.4f} max={max(a[0] for a in alld):.4f}, n={alld[0][1]}')
# ---- 对齐性检验: ID 的 Delineation 能不能预测 OOD 的 Detection ----
print()
print('=' * 100)
print('F. 对齐性: ID 上测到的能力  vs  域外真正决定成败的能力')
print('=' * 100)
rows = []
for a in ARCH:
    f = IDPATH[a]
    g = pd.read_csv(f)
    id_det = (g['Recall'] > 0).mean()
    id_dl  = g.loc[g['Recall'] > 0, 'Dice'].mean()
    sub = O[O['arch'] == a]
    rows.append(dict(arch=a, id_dice=g['Dice'].mean(), id_det=id_det, id_dl=id_dl,
                     ood_det=sub['det'].mean(), ood_dl=sub['dl'].mean(), ood_dice=sub['dice'].mean()))
A = pd.DataFrame(rows).sort_values('id_dl', ascending=False)
print(A.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
if len(A) >= 4:
    r1 = stats.spearmanr(A['id_dl'], A['ood_det'])
    r2 = stats.spearmanr(A['id_dl'], A['ood_dice'])
    r3 = stats.spearmanr(A['ood_det'], A['ood_dice'])
    print(f"\nSpearman(ID Delineation, OOD Detection) = {r1[0]:+.3f}  p={r1[1]:.4f}   <-- 应接近 0/负 = 错位")
    print(f"Spearman(ID Delineation, OOD Dice)      = {r2[0]:+.3f}  p={r2[1]:.4f}")
    print(f"Spearman(OOD Detection,  OOD Dice)      = {r3[0]:+.3f}  p={r3[1]:.4f}   <-- 应接近 +1 = 决定成败")
