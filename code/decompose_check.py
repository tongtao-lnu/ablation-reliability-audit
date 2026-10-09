import pandas as pd, numpy as np

pd.set_option('display.width', 200)

# ---------- 1) 跨域(分布外): 用 recall_gt>0 作为"检出"定义, 验证恒等式 ----------
d = pd.read_csv(r'C:/Users/Lenovo/WorkBuddy/2026-09-12-19-51-30/verify/out/paper2_per_sample.csv')
print('=' * 96)
print('A. 分布外(OOD) 恒等式检验:  E[Dice] = Detection x Delineation')
print('   Detection   = P(recall_gt > 0)            # 预测与GT有交集 -> 检出')
print('   Delineation = E[Dice | recall_gt > 0]     # 已检出条件下的勾画质量')
print('=' * 96)
print(f"{'dir':<16}{'arch':<16}{'n':>5}{'meanDice':>10}{'Detect':>9}{'Deline':>9}{'Det*Del':>10}{'err':>8}{'medRecallGt':>13}")
rows = []
for (dr, ar), g in d.groupby(['dir', 'arch']):
    n = len(g)
    det = (g['recall_gt'] > 0).mean()
    hsl = g['recall_gt'] > 0
    dl = g.loc[hsl, 'dice'].mean()
    prod = det * dl
    err = abs(prod - g['dice'].mean())
    rows.append((dr, ar, n, g['dice'].mean(), det, dl, prod, err, g['recall_gt'].median()))
    print(f"{dr:<16}{ar:<16}{n:>5}{g['dice'].mean():>10.2f}{det:>9.3f}{dl:>9.2f}{prod:>10.2f}{err:>8.3f}{g['recall_gt'].median():>13.3f}")
print()
print('恒等式最大误差:', max(r[7] for r in rows))

# ---------- 2) 分布内(ID): 用消融 sample_metrics.csv 的 Recall>0 ----------
print()
print('=' * 96)
print('B. 分布内(ID) 同样分解 (Recall 列 = |P n G| / |G|)')
print('=' * 96)
print(f"{'config':<12}{'n':>5}{'meanDice':>10}{'Detect':>9}{'Deline':>9}{'Det*Del':>10}{'err':>8}{'medRecall':>11}")
import os
cfgs = ['baseline', 'egm_only', 'dpa_only', 'msfa_only', 'egm_dpa', 'egm_msfa', 'dpa_msfa']
idtab = {}
for c in cfgs:
    f = f'D:/medical_segmentation/experiments/ablation/{c}/sample_metrics.csv'
    if not os.path.exists(f):
        print(c, 'MISSING'); continue
    g = pd.read_csv(f)
    det = (g['Recall'] > 0).mean()
    dl = g.loc[g['Recall'] > 0, 'Dice'].mean()
    idtab[c] = dict(n=len(g), dice=g['Dice'].mean() * 100, det=det, dl=dl * 100, medrec=g['Recall'].median())
    print(f"{c:<12}{len(g):>5}{g['Dice'].mean()*100:>10.2f}{det:>9.3f}{dl*100:>9.2f}{det*dl*100:>10.2f}{abs(det*dl*100-g['Dice'].mean()*100):>8.3f}{g['Recall'].median():>11.3f}")

print()
print('=' * 96)
print('C. 构念错配的量化: 同一个 Dice, 两个分布下"检出"贡献了多少')
print('=' * 96)
print('ID  (7 配置均值): Detect = %.3f, Deline = %.2f' % (
    np.mean([v['det'] for v in idtab.values()]), np.mean([v['dl'] for v in idtab.values()])))
print('OOD (16 组合均值): Detect = %.3f, Deline = %.2f' % (
    np.mean([r[4] for r in rows]), np.mean([r[5] for r in rows]) * 100))
