# 分解恒等式与方差归属 —— 实测证据（2026-09-13）

> **⚠️ 溯源（2026-09-13 P2-1 后补）**
> 本文件 §A–C 的原型输出（`decompose_check.py`）已被**正式产物取代**：
> - 脚本：`02_code/analysis/decompose.py`（含 B1–B6 边界条件 + 单元自检）
> - 数据：`03_results/stats/decomposition.csv` / `.json`（36 行 = 16 跨域格 + 20 组分布内）
> - 报告：`03_results/stats/decomposition_report.md`（机器生成）
>
> 正式产物结论与 §A–C **逐位一致**（最大残差 5.248540e-10；ID 检出 20/20 = 1.000）。
> **§D–F（方差归属 / 对齐性）仍为有效来源**，但 n=4 是初值，终验在 P2-2b。
> 引用请优先使用 `decomposition_report.md`。

## A. 恒等式检验（decompose_check.py）
```
================================================================================================
A. 分布外(OOD) 恒等式检验:  E[Dice] = Detection x Delineation
   Detection   = P(recall_gt > 0)            # 预测与GT有交集 -> 检出
   Delineation = E[Dice | recall_gt > 0]     # 已检出条件下的勾画质量
================================================================================================
dir             arch                n  meanDice   Detect   Deline   Det*Del     err  medRecallGt
cvc_to_etis     AttentionUNet     196      0.30    0.566     0.54      0.30   0.000        0.041
cvc_to_etis     EGAUNet           196      0.56    0.791     0.70      0.56   0.000        0.914
cvc_to_etis     TransUNet         196      0.21    0.505     0.41      0.21   0.000        0.004
cvc_to_etis     UNet              196      0.34    0.689     0.49      0.34   0.000        0.297
cvc_to_kvasir   AttentionUNet    1000      0.35    0.767     0.46      0.35   0.000        0.205
cvc_to_kvasir   EGAUNet          1000      0.77    0.979     0.79      0.77   0.000        0.883
cvc_to_kvasir   TransUNet        1000      0.39    0.895     0.43      0.39   0.000        0.396
cvc_to_kvasir   UNet             1000      0.51    0.934     0.55      0.51   0.000        0.512
kvasir_to_cvc   AttentionUNet     612      0.45    0.848     0.53      0.45   0.000        0.494
kvasir_to_cvc   EGAUNet           612      0.70    0.938     0.75      0.70   0.000        0.828
kvasir_to_cvc   TransUNet         612      0.50    0.962     0.52      0.50   0.000        0.661
kvasir_to_cvc   UNet              612      0.55    0.895     0.61      0.55   0.000        0.678
kvasir_to_etis  AttentionUNet     196      0.38    0.577     0.66      0.38   0.000        0.055
kvasir_to_etis  EGAUNet           196      0.55    0.755     0.73      0.55   0.000        0.852
kvasir_to_etis  TransUNet         196      0.45    0.755     0.60      0.45   0.000        0.616
kvasir_to_etis  UNet              196      0.39    0.607     0.64      0.39   0.000        0.085

恒等式最大误差: 5.248539935998053e-10

================================================================================================
B. 分布内(ID) 同样分解 (Recall 列 = |P n G| / |G|)
================================================================================================
config          n  meanDice   Detect   Deline   Det*Del     err  medRecall
baseline      242     81.97    1.000    81.97     81.97   0.000      0.941
egm_only      242     86.67    1.000    86.67     86.67   0.000      0.953
dpa_only      242     86.65    1.000    86.65     86.65   0.000      0.955
msfa_only     242     87.46    1.000    87.46     87.46   0.000      0.961
egm_dpa       242     86.96    1.000    86.96     86.96   0.000      0.955
egm_msfa      242     88.32    1.000    88.32     88.32   0.000      0.959
dpa_msfa      242     87.12    1.000    87.12     87.12   0.000      0.963

================================================================================================
C. 构念错配的量化: 同一个 Dice, 两个分布下"检出"贡献了多少
================================================================================================
ID  (7 配置均值): Detect = 1.000, Deline = 86.45
OOD (16 组合均值): Detect = 0.779, Deline = 58.79
```

## B. 方差归属与对齐性（decompose_variance.py）
```
====================================================================================================
D. OOD 的方差归属 (log 尺度):  log Dice = log Detection + log Delineation
====================================================================================================
var(logDice)=0.1067  var(logDet)=0.0442 ( 41.5%)  var(logDel)=0.0393 ( 36.9%)  2cov=0.0231 ( 21.7%)
Detection  范围: 0.505 - 0.979  (极差 0.474)
Delineation范围: 0.412 - 0.787  (极差 0.375)
按 log 尺度极差: Detection 0.662 | Delineation 0.648

====================================================================================================
E. 分布内 (ID) 的检出率 —— 是否退化为 1.000
====================================================================================================
set                             n    Detect    Deline
ablation/baseline             242     1.000     81.97
ablation/egm_only             242     1.000     86.67
ablation/dpa_only             242     1.000     86.65
ablation/msfa_only            242     1.000     87.46
ablation/egm_dpa              242     1.000     86.96
ablation/egm_msfa             242     1.000     88.32
ablation/dpa_msfa             242     1.000     87.12
baseline/UNet                 242     1.000     82.38
baseline/AttentionUNet        242     1.000     83.95
baseline/TransUNet            242     1.000     83.91
baseline/EGAUNet              242     1.000     88.53

--- 全部 13 个已训练 baseline 的 ID 检出率 ---
  [skip MultiResUNet] cols=['Dice', 'IoU', 'HD95', 'sample_idx']
  [skip PraNet] cols=['Dice', 'IoU', 'HD95', 'sample_idx']
共 11 个架构, 检出率 min=1.0000 max=1.0000, n=242

====================================================================================================
F. 对齐性: ID 上测到的能力  vs  域外真正决定成败的能力
====================================================================================================
         arch  id_dice  id_det  id_dl  ood_det  ood_dl  ood_dice
      EGAUNet    0.885   1.000  0.885    0.866   0.743     0.645
AttentionUNet    0.839   1.000  0.839    0.689   0.545     0.370
    TransUNet    0.839   1.000  0.839    0.779   0.491     0.387
         UNet    0.824   1.000  0.824    0.781   0.573     0.447

Spearman(ID Delineation, OOD Detection) = +0.200  p=0.8000   <-- 应接近 0/负 = 错位
Spearman(ID Delineation, OOD Dice)      = +0.200  p=0.8000
Spearman(OOD Detection,  OOD Dice)      = +1.000  p=0.0000   <-- 应接近 +1 = 决定成败
```
