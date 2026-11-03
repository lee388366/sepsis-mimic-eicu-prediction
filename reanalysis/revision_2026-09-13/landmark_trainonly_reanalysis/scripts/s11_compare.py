#!/usr/bin/env python3
"""s11_compare.py —— 生成"新旧结果对照"总表与其他最终交付物

产出:
  results_summary_old_vs_trainonly.csv     新旧主性能/校准/阈值逐项对照
  tables/primary_results_all.csv           A/B 变体主结果并排
  NUMBER_DIFF_OLD_VS_NEW.md                稿件数字待改清单（供阶段11使用）
  threshold_interpretation.md               A/B 合并的阈值解释
  trainonly_reanalysis_manifest.json        全部产物清单 + 哈希 + 命令 + 版本

注意：本脚本只读，不修改任何稿件/DOCX。
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common_eval import f3

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent

MODELS = ['LSTM', 'RandomForest', 'SVM', 'LogisticRegression',
          'GaussianNB', 'KNN', 'DecisionTree']


def sha256(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT = HERE

    # ---------- 1. 主性能对照 ----------
    old_b = pd.read_csv(ROOT / 'baseline_results.csv')
    old_s = pd.read_csv(ROOT / 'sequence_results.csv')
    old = pd.concat([old_b, old_s], ignore_index=True)
    old_map = {r['model']: r for _, r in old.iterrows()}

    rows = []
    for V in ('A', 'B'):
        new = pd.read_csv(HERE / 'tables' / V / 'table2_model_performance.csv')
        for _, r in new.iterrows():
            m = r['Model']
            o = old_map.get(m)
            rows.append({
                'variant': V, 'model': m,
                'old_AUC': round(float(o['AUC']), 4) if o is not None else np.nan,
                'new_AUC': r['AUC'],
                'delta_AUC': round(r['AUC'] - float(o['AUC']), 4) if o is not None else np.nan,
                'new_AUC_CI_low_subject': r['AUC_CI_low_subject'],
                'new_AUC_CI_high_subject': r['AUC_CI_high_subject'],
                'old_AUPRC': round(float(o['AUPRC']), 4) if o is not None else np.nan,
                'new_AUPRC': r['AUPRC'],
                'old_threshold': round(float(o['best_threshold']), 2) if o is not None else np.nan,
                'new_threshold': r['F3_research_threshold'],
                'old_Precision': round(float(o['Precision']), 4) if o is not None else np.nan,
                'new_PPV': r['Precision_PPV'],
                'old_Recall': round(float(o['Recall']), 4) if o is not None else np.nan,
                'new_Sensitivity': r['Sensitivity'],
                'new_Specificity': r['Specificity'],
                'new_NPV': r['NPV'],
            })
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / 'results_summary_old_vs_trainonly.csv', index=False)
    print('===== 新旧 AUC 对照 =====')
    print(cmp[['variant', 'model', 'old_AUC', 'new_AUC', 'delta_AUC']].to_string(index=False))
    print(f'\nΔAUC 范围: {cmp.delta_AUC.min():+.4f} ~ {cmp.delta_AUC.max():+.4f} | '
          f'绝对值均值 {cmp.delta_AUC.abs().mean():.4f}')

    # 主结果并排（A/B）
    prim = []
    for V in ('A', 'B'):
        t = pd.read_csv(HERE / 'tables' / V / 'table2_model_performance.csv').copy()
        t.insert(0, 'variant', V)
        prim.append(t)
    pd.concat(prim, ignore_index=True).to_csv(OUT / 'tables' / 'primary_results_all.csv',
                                              index=False)

    # ---------- 2. 阈值解释合并 ----------
    parts = []
    for V in ('A', 'B'):
        f = HERE / f'threshold_interpretation_{V}.md'
        if f.exists():
            parts.append(f.read_text(encoding='utf-8'))
    parts.append("""
---

## 变体 A / B 的区别（必须写进方法或补充材料）

- **变体 A（主）**：数据库定义回顾性地标队列 —— MIMIC `sepsis3=TRUE` 且
  `suspected_infection_time` 非空，不限制感染时间落在 48 小时窗内。共 24,894 条记录 /
  21,608 位受试者，死亡 4,795（19.26%）。
- **变体 B（限制性地标敏感性）**：要求 `suspected_infection_time` 落在入 ICU 后 48 小时内，
  即在 48 小时地标时点**可判定**脓毒症表型。共 22,778 条 / 19,820 位受试者，死亡 4,313（18.93%）。
- 被 B 排除的 2,116 条记录（感染时间 > 48h）的死亡率更高（22.78%），说明
  "晚期感染"记录在临床上并不等价，不能默认它们对性能无影响。
- 报告任何数字时都必须注明属于 A 还是 B；不得把两者混用。
""")
    (OUT / 'threshold_interpretation.md').write_text('\n\n'.join(parts), encoding='utf-8')

    # ---------- 3. 稿件数字待改清单 ----------
    ext_a = pd.read_csv(HERE / 'tables' / 'A' / 'table9_external_validation.csv')
    ext_b = pd.read_csv(HERE / 'tables' / 'B' / 'table9_external_validation.csv')
    recal_a = pd.read_csv(HERE / 'tables' / 'A' / 'table10_recalibration_eicu_patient_split.csv')
    recal_b = pd.read_csv(HERE / 'tables' / 'B' / 'table10_recalibration_eicu_patient_split.csv')
    cal_a = pd.read_csv(HERE / 'calibration_primary_patient_clustered_A.csv')
    cal_b = pd.read_csv(HERE / 'calibration_primary_patient_clustered_B.csv')
    inter = {v: pd.read_csv(HERE / 'tables' / v / 'table11_subgroup_interaction.csv')
             for v in ('A', 'B')}
    old_t5 = pd.read_csv(ROOT / 'tables' / 'table5_sofa_comparison.csv')
    new_t5 = pd.read_csv(HERE / 'tables' / 'A' / 'table5_sofa_comparison.csv')

    def au(df, model, col):
        v = df.loc[df.Model == model, col]
        return float(v.iloc[0]) if len(v) else np.nan

    lines = ['# 稿件数字待改清单（由新产物自动提取）', '',
             '> 本文件仅列出**必须由新分析产物改写**的数字。未列出的数字（如队列人数、',
             '> 人口学、缺失率）在变体 A 下未变，但若改用变体 B 则全部需要替换。', '',
             '## 1. 内部测试集判别性能（变体 A，n=3,742 / 3,242 受试者 / 712 死亡）', '',
             '| 模型 | 旧 AUC | 新 AUC | Δ | 新受试者聚类 95%CI |', '|---|---|---|---|---|']
    for _, r in cmp[cmp.variant == 'A'].iterrows():
        lines.append(f"| {r.model} | {f3(r.old_AUC)} | {f3(r.new_AUC)} | {r.delta_AUC:+.4f} | "
                     f"{r.new_AUC_CI_low_subject:.3f}–{r.new_AUC_CI_high_subject:.3f} |")
    lines += ['', '## 2. 配对差值 LSTM − RF（LSTM 真正重训后）', '']
    pd_ = pd.read_csv(HERE / 'tables' / 'A' / 'table2_ci_paired_diff_lstm_rf.csv')
    for _, r in pd_.iterrows():
        lines.append(f"- **{r.Metric}**：点估计 {r.Point_estimate:+.4f}，"
                     f"受试者聚类 95%CI {r.CI_low_subject:+.4f}–{r.CI_high_subject:+.4f}，"
                     f"是否跨 0：{r.Crosses_zero_subject}")
    lines += ['', '> ⚠️ LSTM 是真正重新训练的。新 LSTM 与旧 LSTM 在测试集上的预测'
              '相关系数仅 r=0.968、平均绝对差 0.043（其余 6 个模型 r≥0.9985）。',
              '> 因此任何依赖单次 LSTM 拟合的推断都必须附上种子间离散度（见 '
              '`qa/s06_lstm_seed_stability.json` 与 `tables/A/lstm_seed_stability.csv`）。', '']

    lines += ['## 3. 校准（内部，raw 概率）', '',
              '| 模型 | Brier | slope | slope 95%CI（受试者聚类） | CITL | CITL 95%CI |',
              '|---|---|---|---|---|---|']
    for _, r in cal_a.iterrows():
        lines.append(f"| {r.Model} | {r.brier_raw:.3f} | {r.slope:.2f} | "
                     f"{r.slope_CI_low_subject:.2f}–{r.slope_CI_high_subject:.2f} | "
                     f"{r.CITL:.2f} | {r.CITL_CI_low_subject:.2f}–{r.CITL_CI_high_subject:.2f} |")
    lines += ['', '## 4. 内部阈值操作特征（F3 研究性操作点）', '',
              '| 模型 | 阈值 | 高危% | 死亡捕获 | 敏感度 | 特异度 | PPV | 1−PPV | 全部记录中假阳性% |',
              '|---|---|---|---|---|---|---|---|---|']
    th = pd.read_csv(HERE / 'tables' / 'A' / 'threshold_operating_characteristics_primary.csv')
    for _, r in th.iterrows():
        lines.append(f"| {r.Model} | {r.threshold:.2f} | {r.flagged_percent:.1f}% | "
                     f"{r.deaths_captured} | {r.death_recall_sensitivity:.3f} | "
                     f"{r.specificity:.3f} | {r.PPV:.3f} | {r.one_minus_PPV:.3f} | "
                     f"{r.false_positive_percent_of_all:.1f}% |")
    lines += ['', '> ⚠️ 旧稿把 `1−PPV` 写成"假阳性率"；两者分母不同。'
              '假阳性率的分母是全部记录（见最后一列）。', '']

    lines += ['## 5. eICU 跨库评价（uniquepid 聚类 95%CI）', '',
              '| 模型 | 变体A AUC | 变体A 95%CI | 变体B AUC | 变体B 95%CI | 旧 AUC |',
              '|---|---|---|---|---|---|']
    old_ex = pd.read_csv(ROOT / 'tables' / 'table9_external_validation.csv')
    for m in MODELS + ['SOFA (rule-based, 48h max)', 'qSOFA-6h (rule-based)']:
        oa = old_ex.loc[old_ex.Model == m, 'AUC_ext']
        lines.append(f"| {m} | {au(ext_a, m, 'AUC_ext'):.3f} | "
                     f"{au(ext_a, m, 'AUC_CI_low_patient'):.3f}–{au(ext_a, m, 'AUC_CI_high_patient'):.3f} | "
                     f"{au(ext_b, m, 'AUC_ext'):.3f} | "
                     f"{au(ext_b, m, 'AUC_CI_low_patient'):.3f}–{au(ext_b, m, 'AUC_CI_high_patient'):.3f} | "
                     f"{float(oa.iloc[0]):.3f} |" if len(oa) else
                     f"| {m} | {au(ext_a, m, 'AUC_ext'):.3f} | — | {au(ext_b, m, 'AUC_ext'):.3f} | — | — |")
    lines += ['', '> ⚠️ eICU 的脓毒症识别基于诊断字符串（24h 窗），与 MIMIC `sepsis3` 定义不同，',
              '> 只能称 **跨数据库探索性评价**，不得称 external validation。', '']

    lines += ['## 6. eICU 患者级再校准（uniquepid 划半，患者交集=0）', '',
              '| 模型 | 拟合半 Brier(raw) | 评价半 Brier(raw) | logistic 后 Brier | isotonic 后 Brier |',
              '|---|---|---|---|---|']
    for _, r in recal_a.iterrows():
        lines.append(f"| {r.Model} | — | {r.Brier_raw:.3f} | {r.Brier_logistic:.3f} | "
                     f"{r.Brier_isotonic:.3f} |")
    lines += ['', '> LSTM：旧稿记录级划半为 0.235→0.129；患者级划半（患者不重叠）为 '
              f"{au(recal_a, 'LSTM', 'Brier_raw'):.3f}→"
              f"{au(recal_a, 'LSTM', 'Brier_logistic'):.3f}。",
              '> 结论方向不变，但必须说明这是"重新拟合截距/斜率"带来的改善，',
              '> **不是**判别能力或临床获益的证明；再校准后各模型 Brier 趋于接近，',
              '> 说明其主要是概率尺度的修正。', '']

    lines += ['## 7. 亚组交互检验（对 LSTM 重训高度敏感，必须同时报告变体与种子不确定性）', '',
              '| 维度 | 旧 raw p | 旧 Bonferroni | 新(A) raw p | 新(A) Bonferroni | 新(B) raw p |',
              '|---|---|---|---|---|---|']
    old_i = pd.read_csv(ROOT / 'tables' / 'table11_subgroup_interaction.csv')
    for dim, col in [('Age', 'Age'), ('Initial SOFA', 'Initial SOFA'),
                     ('Septic shock', 'Cardiovascular SOFA')]:
        o = old_i.loc[old_i.Dimension == dim]
        a = inter['A'].loc[inter['A'].Dimension == col]
        b = inter['B'].loc[inter['B'].Dimension == col]
        lines.append(f"| {dim} | {float(o.interaction_p.iloc[0]):.3f} | "
                     f"{float(o.p_bonferroni.iloc[0]):.3f} | "
                     f"{float(a.interaction_p.iloc[0]):.3f} | "
                     f"{float(a.p_bonferroni.iloc[0]):.3f} | "
                     f"{float(b.interaction_p.iloc[0]):.3f} |")
    lines += ['', '> ⚠️ 交互检验 p 值在 LSTM 重训后从 0.301 变为 0.903（仅权重不同）。',
              '> 这说明该检验对深度模型的随机拟合高度不稳定。正文若继续引用单一 p 值，',
              '> 必须附上多种子范围，或改述为"未发现稳定的交互效应证据"。', '']

    lines += ['## 8. SOFA / qSOFA（同一评测子集、ID 与 y_true 已逐条核验一致）', '',
              '| 模型 | 旧 AUC | 新 AUC（变体A） |', '|---|---|---|']
    # 旧表的命名与新表不同，显式映射，避免出现"—"造成"未核对"的假象
    NAME_MAP = {'SOFA (rule-based, 48h max)': 'SOFA (rule-based)',
                'qSOFA-6h (rule-based)': 'qSOFA (rule-based)'}
    old_q = ROOT / 'tables' / 'table5_qsofa_comparison.csv'
    if old_q.exists():
        old_t5 = pd.concat([old_t5, pd.read_csv(old_q)], ignore_index=True)
    for m in new_t5.Model:
        oa = old_t5.loc[old_t5.Model == m, 'AUC']
        if not len(oa):
            oa = old_t5.loc[old_t5.Model == NAME_MAP.get(m, m), 'AUC']
        na = new_t5.loc[new_t5.Model == m, 'AUC']
        lines.append(f"| {m} | {float(oa.iloc[0]):.3f} | {float(na.iloc[0]):.3f} |"
                     if len(oa) else f"| {m} | 旧表无此评分 | {float(na.iloc[0]):.3f} |")
    lines += ['', '> SOFA/qSOFA 分数本身取自 0–48h 窗（与地标一致），未重算；',
              '> 仅把评测子集按 `stay_id` 对齐到新测试集，并逐条核验 `y_true` 一致。',
              '> SOFA AUC 与旧值相同（0.672），因为规则评分不依赖重训模型；',
              '> qSOFA 的旧 AUC 在旧表的 `table5_qsofa_comparison.csv` 中。', '']
    (OUT / 'NUMBER_DIFF_OLD_VS_NEW.md').write_text('\n'.join(lines), encoding='utf-8')

    # ---------- 4. manifest ----------
    files = sorted([p for p in OUT.rglob('*') if p.is_file()]
                   + [p for p in (OUT / 'models_trainonly').rglob('*') if p.is_file()])
    scripts = sorted((OUT / 'scripts').glob('*.py'))
    man = {
        'reanalysis': 'train-only imputation + 48-hour landmark + patient-level recalibration',
        'working_dir': str(OUT),
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'hashes': {str(p.relative_to(OUT)): sha256(p) for p in files},
        'scripts': {p.name: sha256(p) for p in scripts},
        'commands': [
            'python scripts/s00_snapshot.py',
            'python scripts/s01_landmark_audit.py',
            'python scripts/s02_build_trainonly_data.py',
            'python scripts/s03_train_models.py A',
            'python scripts/s03_train_models.py B',
            'python scripts/s04_internal_performance.py A',
            'python scripts/s04_internal_performance.py B',
            'python scripts/s05_subgroup.py A',
            'python scripts/s05_subgroup.py B',
            'python scripts/s06_lstm_seed_stability.py',
            'python scripts/s07_eicu_external.py A',
            'python scripts/s07_eicu_external.py B',
            'python scripts/s08_explain.py A',
            'python scripts/s09_cv.py',
            'python scripts/s10_figures.py A',
            'python scripts/s11_compare.py',
        ],
        'seeds': {'split': 42, 'models': 42, 'bootstrap': 42,
                  'lstm_seed_stability': [42, 43, 44, 45, 46, 47]},
        'n_bootstrap': {'internal_subject_clustered': 1000, 'internal_record': 1000,
                        'external_uniquepid': 1000, 'external_record': 1000},
        'isolation': {
            'old_files_overwritten': False,
            'old_predictions_loaded_in_main_analysis': False,
            'old_model_weights_loaded': False,
            'old_label': 'pre-split static imputation analysis (historical sensitivity)',
        },
        'n_files': len(files),
    }
    try:
        import sklearn, scipy, torch
        man['versions'] = {'sklearn': sklearn.__version__, 'scipy': scipy.__version__,
                           'torch': torch.__version__, 'numpy': np.__version__,
                           'pandas': pd.__version__}
    except Exception:
        pass
    (OUT / 'trainonly_reanalysis_manifest.json').write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n-> results_summary_old_vs_trainonly.csv / NUMBER_DIFF_OLD_VS_NEW.md / '
          f'trainonly_reanalysis_manifest.json（{len(files)} 个文件已哈希）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
