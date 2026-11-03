#!/usr/bin/env python3
# =====================================================================
# s00_snapshot.py —— 阶段1：隔离与快照
#
# 目的（对应用户要求的"输出隔离规则"）：
#   1. 记录旧模型 / 旧预测 / 旧CSV / 旧图 / 旧DOCX 的完整清单与 SHA-256，
#      作为"旧结果作为历史分析保留"的证据基线。
#   2. 记录环境版本、随机种子、执行命令。
#   3. 任何后续步骤都不得写入这些旧路径（本脚本只读）。
#
# 输出：00_manifest/old_files_sha256.csv
#       00_manifest/manifest.json
#       00_manifest/ENVIRONMENT.md
# =====================================================================
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # .../landmark_trainonly_reanalysis
REV = HERE.parent                                  # .../revision_2026-09-13
SITE = REV.parent                                  # .../claude code辅助后的
ROOT = SITE.parent                                 # 项目根
MAN = HERE / '00_manifest'
MAN.mkdir(exist_ok=True)

SEED = 42


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---- 旧交付清单（只读；这些路径禁止被本阶段的任何脚本写入）----
OLD = {
    '旧模型权重': [ROOT / 'models'],
    '旧预测文件': [ROOT / 'baseline_predictions.npz', ROOT / 'sequence_predictions.npz',
                   ROOT / 'all_predictions.npz',
                   ROOT / 'eicu_predictions.npz'],
    '旧训练数据': [ROOT / 'sepsis_training_data.npz', ROOT / 'eicu_training_data.npz',
                   ROOT / 'eicu_training_data_imputed.npz'],
    '旧插补中间件': [ROOT / 'sepsis_hourly_imputed_v2.csv'],
    '旧结果CSV': [ROOT / 'baseline_results.csv', ROOT / 'sequence_results.csv'],
    '旧结果表': [ROOT / 'tables'],
    '旧流水线脚本': [ROOT / '03_impute.py', ROOT / '05_assemble_training_data.py',
                     ROOT / '06_train_baseline.py', ROOT / '06b_train_sequence.py',
                     ROOT / 'split_utils.py', ROOT / 'make_merged_predictions.py'],
    '旧图': [SITE / 'figs', ROOT / 'figs'],
    '旧稿件与DOCX': [SITE / 'manuscript_EN_final.md', SITE / 'manuscript_ZH_translation.md',
                     SITE / 'supplementary_EN.md', SITE / 'author_notes.md',
                     SITE / 'Sepsis_Mortality_Manuscript_EN.docx',
                     SITE / 'Sepsis_Mortality_Manuscript_ZH.docx',
                     SITE / 'Sepsis_Mortality_Supplementary_Information.docx',
                     SITE / 'Author_Notes.docx'],
}

rows = []
for role, paths in OLD.items():
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.is_file():
                    rows.append({'role': role, 'path': str(f), 'bytes': f.stat().st_size,
                                 'sha256': sha256(f)})
        elif p.exists():
            rows.append({'role': role, 'path': str(p), 'bytes': p.stat().st_size,
                         'sha256': sha256(p)})
        else:
            rows.append({'role': role, 'path': str(p), 'bytes': None,
                         'sha256': None, 'note': 'MISSING'})

import csv
with open(MAN / 'old_files_sha256.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['role', 'path', 'bytes', 'sha256', 'note'])
    w.writeheader()
    w.writerows(rows)

# ---- 环境 ----
env = {
    'python': sys.version.split()[0],
    'platform': platform.platform(),
    'machine': platform.machine(),
}
for mod in ('numpy', 'pandas', 'sklearn', 'scipy', 'torch', 'joblib', 'statsmodels',
            'matplotlib', 'shap', 'lifelines'):
    try:
        m = __import__(mod)
        env[mod] = getattr(m, '__version__', 'unknown')
    except Exception as e:                      # noqa: BLE001
        env[mod] = f'NOT_AVAILABLE ({type(e).__name__})'

torch_info = {}
try:
    import torch
    torch_info = {'threads': torch.get_num_threads(),
                  'mps_built': torch.backends.mps.is_built(),
                  'mps_available': torch.backends.mps.is_available(),
                  'cuda_available': torch.cuda.is_available()}
except Exception:                               # noqa: BLE001
    pass

try:
    psql = subprocess.run(['psql', '--version'], capture_output=True, text=True).stdout.strip()
except Exception:                               # noqa: BLE001
    psql = 'NOT_AVAILABLE'

man = {
    'stage': 'landmark_trainonly_reanalysis',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'seed': SEED,
    'paths': {'work_dir': str(HERE), 'revision_dir': str(REV), 'site_dir': str(SITE),
              'project_root': str(ROOT)},
    'environment': env,
    'torch_runtime': torch_info,
    'psql': psql,
    'isolation_policy': {
        'rule': '本阶段所有写入仅限 work_dir；旧路径(ROOT 与 SITE 根目录)禁止写入',
        'old_model_weights': '不加载、不复用；全部模型重新训练',
        'old_predictions': '不加载、不复用；全部预测重新生成',
        'old_imputation': 'sepsis_hourly_imputed_v2.csv 不作为正式主分析输入;'
                          '正式主分析从 sepsis_hourly.csv 重新插补',
        'loss_function_legacy_cla': '旧结果标记为 previous_analysis (pre-split static imputation)',
    },
    'file_count': len(rows),
    'files_hashing_failed': [r['path'] for r in rows if r['sha256'] is None],
}
(MAN / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ['# 环境与随机种子', '',
         f'- 时间：{man["created_utc"]}', f'- Python：{env["python"]}',
         f'- 平台：{env["platform"]}', f'- 随机种子：SEED = {SEED}', '']
lines += [f'- {k}：{v}' for k, v in env.items() if k not in ('python', 'platform')]
lines += ['', '## torch 运行时', ''] + [f'- {k}：{v}' for k, v in torch_info.items()]
lines += ['', f'- psql：{psql}', '', '## 工作目录', '', f'- work_dir：{HERE}',
          f'- project_root：{ROOT}', '', '## 隔离规则', '',
          '- 本阶段所有输出仅写入 `landmark_trainonly_reanalysis/`。',
          '- 旧模型权重、旧预测 npz、旧 tables/、旧 figs/、根目录稿件与 DOCX 一律只读。',
          '- 正式主分析从 `sepsis_hourly.csv` 重新插补，不使用 `sepsis_hourly_imputed_v2.csv`。',
          '- 旧结果以 `previous_analysis (pre-split static imputation)` 保留。']
(MAN / 'ENVIRONMENT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(f'已快照 {len(rows)} 个旧文件 -> {(MAN / "old_files_sha256.csv")}')
print('缺失/未找到的路径：', man['files_hashing_failed'] or '无')
print('环境：', json.dumps(env, ensure_ascii=False))
print('torch：', torch_info)
