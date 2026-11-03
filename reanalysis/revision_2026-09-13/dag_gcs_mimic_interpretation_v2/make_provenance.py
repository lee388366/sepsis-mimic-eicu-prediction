#!/usr/bin/env python3
# =====================================================================
# make_provenance.py —— 重新生成 provenance.json（v2）
#
# 依接收决定第 1 条：报告字段级差异及哈希。因此本文件同时记录：
#   - 上游输入（只读）的 SHA-256
#   - v1 与 v2 底表的 SHA-256（供独立复核"旧版未被覆盖"）
#   - 本阶段代码与交付文本的 SHA-256
# =====================================================================
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
V1_DIR = REV / 'dag_gcs_mimic_interpretation'
UP = REV / 'dag_gcs_next_stage' / 'output'


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rec(p: Path, role: str) -> dict:
    if not p.exists():
        return {'path': str(p), 'role': role, 'exists': False}
    return {'path': str(p), 'role': role, 'exists': True,
            'bytes': p.stat().st_size, 'sha256': sha256(p)}


def main() -> int:
    prov = {
        'generated_by': ('dag_gcs_mimic_interpretation_v2 / build_base_table.py + '
                         'qa_base_table.py + diff_v1_v2.py + make_provenance.py'),
        'version': 'v2',
        'v2_fixes': {
            'FIX-1': '治疗区间依据错配：v1 用累计最大终点判断存在覆盖，却返回起点最晚的'
                     '（可能已结束的）区间。v2 只返回真正满足 start <= t <= end 的区间，'
                     '并发取起点最晚者，并落盘 n_covering / active_all_drugs / '
                     'basis_covers_timepoint。',
            'FIX-2': '总分规则标记：v1 对三个总分分支统一套用同一套分项逻辑，'
                     'default 标记 761 行、carry 标记 1,534 行与实际分支不符。'
                     'v2 按分支实际算式分别设置，并从落盘 CSV 逐分支验证恒等式。',
            'detail': 'CHANGELOG_from_v1.md',
        },
        'environment': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'platform': platform.platform(),
        },
        'database': {
            'mimic': 'mimiciv31 (PostgreSQL 12, 只读)',
            'note': 'v2 未做任何数据库提取；两个生理变量提取来自 v1，输出与 v1 逐字节一致',
        },
        'upstream_delivery_readonly': [
            rec(UP / f, '上游既有交付（只读）') for f in (
                'mimic_00_cohort_index.csv', 'mimic_01_gcs_event_raw.csv',
                'mimic_02_gcs_derived.csv', 'mimic_03_drug_inputevents.csv',
                'mimic_04_drug_emar.csv', 'mimic_05_drug_prescriptions.csv',
                'mimic_06_ventilation_derived.csv', 'mimic_09_rass_cam_icu.csv')
        ],
        'extraction_outputs_readonly': [
            rec(HERE / 'output' / f, '上轮新增提取输出（与 v1 逐字节一致）')
            for f in ('mimic_12_vitals_timeseries.csv', 'mimic_13_bg_ph.csv')
        ],
        'sql_readonly': [
            rec(HERE / 'sql' / f, '上轮新增 SQL（v2 未修改）')
            for f in ('mimic_12_vitals_timeseries.sql', 'mimic_13_bg_ph.sql')
        ],
        'code': [
            rec(HERE / f, 'v2 代码') for f in (
                'build_base_table.py', 'qa_base_table.py', 'make_deliverables.py',
                'diff_v1_v2.py', 'make_provenance.py', 'run_extract.sh',
                'probe_reconstruct_gcs.py')
        ],
        'outputs': [
            rec(HERE / f, 'v2 交付') for f in (
                'base/mimic_gcs_timebase.csv', 'data_dictionary.csv', 'QA_REPORT.md',
                'JOIN_AND_ALIGNMENT_RULES.md', 'GAPS.md', 'HANDOFF.md',
                'CHANGELOG_from_v1.md',
                'qa/build_facts.json', 'qa/qa_facts.json',
                'qa/diff_v1_v2.json', 'qa/diff_v1_v2_columns.csv')
        ],
        'v1_vs_v2': {
            'v1_dir': str(V1_DIR),
            'v1_unchanged': True,
            'v1_base_table': rec(V1_DIR / 'base' / 'mimic_gcs_timebase.csv', 'v1 底表（保留）'),
            'v2_base_table': rec(HERE / 'base' / 'mimic_gcs_timebase.csv', 'v2 底表'),
        },
        'external_reference': {
            'mimic-code gcs.sql': ('https://raw.githubusercontent.com/MIT-LCP/mimic-code/'
                                   'main/mimic-iv/concepts/measurement/gcs.sql'),
            'note': '仅作为复现规则的依据；复现结果已与本地 mimiciv_derived.gcs 逐格比对为 0 差异',
        },
    }
    (HERE / 'provenance.json').write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf-8')
    print('provenance.json 已生成')
    print('  v1 底表 sha256 =', prov['v1_vs_v2']['v1_base_table'].get('sha256'))
    print('  v2 底表 sha256 =', prov['v1_vs_v2']['v2_base_table'].get('sha256'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
