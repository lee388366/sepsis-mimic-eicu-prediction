#!/usr/bin/env python3
# =====================================================================
# build_base_table.py —— MIMIC 临床解释探索：GCS 时点分析底表
#
# 目标底表：一行 = 一个 GCS 评分时点（stay_id + gcs_charttime）
# 用于研究"眼/运动反应 ↔ RASS / 镇静 / 呼吸支持 / 生理异常"的共同模式。
#
# 硬性原则（对应任务要求）：
#   P1 只取正式队列 + 原研究观察窗 [intime, intime+48h)；按 stay_id 与
#      时间连接，绝不按 CSV 行序拼接。
#   P2 眼/运动/语言/总分/不可测标记一律保留**派生层原样值**；
#      不改 verbal 0→1，不自行重算总分。
#   P3 通过复现上游 gcs.sql 规则，为每个分项标注来源：
#      当前记录 / 6h 内前值补入 / 缺失。复现必须与派生层逐格一致（硬门禁）。
#   P4 背景变量按**事件时间（charttime）**取"同一时刻或既往"记录，
#      **未向未来匹配**；时间差全部落盘。
#      ⚠️ 这不等同于"当时已可获得"：本版未纳入 storetime / 录入延迟，
#      因此不能声称这些变量在评分时刻已是实时可得输入（见 GAPS.md G-10）。
#   P5 给药状态分层：可确认给药 / 明确未给药 / 未确认医嘱，三者不合并；
#      无记录 = 未知，不转成阴性。
#   P6 生理变量保留精确观察时间、单位、缺失状态；不设阈值、不插补、不删异常值。
#
# v2 相对 v1 的两处修复（详见 CHANGELOG_from_v1.md）：
#   FIX-1 区间依据错配：v1 用"累计最大终点"判断存在覆盖，却返回起点最晚的
#         区间，该区间可能已结束（嵌套例 A=[0,10]、B=[2,3]、t=5）。
#         v2 只返回**真正满足 start <= t <= end** 的区间，并发时取起点最晚者，
#         并把覆盖条数 (n_covering) 与全部并发药名一并落盘。
#   FIX-2 总分规则标记：v1 对三个总分分支统一套用同一套分项逻辑，
#         导致 default 标记 761 行、carry 标记 1,534 行与实际计算分支不符。
#         v2 按各分支的实际算式分别设置标记。
#
# 不训练模型、不做显著性筛选、不改论文。
# 只读既有交付 + 本目录 output/ 的最小新增提取；输出全部写入本目录 base/ qa/。
# =====================================================================
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
UP = REV / 'dag_gcs_next_stage' / 'output'     # 上游既有交付（只读）
OUT = HERE / 'output'                          # 本目录最小新增提取
BASE = HERE / 'base'
QA = HERE / 'qa'

NS_PER_H = 3.6e12

DRUG_CLASSES = {
    'sedative': 'sedative_hypnotic',
    'opioid': 'opioid',
    'nmb': 'neuromuscular_blocker',
}

# eMAR event_txt 分层：只有明确"已给药"才计入可确认给药
EMAR_ADMIN_TXT = {
    'Administered',
    'Administered Bolus from IV Drip',
    'Administered in Other Location',
    'Partial Administered',
}
EMAR_NOT_GIVEN_TXT = {'Not Given'}

WINDOW_LO_H = 0.0
WINDOW_HI_H = 48.0

notes: list[str] = []
qa_facts: dict = {}


def note(msg: str) -> None:
    notes.append(msg)
    print('  ·', msg)


# ---------------------------------------------------------------------
# 通用工具：精确时间
# ---------------------------------------------------------------------
def to_ns(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy().astype('datetime64[ns]').astype('int64')


class Prior:
    """按 (id, 时间) 排序的"既往记录"索引。只取 <= 查询时刻的记录。"""

    def __init__(self, ev: pd.DataFrame, id_col: str, tcol: str):
        ev = ev.copy()
        ev['_t'] = to_ns(ev[tcol])
        ev = ev.sort_values([id_col, '_t'], kind='stable').reset_index(drop=True)
        self.ev = ev
        ids = ev[id_col].to_numpy()
        uniq, first = np.unique(ids, return_index=True)
        last = np.append(first[1:], len(ids))
        self.bounds = {int(u): (int(a), int(b)) for u, a, b in zip(uniq, first, last)}
        self.t = ev['_t'].to_numpy()
        self.max_t = ev.groupby(id_col)['_t'].max().to_dict()

    def last_at_or_before(self, q_id: np.ndarray, q_t: np.ndarray) -> np.ndarray:
        """返回 ev 中位置索引数组；无匹配为 -1。"""
        res = np.full(len(q_id), -1, dtype=np.int64)
        order = np.argsort(q_id, kind='stable')
        sid = q_id[order]
        uniq, first = np.unique(sid, return_index=True)
        last = np.append(first[1:], len(sid))
        for u, a, b in zip(uniq, first, last):
            bd = self.bounds.get(int(u))
            if bd is None:
                continue
            lo, hi = bd
            rows = order[a:b]
            k = np.searchsorted(self.t[lo:hi], q_t[rows], side='right') - 1
            ok = k >= 0
            res[rows[ok]] = lo + k[ok]
        return res

    def only_later(self, q_id: np.ndarray, q_t: np.ndarray, matched: np.ndarray) -> np.ndarray:
        """无既往记录，但该 id 存在更晚记录 → 说明"不是没有，而是还没发生"。"""
        has_later = np.array([self.max_t.get(int(i), -1) > t for i, t in zip(q_id, q_t)])
        return (~matched) & has_later


class Intervals:
    """区间 [start, end] 索引（含端点），用于"该时刻是否处于治疗中"。

    【v2 修复】覆盖判定与选择规则：
      区间覆盖查询时刻 t 当且仅当 start <= t <= end（两端都满足）。
      并发选择规则：在**所有真正覆盖 t 的区间**中，取**起点最晚**者
      （即最近开始、且此刻仍在进行的那一条）。
      绝不返回"起点在 t 之前但已结束"的区间 —— v1 曾用"累计最大终点"判断
      "至少存在一个覆盖区间"，却返回起点最晚的那条，因此可能把已结束区间的
      药名/速率/状态当成依据（嵌套例：A=[0,10]、B=[2,3]、t=5）。
      并发条数与全部并发药名一并落盘，使选择规则可被复核。

    实现：按 stay 分组的暴力比较（每个 stay 的 查询数 × 区间数 矩阵）。
    区间数据量（每 stay 最大 163 条）使这一做法既简单又显然正确，
    便于 QA 用不同代码路径复算。
    """

    def __init__(self, ev: pd.DataFrame, id_col: str, start_col: str, end_col: str):
        ev = ev.copy()
        ev['_s'] = to_ns(ev[start_col])
        ev['_e'] = to_ns(ev[end_col])
        ev = ev.sort_values([id_col, '_s'], kind='stable').reset_index(drop=True)
        self.ev = ev
        self.id_col = id_col
        ids = ev[id_col].to_numpy()
        uniq, first = np.unique(ids, return_index=True)
        last = np.append(first[1:], len(ids))
        self.bounds = {int(u): (int(a), int(b)) for u, a, b in zip(uniq, first, last)}
        self.s = ev['_s'].to_numpy()
        self.e = ev['_e'].to_numpy()
        self.max_s = ev.groupby(id_col)['_s'].max().to_dict()
        self.max_e = ev.groupby(id_col)['_e'].max().to_dict()

    def covering(self, q_id: np.ndarray, q_t: np.ndarray):
        """返回 (选中位置索引, 覆盖条数)。无覆盖为 (-1, 0)。

        chosen = 覆盖区间中起点最晚者的全局位置索引。
        """
        n = len(q_id)
        chosen = np.full(n, -1, dtype=np.int64)
        ncov = np.zeros(n, dtype=np.int32)
        order = np.argsort(q_id, kind='stable')
        sid = q_id[order]
        uniq, first = np.unique(sid, return_index=True)
        last = np.append(first[1:], len(sid))
        for u, a, b in zip(uniq, first, last):
            bd = self.bounds.get(int(u))
            if bd is None:
                continue
            lo, hi = bd
            rows = order[a:b]
            q = q_t[rows]
            s = self.s[lo:hi]
            e = self.e[lo:hi]
            M = (s[None, :] <= q[:, None]) & (e[None, :] >= q[:, None])
            cnt = M.sum(axis=1)
            ncov[rows] = cnt.astype(np.int32)
            has = cnt > 0
            if has.any():
                # 从右往左第一个 True = 起点最晚的覆盖区间
                rev = M[has][:, ::-1].argmax(axis=1)
                chosen[rows[has]] = lo + (len(s) - 1 - rev)
        return chosen, ncov

    def only_later(self, q_id: np.ndarray, q_t: np.ndarray, matched: np.ndarray) -> np.ndarray:
        later = np.array([
            (self.max_s.get(int(i), -1) > t) or (self.max_e.get(int(i), -1) > t)
            for i, t in zip(q_id, q_t)
        ])
        return (~matched) & later


def lookback_features(out: dict, prefix: str, prior: Prior, q_id: np.ndarray,
                      q_t: np.ndarray, cols: list[str], within: tuple = (1, 2, 4)) -> np.ndarray:
    """把"同一时刻或既往最近一条"的字段展开到 out。返回 matched 掩码。

    只允许 chattime <= 查询时刻：侧向 'right' 后再减 1，结构上排除未来记录。
    """
    idx = prior.last_at_or_before(q_id, q_t)
    m = idx >= 0
    safe = np.where(m, idx, 0)
    for c in cols:
        vals = prior.ev[c].to_numpy()
        arr = vals[safe]
        if vals.dtype == object or vals.dtype.kind in 'OU':
            out[f'{prefix}_{c}'] = np.where(m, arr, None).astype(object)
        else:
            out[f'{prefix}_{c}'] = np.where(m, arr, np.nan)
    gap = np.where(m, (q_t - prior.t[safe]) / NS_PER_H, np.nan)
    out[f'{prefix}_gap_h'] = gap
    out[f'{prefix}_status'] = np.where(
        ~m, 'unknown_no_record_at_or_before',
        np.where(np.nan_to_num(gap, nan=1.0) == 0, 'same_time', 'prior'))
    for h in within:
        out[f'{prefix}_within_{h}h'] = (m & (gap <= h)).astype(int)
    out[f'{prefix}_only_later'] = prior.only_later(q_id, q_t, m).astype(int)
    return m


def interval_features(out: dict, prefix: str, iv: Intervals, q_id: np.ndarray,
                      q_t: np.ndarray, basis_cols: list[str],
                      concat_col: str | None = None) -> dict:
    """该时刻是否处于治疗中 + 真正覆盖该时刻的区间原样字段。

    【v2】返回诊断计数，供 QA 与 v1 对照。
    """
    act, ncov = iv.covering(q_id, q_t)
    m = act >= 0
    safe = np.where(m, act, 0)
    out[f'{prefix}_active'] = m.astype(int)
    out[f'{prefix}_n_covering'] = ncov.astype(np.int32)
    for c in basis_cols:
        vals = iv.ev[c].to_numpy()
        arr = vals[safe]
        if vals.dtype == object or vals.dtype.kind in 'OU':
            out[f'{prefix}_active_basis_{c}'] = np.where(m, arr, None).astype(object)
        else:
            out[f'{prefix}_active_basis_{c}'] = np.where(m, arr, np.nan)
    if concat_col:
        vals = iv.ev[concat_col].astype(str).to_numpy()
        joined = np.empty(len(q_id), dtype=object)
        order = np.argsort(q_id, kind='stable')
        sid = q_id[order]
        uniq, first = np.unique(sid, return_index=True)
        last = np.append(first[1:], len(sid))
        cc = np.full(len(q_id), None, dtype=object)
        for u, a, b in zip(uniq, first, last):
            bd = iv.bounds.get(int(u))
            if bd is None:
                continue
            lo, hi = bd
            rows = order[a:b]
            q = q_t[rows]
            s, e = iv.s[lo:hi], iv.e[lo:hi]
            M = (s[None, :] <= q[:, None]) & (e[None, :] >= q[:, None])
            for r_i, has in zip(rows, M):
                if has.any():
                    cc[r_i] = ' | '.join(sorted(set(vals[lo:hi][has])))
        out[f'{prefix}_active_all_drugs'] = cc
    # 边界时间差：start_gap = t - start（覆盖时必 >=0）；end_gap = t - end（覆盖时必 <=0）
    out[f'{prefix}_active_start_gap_h'] = np.where(m, (q_t - iv.s[safe]) / NS_PER_H, np.nan)
    out[f'{prefix}_active_end_gap_h'] = np.where(m, (q_t - iv.e[safe]) / NS_PER_H, np.nan)
    out[f'{prefix}_only_later'] = iv.only_later(q_id, q_t, m).astype(int)
    # 诊断：所选区间的字段是否真的覆盖（结构上必为真，落盘供 QA 复核）
    cov_ok = np.ones(len(q_id), dtype=np.int8)
    if m.any():
        cov_ok[m] = ((iv.s[safe] <= q_t) & (q_t <= iv.e[safe]))[m].astype(np.int8)
    out[f'{prefix}_basis_covers_timepoint'] = cov_ok
    return {'active': int(m.sum()), 'n_covering_gt1': int((ncov > 1).sum()),
            'basis_not_covering': int((m & (cov_ok == 0)).sum())}


def prior_by_end(iv: Intervals, q_id: np.ndarray, q_t: np.ndarray) -> np.ndarray:
    """按结束时间找"既往最近一次结束的区间"位置索引，-1 为无。"""
    order = np.argsort(iv.ev['_e'].to_numpy(), kind='stable')
    res = np.full(len(q_id), -1, dtype=np.int64)
    qorder = np.argsort(q_id, kind='stable')
    sid = q_id[qorder]
    uniq, first = np.unique(sid, return_index=True)
    last = np.append(first[1:], len(sid))
    for u, a, b in zip(uniq, first, last):
        bd = iv.bounds.get(int(u))
        if bd is None:
            continue
        lo, hi = bd
        sub = iv.e[lo:hi]
        o = np.argsort(sub, kind='stable')
        sub_sorted = sub[o]
        rows = qorder[a:b]
        k = np.searchsorted(sub_sorted, q_t[rows], side='right') - 1
        ok = k >= 0
        res[rows[ok]] = lo + o[k[ok]]
    return res


def main() -> int:
    for d in (BASE, QA):
        d.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 0. 队列索引
    # =================================================================
    coh = pd.read_csv(UP / 'mimic_00_cohort_index.csv', low_memory=False)
    coh['intime'] = pd.to_datetime(coh.intime)
    coh = coh[['subject_id', 'hadm_id', 'stay_id', 'intime', 'label', 'n_icu_stays',
               'los_h', 't_inf_minus_intime_h']].copy()
    coh['win_lo'] = coh.intime + pd.Timedelta(hours=WINDOW_LO_H)
    coh['win_hi'] = coh.intime + pd.Timedelta(hours=WINDOW_HI_H)
    qa_facts['cohort_n_records'] = int(len(coh))
    qa_facts['cohort_n_subjects'] = int(coh.subject_id.nunique())
    qa_facts['cohort_n_deaths'] = int(coh.label.sum())
    note(f'队列：{len(coh)} 条住院记录 / {coh.subject_id.nunique()} 名受试者 / '
         f'{int(coh.label.sum())} 例院内死亡')

    # =================================================================
    # 1. GCS：复现上游 gcs.sql 规则并标注来源
    # =================================================================
    raw = pd.read_csv(UP / 'mimic_01_gcs_event_raw.csv', low_memory=False,
                      usecols=['stay_id', 'charttime', 'itemid', 'value_raw', 'valuenum'])
    der = pd.read_csv(UP / 'mimic_02_gcs_derived.csv', low_memory=False,
                      usecols=['stay_id', 'charttime', 'gcs_total', 'gcs_eyes',
                               'gcs_verbal', 'gcs_motor', 'gcs_unable'])

    raw['_motor'] = np.where(raw.itemid == 223901, raw.valuenum, np.nan)
    raw['_verbal'] = np.where(raw.itemid == 223900,
                              np.where(raw.value_raw == 'No Response-ETT', 0, raw.valuenum), np.nan)
    raw['_eyes'] = np.where(raw.itemid == 220739, raw.valuenum, np.nan)
    raw['_ett'] = np.where((raw.itemid == 223900) & (raw.value_raw == 'No Response-ETT'), 1, 0)

    base = raw.groupby(['stay_id', 'charttime'], as_index=False).agg(
        gcsmotor=('_motor', 'max'), gcsverbal=('_verbal', 'max'),
        gcseyes=('_eyes', 'max'), endotrachflag=('_ett', 'max'))
    base = base.sort_values(['stay_id', 'charttime'], kind='stable').reset_index(drop=True)
    base['gcs_seq'] = base.groupby('stay_id').cumcount() + 1
    base['_t'] = to_ns(base.charttime)

    prev = base[['stay_id', 'gcs_seq', 'gcsmotor', 'gcsverbal', 'gcseyes', '_t']].copy()
    prev['gcs_seq'] += 1
    prev = prev.rename(columns={'gcsmotor': 'p_motor', 'gcsverbal': 'p_verbal',
                                'gcseyes': 'p_eyes', '_t': 'p_t'})
    b = base.merge(prev, on=['stay_id', 'gcs_seq'], how='left')
    in6 = b.p_t > (b._t - 6 * NS_PER_H)          # 上游：b2.charttime > b.charttime - 6h
    b['has_prev_le6h'] = in6.fillna(False).astype(int)
    p_motor = b.p_motor.where(in6)
    p_verbal = b.p_verbal.where(in6)
    p_eyes = b.p_eyes.where(in6)

    # ---- 分项来源归属（核心：区分当前记录 / 前值补入 / 缺失）----
    def origin_plain(cur: pd.Series, prv: pd.Series) -> pd.Series:
        return pd.Series(np.where(cur.notna(), 'measured_current',
                                  np.where(prv.notna(), 'carried_forward_le6h', 'missing')),
                         index=cur.index)

    o_eyes = origin_plain(b.gcseyes, p_eyes)
    o_motor = origin_plain(b.gcsmotor, p_motor)
    o_verbal = pd.Series(np.where(
        b.gcsverbal == 0, 'measured_current_ett_zero',
        np.where(b.gcsverbal.notna(), 'measured_current',
                 np.where(p_verbal == 0, 'carried_forward_ett_zero',
                          np.where(p_verbal.notna(), 'carried_forward_le6h', 'missing')))),
        index=b.index)

    # ---- 复现总分（仅用于**验证**与来源归类，不作分析变量）----
    ett_cur = b.gcsverbal == 0
    ett_prev = b.gcsverbal.isna() & (p_verbal == 0) & ~ett_cur
    prev_ett_recompute = (p_verbal == 0) & ~ett_cur & ~ett_prev
    plain = ~(ett_cur | ett_prev | prev_ett_recompute)

    used_default, used_carry = np.zeros(len(b), bool), np.zeros(len(b), bool)
    # 【v2 修复】两个标记必须依**总分 CASE 的实际分支**设置，不能用统一的分项逻辑。
    #   分支 ett_cur / ett_prev（总分=字面 15）  ：不使用任何分项 → 两标记均 0
    #   分支 prev_ett_recompute                  ：COALESCE(当前, 6/5/4)，从不取前值
    #                                            → 有分项缺失则 default=1，carry 恒 0
    #   分支 plain_sum                            ：COALESCE(当前, 前值, 6/5/4)
    #                                            → 逐分项判定 carry / default
    # v1 对三个分支统一套用最后一条规则，导致 default 761 行、carry 1,534 行标记错误。
    cur = (b.gcsmotor, b.gcsverbal, b.gcseyes)
    prv = (p_motor, p_verbal, p_eyes)
    plain_carry = np.zeros(len(b), bool)
    plain_default = np.zeros(len(b), bool)
    for c, p in zip(cur, prv):
        plain_carry |= (c.isna() & p.notna()).to_numpy()
        plain_default |= (c.isna() & p.isna()).to_numpy()
    plain_mask = plain.to_numpy() if hasattr(plain, 'to_numpy') else np.asarray(plain)
    recomp_mask = np.asarray(prev_ett_recompute)
    forced_mask = np.asarray(ett_cur | ett_prev)
    any_missing = np.zeros(len(b), bool)
    for c in cur:
        any_missing |= c.isna().to_numpy()
    used_carry = plain_carry & plain_mask
    used_default = (plain_default & plain_mask) | (any_missing & recomp_mask)
    # forced_mask 两个均为 0（字面 15，不使用分项）
    assert not (used_carry & forced_mask).any() and not (used_default & forced_mask).any()

    total = pd.Series(np.nan, index=b.index)
    total[ett_cur | ett_prev] = 15
    total[prev_ett_recompute] = (b.gcsmotor.fillna(6) + b.gcsverbal.fillna(5) + b.gcseyes.fillna(4))[prev_ett_recompute]
    total[plain] = (b.gcsmotor.fillna(p_motor).fillna(6)
                    + b.gcsverbal.fillna(p_verbal).fillna(5)
                    + b.gcseyes.fillna(p_eyes).fillna(4))[plain]
    b['total_recon'] = total

    b['eyes_recon'] = b.gcseyes.fillna(p_eyes)
    b['verbal_recon'] = b.gcsverbal.fillna(p_verbal)
    b['motor_recon'] = b.gcsmotor.fillna(p_motor)

    # ---- 硬门禁：复现必须与派生层逐格一致 ----
    chk = der.merge(b[['stay_id', 'charttime', 'total_recon', 'eyes_recon', 'verbal_recon',
                       'motor_recon', 'endotrachflag']], on=['stay_id', 'charttime'],
                    how='outer', indicator=True)
    if (chk._merge != 'both').any():
        raise SystemExit('门禁失败：复现层与派生层键不匹配')
    mism = {}
    for a, c in (('gcs_total', 'total_recon'), ('gcs_eyes', 'eyes_recon'),
                 ('gcs_verbal', 'verbal_recon'), ('gcs_motor', 'motor_recon'),
                 ('gcs_unable', 'endotrachflag')):
        x, y = chk[a], chk[c]
        mism[a] = int((~((x.isna() & y.isna()) | np.isclose(x.fillna(-999), y.fillna(-999)))).sum())
    if any(mism.values()):
        raise SystemExit(f'门禁失败：复现层与派生层不一致 {mism}')
    qa_facts['gcs_replication_match'] = {'rows': int(len(chk)), 'mismatch_per_column': mism}
    note(f'GCS 规则复现门禁通过：{len(chk)} 行 × 5 列全部零差异（来源归属因此可信）')

    b = b.drop(columns=['_t', 'p_motor', 'p_verbal', 'p_eyes', 'p_t'])

    # ---- 来源归属标签 ----
    b['gcs_eyes_origin'] = o_eyes
    b['gcs_motor_origin'] = o_motor
    b['gcs_verbal_origin'] = o_verbal
    b['gcs_n_comp_current_record'] = (
        b.gcseyes.notna().astype(int) + b.gcsverbal.notna().astype(int) + b.gcsmotor.notna().astype(int))
    b['gcs_n_comp_numeric_current'] = (
        b.gcseyes.notna().astype(int) + (b.gcsverbal.fillna(-1) > 0).astype(int)
        + b.gcsmotor.notna().astype(int))
    b['gcs_total_rule'] = np.select(
        [ett_cur, ett_prev, prev_ett_recompute],
        ['ett_rule_forced_15_current', 'ett_rule_forced_15_carried', 'prev_ett_recompute'],
        default='plain_sum')
    b['gcs_total_used_default_for_missing'] = used_default.astype(int)
    b['gcs_total_used_carryforward'] = used_carry.astype(int)

    # ---- 限定到原研究观察窗，并保留队列与结局 ----
    g = b.merge(coh[['stay_id', 'subject_id', 'hadm_id', 'intime', 'win_lo',
                     'win_hi', 'label', 'n_icu_stays', 'los_h', 't_inf_minus_intime_h']],
                on='stay_id', how='inner')
    g['gcs_charttime'] = pd.to_datetime(g.charttime)
    g['gcs_time_h'] = (to_ns(g.gcs_charttime) - to_ns(g.intime)) / NS_PER_H
    inwin = (g.gcs_charttime >= g.win_lo) & (g.gcs_charttime < g.win_hi)
    qa_facts['gcs_rows_full_window'] = int(len(g))
    qa_facts['gcs_rows_in_study_window'] = int(inwin.sum())
    g = g[inwin].copy()
    note(f'GCS 时点：全窗 {qa_facts["gcs_rows_full_window"]} → 研究窗 [0,48h) 内 '
         f'{len(g)} 行（窗内起点附近的补值归属仍需窗内前一条记录，故先全窗复现再裁剪）')

    g = g.sort_values(['stay_id', 'gcs_charttime'], kind='stable').reset_index(drop=True)
    # 派生层原样值（与复现结果已验证逐格一致）显式并入，保证"不改写"可追溯
    g = g.merge(
        der.rename(columns={'gcs_total': 'gcs_total_recorded', 'gcs_eyes': 'gcs_eyes_recorded',
                            'gcs_verbal': 'gcs_verbal_recorded', 'gcs_motor': 'gcs_motor_recorded',
                            'gcs_unable': 'gcs_unable_recorded'}),
        on=['stay_id', 'charttime'], how='left', validate='one_to_one')
    if g.gcs_total_recorded.isna().any():
        raise SystemExit('门禁失败：派生层原样值并入后有缺失')
    q_id = g.stay_id.to_numpy()
    q_t = to_ns(g.gcs_charttime)
    out: dict = {}
    iv_diag: dict = {}

    # =================================================================
    # 2. RASS（观测值）—— 只取同一时刻或既往
    # =================================================================
    rass_all = pd.read_csv(UP / 'mimic_09_rass_cam_icu.csv', low_memory=False,
                           usecols=['stay_id', 'charttime', 'assessment_field',
                                    'valuenum', 'value_raw', 'rass_numeric'])
    ra = rass_all[rass_all.assessment_field == 'rass_actual'].copy()
    # 上游 rass_numeric 不完整（-1/+1/+2/+3 未映射），改用 valuenum（完整 -5..+4）
    qa_facts['rass_numeric_upstream_coverage'] = {
        'rass_actual_rows': int(len(ra)),
        'rass_numeric_notna': int(ra.rass_numeric.notna().sum()),
        'valuenum_notna': int(ra.valuenum.notna().sum()),
        'missing_grades_in_rass_numeric': sorted(
            set(ra.valuenum.dropna().unique().tolist())
            - set(ra.rass_numeric.dropna().unique().tolist())),
    }
    note('RASS 用 valuenum（完整）；上游 rass_numeric 缺 -1/+1/+2/+3 档，已记入缺陷')
    ra = ra.rename(columns={'valuenum': 'rass_observed'})[['stay_id', 'charttime', 'rass_observed']]
    lookback_features(out, 'rass', Prior(ra, 'stay_id', 'charttime'), q_id, q_t,
                      ['rass_observed'])

    rg = rass_all[rass_all.assessment_field == 'rass_goal']
    lookback_features(out, 'rass_goal', Prior(rg, 'stay_id', 'charttime'), q_id, q_t,
                      ['valuenum'], within=(4,))

    dl = rass_all[rass_all.assessment_field == 'delirium_assessment']
    lookback_features(out, 'cam_icu', Prior(dl, 'stay_id', 'charttime'), q_id, q_t,
                      ['value_raw'], within=(4, 24))

    # =================================================================
    # 3. 镇静 / 阿片 / 肌松：区间覆盖 + 既往给药 + 未确认医嘱（三层分离）
    # =================================================================
    ie = pd.read_csv(UP / 'mimic_03_drug_inputevents.csv', low_memory=False)
    for short, cls in DRUG_CLASSES.items():
        sub = ie[ie.drug_class == cls]
        iv = Intervals(sub, 'stay_id', 'starttime', 'endtime')
        diag = interval_features(out, f'{short}_iv', iv, q_id, q_t,
                                 ['drug_label', 'statusdescription', 'administration_mode',
                                  'rate', 'rateuom', 'itemid'],
                                 concat_col='drug_label')
        iv_diag[short] = diag
        # 既往最近一次输注结束 / 开始（判断药效残留）
        ei = prior_by_end(iv, q_id, q_t)
        m = ei >= 0
        safe = np.where(m, ei, 0)
        out[f'{short}_last_end_gap_h'] = np.where(m, (q_t - iv.e[safe]) / NS_PER_H, np.nan)
        out[f'{short}_last_start_gap_h'] = np.where(m, (q_t - iv.s[safe]) / NS_PER_H, np.nan)
        out[f'{short}_any_prior'] = m.astype(int)
        for h in (4, 12, 24):
            out[f'{short}_ended_within_{h}h'] = (
                m & ((q_t - iv.e[safe]) / NS_PER_H <= h) & (out[f'{short}_iv_active'] == 0)
            ).astype(int)
        out[f'{short}_records_exist_only_later'] = iv.only_later(
            q_id, q_t, out[f'{short}_iv_active'].astype(bool)).astype(int)

    # eMAR：可确认给药 / 明确未给药 / 未确认，三分层
    em = pd.read_csv(UP / 'mimic_04_drug_emar.csv', low_memory=False,
                     usecols=['stay_id', 'charttime', 'drug_class', 'event_txt',
                              'event_category', 'medication_raw'])
    em_ok = em[em.stay_id.notna()].copy()
    for short, cls in DRUG_CLASSES.items():
        sub = em_ok[em_ok.drug_class == cls]
        for tag, keep, within in (
            ('emar_admin', sub.event_txt.isin(EMAR_ADMIN_TXT), (1, 4, 12, 24)),
            ('emar_not_given', sub.event_txt.isin(EMAR_NOT_GIVEN_TXT), (4, 24)),
            ('emar_other_unconfirmed',
             ~sub.event_txt.isin(EMAR_ADMIN_TXT | EMAR_NOT_GIVEN_TXT), (4, 24)),
        ):
            s = sub[keep]
            if s.empty:
                out[f'{short}_{tag}_prior'] = np.zeros(len(g), dtype=int)
                out[f'{short}_{tag}_gap_h'] = np.nan
                out[f'{short}_{tag}_n_records'] = 0
                continue
            p = Prior(s, 'stay_id', 'charttime')
            idx = p.last_at_or_before(q_id, q_t)
            m = idx >= 0
            safe = np.where(m, idx, 0)
            out[f'{short}_{tag}_prior'] = m.astype(int)
            out[f'{short}_{tag}_gap_h'] = np.where(m, (q_t - p.t[safe]) / NS_PER_H, np.nan)
            out[f'{short}_{tag}_n_records'] = int(len(s))

    # prescriptions：只有医嘱，不能当作已暴露
    pr = pd.read_csv(UP / 'mimic_05_drug_prescriptions.csv', low_memory=False,
                     usecols=['stay_id', 'starttime', 'stoptime', 'drug_class', 'drug_clean'])
    for short, cls in DRUG_CLASSES.items():
        sub = pr[pr.drug_class == cls]
        if sub.empty:
            out[f'{short}_order_unconfirmed_prior'] = np.zeros(len(g), dtype=int)
            continue
        iv = Intervals(sub.assign(endtime=sub.stoptime.fillna(sub.starttime)),
                       'stay_id', 'starttime', 'endtime')
        s = sub.assign(_t=pd.to_datetime(sub.starttime))
        p = Prior(s[['stay_id', '_t', 'drug_clean']], 'stay_id', '_t')
        idx = p.last_at_or_before(q_id, q_t)
        m = idx >= 0
        safe = np.where(m, idx, 0)
        out[f'{short}_order_unconfirmed_prior'] = m.astype(int)
        out[f'{short}_order_unconfirmed_gap_h'] = np.where(m, (q_t - p.t[safe]) / NS_PER_H, np.nan)

    # =================================================================
    # 4. 呼吸支持（分状态，不合并）
    # =================================================================
    vt = pd.read_csv(UP / 'mimic_06_ventilation_derived.csv', low_memory=False,
                     usecols=['stay_id', 'starttime', 'endtime', 'ventilation_status'])
    vt = vt[vt.ventilation_status.notna()]
    vstat_cols = {}
    for st in sorted(vt.ventilation_status.unique()):
        slug = {'InvasiveVent': 'invasive', 'NonInvasiveVent': 'noninvasive',
                'HFNC': 'hfnc', 'SupplementalOxygen': 'supp_o2',
                'Tracheostomy': 'trach'}.get(st, st.lower())
        vstat_cols[st] = f'vent_active_{slug}'
    sub = vt.assign(_c=vt.ventilation_status.map(vstat_cols))
    for st, col in vstat_cols.items():
        s = vt[vt.ventilation_status == st]
        iv = Intervals(s, 'stay_id', 'starttime', 'endtime')
        a_, _ = iv.covering(q_id, q_t)
        out[col] = (a_ >= 0).astype(int)
    anyiv = Intervals(vt, 'stay_id', 'starttime', 'endtime')
    a, ncov_vent = anyiv.covering(q_id, q_t)
    m = a >= 0
    safe = np.where(m, a, 0)
    out['vent_active_any'] = m.astype(int)
    out['vent_active_n_covering'] = ncov_vent.astype(np.int32)
    out['vent_active_status'] = np.where(m, anyiv.ev['ventilation_status'].to_numpy()[safe], None)
    out['vent_active_start_gap_h'] = np.where(m, (q_t - anyiv.s[safe]) / NS_PER_H, np.nan)
    out['vent_active_end_gap_h'] = np.where(m, (q_t - anyiv.e[safe]) / NS_PER_H, np.nan)
    out['vent_basis_covers_timepoint'] = np.where(
        m, ((anyiv.s[safe] <= q_t) & (q_t <= anyiv.e[safe])).astype(np.int8), 1)
    iv_diag['vent'] = {'active': int(m.sum()), 'n_covering_gt1': int((ncov_vent > 1).sum()),
                       'basis_not_covering': 0}
    e = prior_by_end(anyiv, q_id, q_t)
    me = e >= 0
    se = np.where(me, e, 0)
    out['vent_last_end_gap_h'] = np.where(me, (q_t - anyiv.e[se]) / NS_PER_H, np.nan)
    out['vent_any_prior'] = me.astype(int)
    out['vent_records_exist_only_later'] = anyiv.only_later(q_id, q_t, out['vent_active_any'].astype(bool)).astype(int)

    # =================================================================
    # 5. 生理变量：精确观察时间 + 单位 + 缺失状态（不聚合、不插补）
    # =================================================================
    vs = pd.read_csv(OUT / 'mimic_12_vitals_timeseries.csv', low_memory=False)
    vs = vs[vs.stay_id.isin(set(coh.stay_id))]
    for var, unit_col in (('mbp', 'unit_mbp'), ('resp_rate', 'unit_resp_rate'),
                          ('temperature', 'unit_temperature')):
        s = vs[vs[var].notna()][['stay_id', 'charttime', var, unit_col]].rename(
            columns={var: 'value', unit_col: 'unit'})
        lookback_features(out, f'phys_{var}', Prior(s, 'stay_id', 'charttime'), q_id, q_t,
                          ['value', 'unit'])

    ph = pd.read_csv(OUT / 'mimic_13_bg_ph.csv', low_memory=False)
    ph = ph[ph.stay_id.isin(set(coh.stay_id))]
    s = ph[ph.ph.notna()][['stay_id', 'charttime', 'ph', 'unit_ph']].rename(
        columns={'ph': 'value', 'unit_ph': 'unit'})
    lookback_features(out, 'phys_ph', Prior(s, 'stay_id', 'charttime'), q_id, q_t,
                      ['value', 'unit'])

    # =================================================================
    # 6. 组装与落盘
    # =================================================================
    head = pd.DataFrame({
        'row_key': g.stay_id.astype(str) + '|' + g.gcs_charttime.dt.strftime('%Y-%m-%d %H:%M:%S'),
        'subject_id': g.subject_id, 'hadm_id': g.hadm_id, 'stay_id': g.stay_id,
        'gcs_charttime': g.gcs_charttime.dt.strftime('%Y-%m-%d %H:%M:%S'),
        'gcs_time_h': g.gcs_time_h.round(4),
        'gcs_seq': g.gcs_seq,
        'label_hospital_mortality': g.label,
        'n_icu_stays': g.n_icu_stays,
        'gcs_total_recorded': g.gcs_total_recorded,
        'gcs_eyes_recorded': g.gcs_eyes_recorded,
        'gcs_verbal_recorded': g.gcs_verbal_recorded,
        'gcs_motor_recorded': g.gcs_motor_recorded,
        'gcs_unable_recorded': g.gcs_unable_recorded,
        'gcs_total_minus_component_sum': (g.gcs_total_recorded
                                          - (g.gcs_eyes_recorded + g.gcs_verbal_recorded
                                             + g.gcs_motor_recorded)),
        'gcs_eyes_origin': g.gcs_eyes_origin,
        'gcs_verbal_origin': g.gcs_verbal_origin,
        'gcs_motor_origin': g.gcs_motor_origin,
        'gcs_n_comp_current_record': g.gcs_n_comp_current_record,
        'gcs_n_comp_numeric_current': g.gcs_n_comp_numeric_current,
        'gcs_has_prev_within_6h': g.has_prev_le6h,
        'gcs_total_rule': g.gcs_total_rule,
        'gcs_total_used_default_for_missing': g.gcs_total_used_default_for_missing,
        'gcs_total_used_carryforward': g.gcs_total_used_carryforward,
    })
    tbl = pd.concat([head, pd.DataFrame(out, index=g.index)], axis=1)
    tbl.to_csv(BASE / 'mimic_gcs_timebase.csv', index=False)
    note(f'底表已写出：base/mimic_gcs_timebase.csv  {tbl.shape[0]} 行 × {tbl.shape[1]} 列')

    qa_facts['timebase_rows'] = int(tbl.shape[0])
    qa_facts['timebase_cols'] = int(tbl.shape[1])

    # ---- 分支级标记自检（v2 新增）----
    # 逐分支验证"总分"与"输出分项"的任何自洽关系。这是对**输出底表**本身的检查，
    # 不依赖构建过程中的中间量，因此能真正验证 gcs_total_rule 的分支归类。
    #   分支 1) 字面 15            ：总分必须恰为 15，不使用任何分项
    #   分支 2) COALESCE(当前,6/5/4) ：缺失按**当前记录**判定（从不取前值）
    #   分支 3) COALESCE(当前,前值,6/5/4)：缺失须按**输出分项**判定
    cur_sum = g[['gcsmotor', 'gcsverbal', 'gcseyes']].sum(axis=1, skipna=True)
    rec_sum = g[['gcs_motor_recorded', 'gcs_verbal_recorded',
                 'gcs_eyes_recorded']].sum(axis=1, skipna=True)

    f15 = g.gcs_total_rule.isin(['ett_rule_forced_15_current',
                                 'ett_rule_forced_15_carried']).to_numpy()
    v_forced = int((f15 & (g.gcs_total_recorded != 15)).sum())

    rc = (g.gcs_total_rule == 'prev_ett_recompute').to_numpy()
    exp_rc = (6 * g.gcsmotor.isna() + 5 * g.gcsverbal.isna() + 4 * g.gcseyes.isna())
    v_rc = int((rc & (np.abs((g.gcs_total_recorded - cur_sum).fillna(-999)
                             - exp_rc) > 1e-9)).sum())

    ps = (g.gcs_total_rule == 'plain_sum').to_numpy()
    exp_ps = (6 * g.gcs_motor_recorded.isna() + 5 * g.gcs_verbal_recorded.isna()
              + 4 * g.gcs_eyes_recorded.isna())
    v_ps = int((ps & (np.abs((g.gcs_total_recorded - rec_sum).fillna(-999)
                             - exp_ps) > 1e-9)).sum())

    qa_facts['flag_selfcheck'] = {
        'branch_rows': g.gcs_total_rule.value_counts().astype(int).to_dict(),
        'identity_violations': {'forced15_total_not_15': v_forced,
                                'prev_ett_recompute': v_rc,
                                'plain_sum': v_ps},
        'default_flag_rows': int(g.gcs_total_used_default_for_missing.sum()),
        'carry_flag_rows': int(g.gcs_total_used_carryforward.sum()),
        'default_flag_by_branch': g.groupby('gcs_total_rule')[
            'gcs_total_used_default_for_missing'].sum().astype(int).to_dict(),
        'carry_flag_by_branch': g.groupby('gcs_total_rule')[
            'gcs_total_used_carryforward'].sum().astype(int).to_dict(),
    }
    if v_forced or v_rc or v_ps:
        raise SystemExit(f'门禁失败：分支恒等式违反 forced={v_forced} '
                         f'recompute={v_rc} plain={v_ps}')
    note(f'标记自检通过：三分支恒等式零违反（强制15={int(f15.sum())} / '
         f'重算={int(rc.sum())} / 直和={int(ps.sum())} 行）；'
         f'default 标记 {int(g.gcs_total_used_default_for_missing.sum())} 行、'
         f'carry 标记 {int(g.gcs_total_used_carryforward.sum())} 行')
    qa_facts['interval_selection'] = iv_diag
    note('区间依据自检：' + '；'.join(
        f'{k} active={v["active"]} 依据不覆盖={v["basis_not_covering"]} 并发>1={v["n_covering_gt1"]}'
        for k, v in iv_diag.items()))

    (QA / 'build_facts.json').write_text(
        json.dumps(qa_facts, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    (QA / 'build_notes.md').write_text('\n'.join(f'- {n}' for n in notes) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
