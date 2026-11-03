#!/usr/bin/env python3
"""s08_explain.py —— SHAP / 注意力 / 遮挡忠实性 / 时间窗消融（全部基于新训练产物）

用法: python s08_explain.py A|B

纪律:
  * 所有模型权重来自 models_trainonly/<V>/（新训练），不加载任何旧权重。
  * 标准化一律使用新训练集拟合的 scaler。
  * **首小时/单点消融的既有局限必须保留**：完整 48 小时张量的插补是
    患者内（patient-local）插补，因此"第 1 小时"取值仍受全窗插补规则影响。
    该消融只能作为 **表示敏感性分析**，不得据此声称"真实早期预测"。
  * 注意力只用于 **模型行为分析**，不构成因果生理机制解释。

输出:
  tables/<V>/shap_top_features.csv
  data/shap_values_<V>.npz
  tables/<V>/table8c_attention_faithfulness.csv
  tables/<V>/table6_timepoint_ablation.csv
  qa/s08_explain_<V>.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent.parent.parent

SEED = 42
KS = [1, 3, 6, 12, 24]
NSEED = 20
HCV = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2', 'temperature',
       'glucose', 'fio2', 'ph', 'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']
CLIN = {'heart_rate': 'Heart Rate', 'sbp': 'Systolic BP', 'dbp': 'Diastolic BP',
        'mbp': 'Mean BP', 'resp_rate': 'Respiratory Rate', 'spo2': 'SpO2',
        'temperature': 'Temperature', 'glucose': 'Glucose', 'fio2': 'FiO2',
        'ph': 'pH', 'gcs': 'GCS Score', 'gcs_eyes': 'GCS Eyes',
        'gcs_motor': 'GCS Motor', 'gcs_verbal': 'GCS Verbal'}


class AttentionPool(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = torch.nn.Linear(dim, 1)

    def forward(self, h):
        att = torch.softmax(self.W(h).squeeze(-1), dim=1)
        return torch.bmm(att.unsqueeze(1), h).squeeze(1), att


class SepsisLSTM(torch.nn.Module):
    def __init__(self, n_seq=14, n_static=4, hidden=64, drop=0.35):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_seq, hidden, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(2 * hidden)
        self.drop = torch.nn.Dropout(drop)
        self.static = torch.nn.Sequential(torch.nn.Linear(n_static, 16), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * hidden + 16, 64), torch.nn.ReLU(),
                                        torch.nn.Dropout(drop), torch.nn.Linear(64, 1))

    def forward(self, xs, xsv, return_attn=False):
        out, _ = self.lstm(xs)
        ctx, att = self.attn(out)
        logit = self.head(torch.cat([self.drop(ctx), self.static(xsv)], 1)).squeeze(1)
        return (logit, att) if return_attn else logit


def load_variant(V):
    d = np.load(HERE / 'data' / 'trainonly_mimic.npz', allow_pickle=False)
    X_hcv, X_sv, X_flat = d['X_hcv'], d['X_sv'], d['X_flat']
    y, subj, split = d['y'].astype(np.int64), d['subject_id'].astype(np.int64), d['split'].astype(str)
    tinf = d['t_inf_h'].astype(float)
    mask = np.ones(len(y), bool) if V == 'A' else (tinf <= 48)
    return (X_hcv[mask], X_sv[mask], X_flat[mask], y[mask], subj[mask], split[mask])


def main() -> int:
    V = (sys.argv[1] if len(sys.argv) > 1 else 'A').upper()
    TAB, QA, DATA = HERE / 'tables' / V, HERE / 'qa', HERE / 'data'
    MD = HERE / 'models_trainonly' / V
    for x in (TAB, QA, DATA):
        x.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = 'cpu'

    X_hcv, X_sv, X_flat, y, subj, split = load_variant(V)
    tr, va, te = (np.flatnonzero(split == s) for s in ('train', 'val', 'test'))
    print(f'变体 {V}: n={len(y)} | train {len(tr)} val {len(va)} test {len(te)}')

    meta = {'variant': V, 'n_test': int(len(te)), 'n_test_subjects': int(len(np.unique(subj[te]))),
            'new_weights_only': True, 'seed': SEED}

    # ================= 1. RF SHAP（新 RF 权重）=================
    sc_flat = joblib.load(MD / 'scaler_flat.joblib')
    rf = joblib.load(MD / 'model_RandomForest.joblib')
    names = []
    for v in HCV:
        names += [f'{CLIN[v]} (mean)', f'{CLIN[v]} (min)', f'{CLIN[v]} (max)']
    names += ['Age', 'Gender', 'Weight', 'Height']
    F_te = sc_flat.transform(X_flat[te])
    import shap
    t0 = time.time()
    # 显式使用 tree_path_dependent：对随机森林等价且远快于 interventional
    expl = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
    sv = expl.shap_values(F_te, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n, p, 2) → 取正类
        sv = sv[:, :, 1]
    print(f'SHAP 完成: {sv.shape} ({time.time()-t0:.1f}s)')
    imp = np.abs(sv).mean(axis=0)
    top = pd.DataFrame({'feature': names, 'mean_abs_shap': imp}) \
        .sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    top['rank'] = np.arange(1, len(top) + 1)
    top.to_csv(TAB / 'shap_top_features.csv', index=False)
    np.savez(DATA / f'shap_values_{V}.npz', shap=sv.astype(np.float32),
             X_std=F_te.astype(np.float32), feature_names=np.array(names),
             stay_id=np.load(HERE / 'data' / f'preds_{V}.npz')['stay_id'][te].astype(np.int64),
             y_true=y[te])
    print('\n===== SHAP Top-10 =====')
    print(top.head(10).round(4).to_string(index=False))
    meta['shap'] = {'n_features': len(names), 'top10': top.head(10).to_dict('records')}

    # ================= 2. LSTM 注意力（新权重）=================
    sc_seq = joblib.load(MD / 'scaler_seq.joblib')
    sc_sv = joblib.load(MD / 'scaler_sv.joblib')
    net = SepsisLSTM()
    net.load_state_dict(torch.load(MD / 'model_LSTM.pt', map_location=DEVICE))
    net.eval()

    def prep(Xh, Xs):
        seq = Xh.transpose(0, 2, 1)
        A = sc_seq.transform(seq.reshape(-1, len(HCV))).reshape(
            len(Xh), 48, len(HCV)).astype(np.float32)
        B = sc_sv.transform(Xs).astype(np.float32)
        return A, B

    A_te, B_te = prep(X_hcv[te], X_sv[te])
    xs = torch.tensor(A_te)
    xsv = torch.tensor(B_te)
    with torch.no_grad():
        logit, att = net(xs, xsv, return_attn=True)
    p_te = torch.sigmoid(logit).numpy()
    att = att.numpy()
    y_te = y[te]
    ba, bp = roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te)
    mean_att = att.mean(axis=0)
    top3 = np.sort(att, 1)[:, ::-1][:, :3].sum(1).mean()
    print(f'\nLSTM（新权重）test AUC={ba:.4f} AUPRC={bp:.4f}')
    print(f'注意力集中度: 每样本 top-3 权重和均值={top3:.3f}（均匀={3/48:.3f}）')
    print(f'平均注意力 top5 小时: {np.argsort(mean_att)[-5:][::-1].tolist()}')
    meta['attention'] = {'baseline_auc': round(ba, 4), 'baseline_auprc': round(bp, 4),
                         'top3_share_mean': round(float(top3), 4),
                         'mean_att_top5_hours': [int(h) for h in np.argsort(mean_att)[-5:][::-1]],
                         'caveat': '注意力仅用于模型行为分析，不是因果机制证据'}

    # ================= 3. 遮挡忠实性（attention occlusion）=================
    all_h = np.arange(48)
    n_te = len(te)
    B = 512

    def masked_prob(keep):
        out = np.empty(n_te)
        for i in range(0, n_te, B):
            m = np.zeros((min(B, n_te - i), 48), dtype=np.float32)
            kk = keep[i:i + B]
            for r in range(m.shape[0]):
                m[r, kk[r]] = 1.0
            with torch.no_grad():
                o, _ = net(torch.tensor(A_te[i:i + B]) * torch.tensor(m)[..., None],
                           torch.tensor(B_te[i:i + B]), return_attn=True)
                out[i:i + m.shape[0]] = torch.sigmoid(o).numpy()
        return out

    def run(p):
        return roc_auc_score(y_te, p), average_precision_score(y_te, p)

    rows = []
    for k in KS:
        keep_glob = np.tile(all_h[~np.isin(all_h, np.argsort(mean_att)[-k:])], (n_te, 1))
        per_top = np.argsort(att, 1)[:, ::-1][:, :k]
        keep_pt = np.array([all_h[~np.isin(all_h, per_top[i])] for i in range(n_te)])
        a_glo, p_glo = run(masked_prob(keep_glob))
        a_pt, p_pt = run(masked_prob(keep_pt))
        ra, rp = [], []
        for s in range(NSEED):
            rng = np.random.default_rng(SEED + s)
            keep_r = np.stack([np.delete(all_h, rng.choice(48, k, replace=False))
                               for _ in range(n_te)])
            a_r, p_r = run(masked_prob(keep_r))
            ra.append(a_r); rp.append(p_r)
        ra, rp = np.array(ra), np.array(rp)
        z = ((ba - a_pt) - (ba - ra.mean())) / ra.std() if ra.std() > 0 else np.nan
        rows.append({'k_hours': k,
                     'AUC_global_top': a_glo, 'AUC_per_sample_top': a_pt,
                     'AUC_random_mean': ra.mean(), 'AUC_random_sd': ra.std(),
                     'AUPRC_global_top': p_glo, 'AUPRC_per_sample_top': p_pt,
                     'AUPRC_random_mean': rp.mean(), 'AUPRC_random_sd': rp.std(),
                     'z_per_sample_top_vs_random': z})
        print(f'  k={k:>2} | 逐样本top ΔAUC={ba-a_pt:+.4f} | 全局top ΔAUC={ba-a_glo:+.4f} '
              f'| 随机 ΔAUC={ba-ra.mean():+.4f}±{ra.std():.4f} | z={z:+.2f}')
    auff = pd.DataFrame(rows)
    auff.to_csv(TAB / 'table8c_attention_faithfulness.csv', index=False)
    meta['occlusion'] = {'baseline_auc': round(ba, 4), 'n_random_seeds': NSEED,
                         'ks': KS,
                         'caveat': '遮挡值=0（训练集均值）；遮挡会破坏校准，故只报 AUC/AUPRC'}

    # ================= 4. 时间窗消融 =================
    RF_KW = dict(n_estimators=200, max_depth=15, class_weight='balanced',
                 n_jobs=-1, random_state=SEED)

    def agg(Xh, lo, hi):
        s = Xh[:, :, lo:hi]
        parts = []
        for j in range(s.shape[1]):
            parts += [s[:, j, :].mean(1), s[:, j, :].min(1), s[:, j, :].max(1)]
        return np.column_stack(parts)

    def agg_point(Xh, hour):
        return Xh[:, :, hour]

    sv4 = X_sv
    variants = {
        'A_single_last_hour_47': agg_point(X_hcv, 47),
        'B_single_first_hour_0': agg_point(X_hcv, 0),
        'C_last24h_window_24_47': agg(X_hcv, 24, 48),
        'D_full48h_window_0_47': agg(X_hcv, 0, 48),
    }
    abl = []
    for tag, Fx in variants.items():
        M = np.column_stack([Fx, sv4])
        sc = StandardScaler().fit(M[tr])
        clf = RandomForestClassifier(**RF_KW).fit(sc.transform(M[tr]), y[tr])
        pv = clf.predict_proba(sc.transform(M[te]))[:, 1]
        abl.append({'variant': tag, 'n_features': M.shape[1], 'model': 'RandomForest',
                    'AUC': round(roc_auc_score(y_te, pv), 4),
                    'AUPRC': round(average_precision_score(y_te, pv), 4)})
    abl.append({'variant': 'E_full_sequence_LSTM (new weights)', 'n_features': 48 * 14,
                'model': 'LSTM', 'AUC': round(ba, 4), 'AUPRC': round(bp, 4)})
    ab = pd.DataFrame(abl)
    ab.to_csv(TAB / 'table6_timepoint_ablation.csv', index=False)
    print('\n===== 时间窗消融（同一测试集）=====')
    print(ab.to_string(index=False))
    meta['timepoint_ablation'] = {
        'variants': ab.to_dict('records'),
        'caveat': ('首小时/单点变体取自患者内插补后的完整 48h 张量，仍受全窗插补规则影响；'
                   '只能作为表示敏感性分析，不得据此声称真实早期预测能力')}

    (QA / f's08_explain_{V}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2,
                                                        default=float), encoding='utf-8')
    print(f'\n-> tables/{V}/ 与 qa/s08_explain_{V}.json 已写出')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
