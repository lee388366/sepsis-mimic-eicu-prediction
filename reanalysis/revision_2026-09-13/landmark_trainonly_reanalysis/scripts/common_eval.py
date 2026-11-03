"""公共评估工具：DeLong、校准指标、DCA、聚类 bootstrap。

本模块只做计算，不读数据、不写文件，供 s04/s05/s07/s08 复用。
口径刻意与旧脚本保持可解释的一致：
  - 校准 slope/intercept/CITL 用 logistic recalibration: logit(Y) = a + b*logit(p)
  - CITL 为斜率固定 1 的 offset 模型截距
  - DCA 用 Vickers 净获益公式
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


# ---------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_auc_cov(y: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray]:
    """返回 (AUC, 方差协方差矩阵 kxk)，scores 形状 [k, n]。"""
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    k, n = scores.shape
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    m, nn = len(pos), len(neg)

    aucs = np.empty(k)
    v01 = np.empty((k, m))
    v10 = np.empty((k, nn))
    for r in range(k):
        s = scores[r]
        x = s[pos]
        yv = s[neg]
        # psi(X_i) = 1/n * sum_j [1{X_i > Y_j} + 0.5*1{X_i == Y_j}]
        cmp = ((x[:, None] > yv[None, :]).astype(float)
               + 0.5 * (x[:, None] == yv[None, :]).astype(float))
        v01[r] = cmp.mean(axis=1)
        # psi(Y_j) = 1/m * sum_i [1{X_i > Y_j} + 0.5*1{X_i == Y_j}]
        v10[r] = cmp.mean(axis=0)
        aucs[r] = v01[r].mean()

    s01 = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    s01 = np.atleast_2d(s01)
    s10 = np.atleast_2d(s10)
    cov = s01 / m + s10 / nn
    return aucs, np.atleast_2d(cov)


def f3(v) -> str:
    """统一的显示口径：先按 4 位小数归整，再四舍五入（half-up）保留 3 位。

    为什么需要：Python float、np.float64.__format__ 与 pandas 在 .xxx5
    边界上会给出不同结果（例：0.7935351540… → '0.794' 或 '0.793'，
    0.1855 → '0.186' 或 '0.185'）。表格、图内标注与稿件必须完全一致，
    因此所有对外显示的数字都走本函数（按十进制字符串判定，不受二进制
    表示影响）。
    """
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(round(float(v), 4)))
    return str(d.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


def f2(v) -> str:
    """同上，保留 2 位小数。"""
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(round(float(v), 4)))
    return str(d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """两模型 AUC 的 DeLong 检验（双侧）。"""
    aucs, cov = delong_auc_cov(y, np.vstack([p_a, p_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {'auc_a': aucs[0], 'auc_b': aucs[1], 'z': np.nan, 'p': np.nan}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    from scipy.stats import norm
    return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]),
            'auc_diff': float(aucs[0] - aucs[1]),
            'z': float(z), 'p': float(2 * (1 - norm.cdf(abs(z)))),
            'se_diff': float(np.sqrt(var))}


# ------------------------------------------------------------ 校准指标
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def invlogit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict:
    """slope / intercept / CITL / Brier（logistic recalibration 口径）。

    slope<1 表示预测过于极端；intercept 为 a；CITL 为固定斜率 1 时的校准偏移。
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    lp = _logit(p)
    out = {'n': int(len(y)), 'deaths': int(y.sum()),
           'prevalence': float(y.mean()), 'brier': float(brier_score_loss(y, p))}
    try:
        fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        a, b = float(fit.params[0]), float(fit.params[1])
        ci = fit.conf_int()
        out.update(slope=b, intercept=a,
                   slope_ci_low=float(ci[1][0]), slope_ci_high=float(ci[1][1]),
                   intercept_ci_low=float(ci[0][0]), intercept_ci_high=float(ci[0][1]),
                   slope_se=float(fit.bse[1]), intercept_se=float(fit.bse[0]))
    except Exception:
        out.update(slope=np.nan, intercept=np.nan, slope_ci_low=np.nan,
                   slope_ci_high=np.nan, intercept_ci_low=np.nan,
                   intercept_ci_high=np.nan, slope_se=np.nan, intercept_se=np.nan)
    try:
        citl = sm.Logit(y, np.ones_like(y, dtype=float), offset=lp).fit(disp=0)
        c = citl.conf_int()[0]
        out.update(citl=float(citl.params[0]), citl_ci_low=float(c[0]),
                   citl_ci_high=float(c[1]), citl_se=float(citl.bse[0]))
    except Exception:
        out.update(citl=np.nan, citl_ci_low=np.nan, citl_ci_high=np.nan, citl_se=np.nan)
    return out


def fit_logistic_recal(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """返回 (a, b)，logit(Y) = a + b*logit(p)。"""
    fit = sm.Logit(np.asarray(y).astype(int), sm.add_constant(_logit(p))).fit(disp=0)
    return float(fit.params[0]), float(fit.params[1])


# ------------------------------------------------------------------ DCA
def dca_net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = (p >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        nb[i] = tp / n - fp / n * (t / (1 - t))
    return nb


def dca_treat_all(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prev = float(np.asarray(y).astype(int).mean())
    return prev - (1 - prev) * thresholds / (1 - thresholds)


def clinical_impact(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (高危人数, 其中真死亡人数)。"""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    flagged = np.array([int((p >= t).sum()) for t in thresholds])
    hits = np.array([int(np.sum((p >= t) & (y == 1))) for t in thresholds])
    return flagged, hits


# ------------------------------------------------------- 聚类 bootstrap
def cluster_batches(cluster_ids: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """有放回抽取聚类单元；同一单元的所有记录整体进入/退出。"""
    ids = np.asarray(cluster_ids)
    groups: dict = {}
    for i, g in enumerate(ids):
        groups.setdefault(g, []).append(i)
    keys = list(groups.keys())
    recs = [np.asarray(groups[k], dtype=int) for k in keys]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        out.append(np.concatenate([recs[j] for j in pick]))
    return out


def boot_ci(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                     batches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """同一次抽样内同时算两模型指标并作差。点估计应另取原样本。"""
    da = np.empty(len(batches))
    dp = np.empty(len(batches))
    for i, idx in enumerate(batches):
        da[i] = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
        from sklearn.metrics import average_precision_score
        dp[i] = (average_precision_score(y[idx], p_a[idx])
                 - average_precision_score(y[idx], p_b[idx]))
    return da, dp


def auc_ap_batches(y: np.ndarray, p: np.ndarray, batches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import average_precision_score
    a = np.empty(len(batches))
    pr = np.empty(len(batches))
    for i, idx in enumerate(batches):
        a[i] = roc_auc_score(y[idx], p[idx])
        pr[i] = average_precision_score(y[idx], p[idx])
    return a, pr
