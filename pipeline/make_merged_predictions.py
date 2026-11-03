# make_merged_predictions.py —— 合并基线 + LSTM 预测为一个 npz
import numpy as np

a = np.load('baseline_predictions.npz')     # A 轨：6 基线
b = np.load('sequence_predictions.npz')     # B 轨：LSTM

# 关键断言：两轨 test set 必须是同一批患者，否则指标不可比
assert (a['y_true'] == b['y_true']).all(), 'A/B 轨 test set 不一致！'
assert (a['stay_id'] == b['stay_id']).all(), 'A/B 轨 stay_id 不一致！'

merged = {k: a[k] for k in a.files}         # 保留全部基线 key
merged['LSTM'] = b['LSTM']                  # 追加 LSTM
np.savez('all_predictions.npz', **merged)
print(f'已生成 all_predictions.npz：{len(a.files)} 基线 + LSTM = {len(merged)} 个 key')
print('A/B 轨 test set 一致 ✅')