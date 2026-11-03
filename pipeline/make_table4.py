import pandas as pd

# =====================================================================
# make_table4.py —— 生成 tables/table4_missing_rate.csv
#
# 数据源: sepsis_hourly.csv（02_extract_hourly.sql 输出，插值前的原始
#         小时级宽表；每行 = 一个患者的一小时，含缺失 NaN）
#
# 列含义:
#   Variable                         临床可读变量名
#   Missing cells (n)                「患者×小时」单元缺失数（HCV 变量）
#   Missing cells (%)                单元缺失率 = 缺失单元 / 总单元
#   Patients completely missing (n)  48h 内完全无观测的患者数
#   Patients completely missing (%)  = 上列 / 患者总数
#   Mean obs per patient (of 48h)    每患者平均观测小时数（满值 48）
#
# 口径说明:
#   - 14 个 HCV 变量报告 单元级 + 患者级 + 平均观测数；
#   - Weight / Height 是患者级常数（同一患者所有小时行同值），
#     只报告患者级缺失率，单元级填 "-"。
#   - cap_refill 在 MIMIC-IV 无有效记录（100% 缺失），已从特征集中
#     排除，因此不在表中。
#
# 输出: tables/table4_missing_rate.csv
# =====================================================================

HCV = {
    'heart_rate':  'Heart rate (bpm)',
    'sbp':         'Systolic BP (mmHg)',
    'dbp':         'Diastolic BP (mmHg)',
    'mbp':         'Mean BP (mmHg)',
    'resp_rate':   'Respiratory rate (/min)',
    'spo2':        'SpO2 (%)',
    'temperature': 'Temperature (°C)',
    'glucose':     'Glucose (mg/dL)',
    'ph':          'pH',
    'fio2':        'FiO2 (fraction)',
    'gcs':         'GCS total',
    'gcs_eyes':    'GCS eyes',
    'gcs_motor':   'GCS motor',
    'gcs_verbal':  'GCS verbal',
}
SV = {
    'weight': 'Weight (kg)',
    'height': 'Height (cm)',
}


def fmt_thousands(x):
    """千分位逗号；值小于 1000 时按现有格式不补零"""
    if x < 1000:
        return str(x)
    return f'{x:,}'


def main():
    df = pd.read_csv('sepsis_hourly.csv')
    n_pat = df['stay_id'].nunique()
    n_cells = len(df)
    print(f'数据: {len(df):,} 行 × {n_pat:,} 患者')

    rows = []
    for var, label in HCV.items():
        cell_miss = int(df[var].isna().sum())
        cell_pct = 100 * cell_miss / n_cells
        # 患者级：该患者 48 小时内该变量全部缺失
        pat_miss = int(df.groupby('stay_id')[var].apply(
            lambda s: s.notna().sum() == 0).sum())
        pat_pct = 100 * pat_miss / n_pat
        mean_obs = df.groupby('stay_id')[var].apply(
            lambda s: s.notna().sum()).mean()
        rows.append([
            label,
            fmt_thousands(cell_miss), f'{cell_pct:.2f}',
            fmt_thousands(pat_miss), f'{pat_pct:.2f}',
            f'{mean_obs:.1f}',
        ])

    for var, label in SV.items():
        pat_miss = int(df.groupby('stay_id')[var].apply(
            lambda s: s.notna().sum() == 0).sum())
        pat_pct = 100 * pat_miss / n_pat
        rows.append([label, '-', '-', fmt_thousands(pat_miss),
                     f'{pat_pct:.2f}', '-'])

    out = pd.DataFrame(rows, columns=[
        'Variable', 'Missing cells (n)', 'Missing cells (%)',
        'Patients completely missing (n)', 'Patients completely missing (%)',
        'Mean obs per patient (of 48h)',
    ])
    out.to_csv('tables/table4_missing_rate.csv', index=False)
    print('\n已保存: tables/table4_missing_rate.csv')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
