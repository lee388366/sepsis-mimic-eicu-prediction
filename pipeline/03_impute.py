import pandas as pd
from scipy.interpolate import interp1d  # scipy 的插值工具：根据已知点，推算未知点的值

# 读取 02 步提取的小时级数据（宽表：每行 = 一个患者的一小时）
df = pd.read_csv('sepsis_hourly.csv')

# =====================================================================
# 固定值表（照抄 Liu 源码 HCV_Var_Impute_Value）
# 用途：某个变量在某患者 48 小时里「一个观测值都没有」时，用它兜底。
# 数值 = 临床生理正常值（正常成人典型水平）。
# =====================================================================
FIXED = {  # 变量名 -> 固定填充值
    'heart_rate':   86,     # 心率，正常 60-100，取 86
    'sbp':          118,    # 收缩压，正常 <120，取 118
    'dbp':          59,     # 舒张压，正常 <80，取 59
    'mbp':          77,     # 平均动脉压，正常 70-100，取 77
    'resp_rate':    19,     # 呼吸频率，正常 12-20，取 19
    'spo2':         98,     # 血氧饱和度，正常 95-100%，取 98
    'temperature':  36,     # 体温，正常 ~36-37°C，取 36
    'glucose':      128,    # 血糖，正常上限附近
    'fio2':         21,     # 吸入氧浓度，空气就是 21%（没上氧）；MIMIC bg.fio2 与 eICU respiratorycharting 均为百分数
    'ph':           7.4,    # 动脉血 pH，正常 7.35-7.45，取 7.4
    'gcs':          12,     # 格拉斯哥昏迷评分总分，正常 15，取 12（偏低）——照 Liu
    'gcs_eyes':     4,      # GCS 睁眼，正常 4
    'gcs_motor':    5,      # GCS 运动，正常 6，取 5——照 Liu
    'gcs_verbal':   4,      # GCS 语言，正常 5，取 4——照 Liu
}

# 需要插值处理的 HCV 变量列表（顺序可自行调整）
HCV_VARS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2',
            'temperature', 'glucose', 'fio2', 'ph',
            'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal']

# 每个患者 48 小时的完整小时轴（0 到 47）
ALL_HOURS = list(range(48))

# =====================================================================
# 按患者逐个处理（一个 stay 就是一个患者的一次 ICU 住院）
# =====================================================================
for stay_id, stay in df.groupby('stay_id'):
    idx = stay.index  # 这一组在原表里的行号，写回时用

    # 对每个 HCV 变量独立判断该患者用哪种方式填充
    for var in HCV_VARS:
        vals = stay[var]                    # 该变量的 48 个小时的值（含 NaN）
        n_obs = vals.notna().sum()          # 统计这个变量在这患者身上有几个真实观测值

        if n_obs == 0:
            # ① 情况一：48 小时里这个变量一次都没记录
            #     → 用固定值（生理正常值）整体填充
            df.loc[idx, var] = FIXED[var]

        elif n_obs < 24:
            # ② 情况二：有记录但不足一半小时（<24 个观测）
            #     → 用「该患者该变量的均值」填充
            df.loc[idx, var] = vals.mean()

        else:
            # ③ 情况三：记录过半（≥24 个观测）
            #     → nearest 插值：每个空的小时取「时间上最接近的非空值」
            obs = vals.notna()                      # 这个变量有观测值的位置
            hours_obs = stay.loc[obs, 'hr'].sort_values()   # 对应的小时（已排序）
            interp = interp1d(hours_obs, vals[obs],
                              kind='nearest',
                              bounds_error=False,
                              fill_value='extrapolate')
            df.loc[idx, var] = interp(ALL_HOURS)

# 保存填补后的结果
df.to_csv('sepsis_hourly_imputed_v2.csv', index=False)