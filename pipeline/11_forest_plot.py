import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ['LSTM', 'RF']
MODEL_STYLE = {
    'LSTM': dict(marker='o', color='#c0392b', label='LSTM'),
    'RF':   dict(marker='s', color='#2471a3', label='Random Forest'),
}
AUC_REF = 0.7

# 两库共用的完整行集（缺失的组留空位，保证跨库 y 轴严格对齐）
RACE_ORDER = ['White', 'Black', 'Hispanic', 'Asian',
              'Other', 'Other/Unknown', 'Unknown/Refused']


def load_race(file):
    df = pd.read_csv(file)
    df = df[df['Subgroup'].str.startswith('Race')].copy()
    df['Race'] = df['Subgroup'].str.replace('Race: ', '', regex=False)
    for mdl in MODELS:
        ci = df[f'{mdl}_95CI'].astype(str)
        df[f'{mdl}_lo'] = ci.str.split('-').str[0].astype(float)
        df[f'{mdl}_hi'] = ci.str.split('-').str[1].astype(float)
    return df.set_index('Race')


def panel(ax, data, title):
    for mdl in MODELS:
        style = MODEL_STYLE[mdl]
        for race in RACE_ORDER:
            if race not in data.index:
                continue
            row = data.loc[race]
            y = RACE_ORDER.index(race)
            auc = row[f'{mdl}_AUC']
            ax.errorbar(auc, y,
                        xerr=[[auc - row[f'{mdl}_lo']],
                              [row[f'{mdl}_hi'] - auc]],
                        fmt=style['marker'], color=style['color'],
                        markersize=7, capsize=3, linewidth=1.2, zorder=3)
        ax.plot([], [], style['marker'], color=style['color'],
                label=style['label'], markersize=7)

    ax.axvline(AUC_REF, ls='--', lw=1.0, color='gray', zorder=1)
    ax.axvline(0.5, ls=':', lw=0.8, color='lightgray', zorder=1)
    ax.set_yticks(range(len(RACE_ORDER)))
    ax.set_yticklabels(RACE_ORDER)
    ax.invert_yaxis()          # 惯例：参考组/最大组 White 放顶部
    ax.set_xlim(0.40, 0.95)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('AUC (95% CI)')
    ax.grid(axis='x', alpha=0.3, zorder=0)

    for race in RACE_ORDER:
        if race not in data.index:
            continue
        row = data.loc[race]
        y = RACE_ORDER.index(race)
        ax.text(0.96, y, f'n={int(row["n"])}', ha='left', va='center',
                fontsize=8, color='#555555')


def main():
    mimic = load_race('tables/table8b_subgroup_race_boot.csv')
    eicu = load_race('tables/table9b_subgroup_external.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    panel(axes[0], mimic, 'MIMIC-IV (development test set)')
    panel(axes[1], eicu, 'eICU (external evaluation)')

    # (A)/(B) 子图字母，期刊惯例，图注引用方便
    axes[0].text(-0.14, 1.02, '(A)', transform=axes[0].transAxes,
                 fontsize=14, fontweight='bold', va='bottom', ha='left')
    axes[1].text(-0.14, 1.02, '(B)', transform=axes[1].transAxes,
                 fontsize=14, fontweight='bold', va='bottom', ha='left')

    axes[0].set_ylabel('Race subgroup')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    import os
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/fig_subgroup_forest.png', dpi=300, bbox_inches='tight')
    print('已保存: figures/fig_subgroup_forest.png')


if __name__ == '__main__':
    main()