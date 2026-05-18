import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
DCML_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Final_ATE_Results_2.csv")
BASE_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Baseline_Comparison_Results.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output/Figures")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print(">>> 正在生成 [终极全量版] 基线对比图 (12 Variables x 3 Stages x 3 Estimators) ...")

    # 1. 数据准备
    df_dcml = pd.read_csv(DCML_PATH)
    df_dcml['Estimator'] = 'DCML (Ours)'
    df_base = pd.read_csv(BASE_PATH)
    df_all = pd.concat([df_dcml, df_base], ignore_index=True)
    
    # 过滤出 ATE 结果
    df_plot = df_all[df_all['H_level'] == 'All (ATE)'].copy()
    
    # 定义完整特征集与展示名称 (确保与数据中的 Component 匹配)
    macro_vars = ['T_con_mkt', 'T_con_soc', 'T_con_rat', 'T_int_fac', 'T_int_vis', 'T_int_sem']
    macro_names = ['Mkt Saliency', 'Social Sig', 'Heuristic Rat', 'Fact Density', 'Vis Utility', 'Sem Align']
    macro_map = dict(zip(macro_vars, macro_names))
    
    atomic_vars = ['a_pri', 'a_gft', 'a_sub', 'a_urg', 'a_spec', 'a_str', 'T_int_fac']
    atomic_names = ['Price Shock', 'Gift Appeal', 'Subsidy', 'Urgency', 'Spec Overlay', 'Struct View', 'Fact Density']
    atomic_map = dict(zip(atomic_vars, atomic_names))
    
    # 确保排序逻辑
    df_plot['Estimator'] = pd.Categorical(df_plot['Estimator'], categories=['Naive OLS', 'IPW', 'DCML (Ours)'], ordered=True)
    
    stages = ['click', 'cart', 'purchase']
    stage_titles = ['Top-Funnel: Click', 'Mid-Funnel: Add-to-Cart', 'Bottom-Funnel: Purchase']

    # 2. 绘图设置 (2 行 x 3 列巨幅图)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    palette = {'Naive OLS': '#95a5a6', 'IPW': '#f39c12', 'DCML (Ours)': '#2ecc71'}
    estimators = ['Naive OLS', 'IPW', 'DCML (Ours)']

    def plot_row(row_idx, res_level, var_list, name_map, row_title_prefix):
        for col_idx, stage in enumerate(stages):
            ax = axes[row_idx, col_idx]
            df_stage = df_plot[(df_plot['Resolution'] == res_level) & (df_plot['Stage'] == stage)].copy()
            
            # 手动计算柱子位置
            bar_width = 0.25
            x_indices = np.arange(len(var_list))
            display_names = [name_map[v] for v in var_list]
            
            for i, est in enumerate(estimators):
                est_data = df_stage[df_stage['Estimator'] == est]
                # 按照 var_list 的顺序提取
                y_vals = []
                y_errs = []
                for v in var_list:
                    match = est_data[est_data['Component'] == v]
                    if not match.empty:
                        y_vals.append(match['Coefficient'].values[0])
                        y_errs.append(match['Std_Error'].values[0] * 1.96)
                    else:
                        y_vals.append(0)
                        y_errs.append(0)
                
                x_pos = x_indices + (i - 1) * bar_width
                
                # 画柱状图
                ax.bar(x_pos, y_vals, width=bar_width, label=est if (row_idx==0 and col_idx==0) else "", 
                       color=palette[est], edgecolor='black', zorder=3)
                
                # 画误差棒 (Error Bars)
                ax.errorbar(x_pos, y_vals, yerr=y_errs, fmt='none', c='black', capsize=3, linewidth=1.2, zorder=4)

            # 样式美化
            ax.set_xticks(x_indices)
            ax.set_xticklabels(display_names, fontsize=11, rotation=35, ha='right')
            ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=1)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
            
            # 标题与标签
            ax.set_title(f"{row_title_prefix} {stage_titles[col_idx]}", fontsize=15, weight='bold', pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"{res_level}-Level\nMarginal Effect (ATE)", fontsize=14, weight='bold')

            # ⚠️ 极为关键的 Y 轴动态截断逻辑 (限制 IPW 爆炸导致的视觉失真)
            # 以 OLS 和 DCML 的范围为基准，给上下留出空间
            valid_data = df_stage[df_stage['Estimator'].isin(['Naive OLS', 'DCML (Ours)'])]
            if not valid_data.empty:
                max_val = valid_data['Coefficient'].max() + valid_data['Std_Error'].max() * 2
                min_val = valid_data['Coefficient'].min() - valid_data['Std_Error'].max() * 2
                span = max_val - min_val
                ax.set_ylim(min_val - span * 0.5, max_val + span * 1.0)

    # 绘制第一行 (Macro)
    plot_row(0, 'Macro', macro_vars, macro_map, "(a)")
    # 绘制第二行 (Atomic)
    plot_row(1, 'Atomic', atomic_vars, atomic_map, "(b)")

    # 底部居中图例
    fig.legend(loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, -0.03), frameon=True, shadow=True)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.15)
    pdf_path = os.path.join(SAVE_DIR, "Fig_Baseline_Failure_Ultimate_12Vars.pdf")
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"✅ [终极全景图] 12特征基线对比已保存至: {pdf_path}")

if __name__ == "__main__":
    main()
    
    
     #. python /home/xzhe162/wh_workspace/casual/model/DML/plot_baselines_vs_dcml.py
    