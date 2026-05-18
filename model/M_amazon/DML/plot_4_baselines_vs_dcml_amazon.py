#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATASETS = ['amazon_appliances', 'amazon_beauty']
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"plot_4_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def get_file(base_dir, pattern):
    files = glob.glob(os.path.join(base_dir, pattern))
    return sorted(files)[-1] if files else None

def main():
    for dataset in tqdm(DATASETS, desc="Plotting Baselines vs DCML"):
        base_dir = f"/home/xzhe162/wh_workspace/casual/processed_data/{dataset}/Final_Causal_Output"
        dcml_path = get_file(base_dir, f"{dataset}_Final_ATE_*.csv")
        base_path = get_file(base_dir, f"{dataset}_Baseline_Comparison_*.csv")
        save_dir = os.path.join(base_dir, "Figures")
        os.makedirs(save_dir, exist_ok=True)

        if not (dcml_path and base_path):
            logger.error(f"找不到 {dataset} 的必要对比文件, 跳过...")
            continue

        df_dcml = pd.read_csv(dcml_path)
        df_dcml['Estimator'] = 'DCML (Ours)'
        df_base = pd.read_csv(base_path)
        df_all = pd.concat([df_dcml, df_base], ignore_index=True)
        
        # 兼容处理 H_level 的命名
        df_plot = df_all[df_all['H_level'].isin(['All_ATE', 'All (ATE)', np.nan])].copy()
        
        macro_vars = ['T_con_mkt', 'T_con_soc', 'T_con_rat', 'T_int_fac', 'T_int_vis', 'T_int_sem']
        macro_map = dict(zip(macro_vars, ['Mkt Saliency', 'Social Sig', 'Heuristic Rat', 'Fact Density', 'Vis Utility', 'Sem Align']))
        atomic_vars = ['a_pri', 'a_gft', 'a_sub', 'a_urg', 'a_spec', 'a_str', 'T_int_fac']
        atomic_map = dict(zip(atomic_vars, ['Price Shock', 'Gift Appeal', 'Subsidy', 'Urgency', 'Spec Overlay', 'Struct View', 'Fact Density']))
        
        df_plot['Estimator'] = pd.Categorical(df_plot['Estimator'], categories=['Naive OLS', 'IPW', 'DCML (Ours)'], ordered=True)
        stages = ['click', 'cart', 'purchase']
        
        plt.rcParams["font.family"] = "serif"
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        palette = {'Naive OLS': '#95a5a6', 'IPW': '#f39c12', 'DCML (Ours)': '#2ecc71'}

        def plot_row(row_idx, res_level, var_list, name_map, row_title_prefix):
            for col_idx, stage in enumerate(stages):
                ax = axes[row_idx, col_idx]
                df_stage = df_plot[(df_plot['Resolution'] == res_level) & (df_plot['Stage'] == stage)].copy()
                bar_width = 0.25
                x_indices = np.arange(len(var_list))
                
                for i, est in enumerate(['Naive OLS', 'IPW', 'DCML (Ours)']):
                    est_data = df_stage[df_stage['Estimator'] == est]
                    y_vals, y_errs = [], []
                    for v in var_list:
                        match = est_data[est_data['Component'] == v]
                        if not match.empty:
                            y_vals.append(match['Coefficient'].values[0])
                            # 容错处理：baseline 文件可能没有 Std_Error，有则画误差棒
                            err = match.get('Std_Error', pd.Series([0])).values[0]
                            y_errs.append(err * 1.96 if pd.notnull(err) else 0)
                        else:
                            y_vals.append(0); y_errs.append(0)
                    
                    x_pos = x_indices + (i - 1) * bar_width
                    ax.bar(x_pos, y_vals, width=bar_width, label=est if (row_idx==0 and col_idx==0) else "", color=palette[est], edgecolor='black', zorder=3)
                    if any(y_errs): ax.errorbar(x_pos, y_vals, yerr=y_errs, fmt='none', c='black', capsize=3, zorder=4)

                ax.set_xticks(x_indices)
                ax.set_xticklabels([name_map[v] for v in var_list], fontsize=11, rotation=35, ha='right')
                ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=1)
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                ax.set_title(f"{row_title_prefix} {['Click', 'Add-to-Cart', 'Purchase'][col_idx]}", fontsize=15, weight='bold')

                valid_data = df_stage[df_stage['Estimator'].isin(['Naive OLS', 'DCML (Ours)'])]
                if not valid_data.empty:
                    err_max = valid_data.get('Std_Error', pd.Series([0])).max()
                    max_val, min_val = valid_data['Coefficient'].max(), valid_data['Coefficient'].min()
                    span = max_val - min_val
                    ax.set_ylim(min_val - span * 0.5 - err_max, max_val + span * 1.0 + err_max)

        plot_row(0, 'Macro', macro_vars, macro_map, "(a)")
        plot_row(1, 'Atomic', atomic_vars, atomic_map, "(b)")

        fig.legend(loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, -0.03))
        plt.subplots_adjust(hspace=0.4, wspace=0.15)
        pdf_path = os.path.join(save_dir, f"{dataset}_Fig_Baseline_Failure_v1.pdf")
        plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ {dataset} 图表已保存: {pdf_path}")

if __name__ == "__main__":
    main()
    
    
    #. python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/plot_4_baselines_vs_dcml_amazon.py