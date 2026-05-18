#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
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
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"plot_1_dual_track_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

def get_file(base_dir, pattern):
    files = glob.glob(os.path.join(base_dir, pattern))
    return sorted(files)[-1] if files else None

def main():
    for dataset in tqdm(DATASETS, desc="Plotting Dual Track Reversal"):
        base_dir = f"/home/xzhe162/wh_workspace/casual/processed_data/{dataset}/Final_Causal_Output"
        ate_path = get_file(base_dir, f"{dataset}_Final_ATE_*.csv")
        save_dir = os.path.join(base_dir, "Figures")
        os.makedirs(save_dir, exist_ok=True)

        if not ate_path:
            logger.error(f"找不到 {dataset} 的 ATE 文件, 跳过...")
            continue

        df = pd.read_csv(ate_path)
        stage_map = {'click': 0, 'cart': 1, 'purchase': 2}
        df['Stage_Order'] = df['Stage'].map(stage_map)
        
        sys1_macro = {'T_con_mkt': ('#e74c3c', 'Mkt Saliency'), 'T_con_soc': ('#e67e22', 'Social Signal'), 'T_con_rat': ('#d35400', 'Heuristic Rat')}
        sys1_atomic = {'a_pri': ('#e74c3c', 'Price Shock'), 'a_gft': ('#e67e22', 'Gift Appeal'), 'a_sub': ('#f39c12', 'Subsidy'), 'a_urg': ('#d35400', 'Urgency')}
        sys2_macro = {'T_int_fac': ('#2980b9', 'Fact Density'), 'T_int_vis': ('#27ae60', 'Vis Utility'), 'T_int_sem': ('#8e44ad', 'Sem Align')}
        sys2_atomic = {'a_spec': ('#2980b9', 'Spec Overlay'), 'a_str': ('#27ae60', 'Structure View'), 'T_int_fac': ('#8e44ad', 'Fact Density')}

        plt.rcParams["font.family"] = "serif"
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        x_pos, x_labels = [0, 1, 2], ['Click', 'Cart', 'Purchase']

        def plot_track(ax, res_level, sys1_dict, sys2_dict, title):
            subset = df[df['Resolution'] == res_level]
            for comp, (color, label) in sys1_dict.items():
                dat = subset[subset['Component'] == comp].sort_values('Stage_Order')
                if len(dat) == 3: ax.plot(x_pos, dat['Coefficient'], marker='o', color=color, linewidth=2.5, markersize=8, label=f"Sys 1: {label}")
            for comp, (color, label) in sys2_dict.items():
                dat = subset[subset['Component'] == comp].sort_values('Stage_Order')
                if len(dat) == 3: ax.plot(x_pos, dat['Coefficient'], marker='s', color=color, linewidth=2.5, markersize=8, linestyle='--', label=f"Sys 2: {label}")

            ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=13)
            ax.set_title(title, fontsize=15, weight='bold', pad=15)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax.set_ylabel("Marginal Causal Effect (ATE)", fontsize=13, weight='bold')
            ax.legend(fontsize=10, loc='best', framealpha=0.8)

        plot_track(axes[0], 'Macro', sys1_macro, sys2_macro, "(a) Macro-Level Multimodal Features")
        plot_track(axes[1], 'Atomic', sys1_atomic, sys2_atomic, "(b) Atomic-Level Information Cues")

        plt.tight_layout()
        pdf_path = os.path.join(save_dir, f"{dataset}_Fig_RQ2_Dual_Track_Reversal_v1.pdf")
        plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ {dataset} 图表已保存: {pdf_path}")

if __name__ == "__main__":
    main()
    
   # python  /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/plot_1_dual_track_reversal_amazon.py