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
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"plot_3_mediation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def get_file(base_dir, pattern):
    files = glob.glob(os.path.join(base_dir, pattern))
    return sorted(files)[-1] if files else None

def main():
    for dataset in tqdm(DATASETS, desc="Plotting Mediation Forest"):
        base_dir = f"/home/xzhe162/wh_workspace/casual/processed_data/{dataset}/Final_Causal_Output"
        med_path = get_file(base_dir, f"{dataset}_Mediation_Results_*.csv")
        save_dir = os.path.join(base_dir, "Figures")
        os.makedirs(save_dir, exist_ok=True)

        if not med_path:
            logger.error(f"找不到 {dataset} 的 Mediation 文件, 跳过...")
            continue

        df = pd.read_csv(med_path)
        name_map = {'T_con_mkt': 'Mkt Saliency', 'T_con_soc': 'Social Signal', 'T_con_rat': 'Heuristic Rat'}
        df['Cue_Name'] = df['Heuristic_Cue'].map(name_map)
        df['Stage_Label'] = df['Stage'].str.capitalize() # 这里你的列名可能是 Stage 而不是 Funnel_Stage
        df['Y_Label'] = df['Cue_Name'] + " \u2192 " + df['Stage_Label']
        
        stage_map = {'click': 2, 'cart': 1, 'purchase': 0}
        df['Stage_Order'] = df['Stage'].map(stage_map)
        df = df.sort_values(['Heuristic_Cue', 'Stage_Order'])

        plt.rcParams["font.family"] = "serif"
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(df))
        
        xerr_lower = df['Indirect_Effect'] - df['95%_CI_Lower']
        xerr_upper = df['95%_CI_Upper'] - df['Indirect_Effect']
        
        ax.errorbar(x=df['Indirect_Effect'], y=y_pos, xerr=[xerr_lower, xerr_upper], 
                    fmt='o', color='#c0392b', ecolor='#34495e', elinewidth=2, capsize=5, markersize=8)

        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Y_Label'], fontsize=12)
        ax.set_xlabel("Estimated Indirect Effect via Subjective Norm ($M_{norm}$)", fontsize=13, weight='bold')
        ax.set_title("Mediation Analysis (Bootstrap 95% CI)", fontsize=15, weight='bold', pad=15)
        
        for idx, row in df.iterrows():
            sig = "***" if row['P_Value'] < 0.01 else ("*" if row['Is_Significant'] == 'Yes' else "ns")
            ax.text(row['95%_CI_Upper'] + 0.0005, y_pos[list(df.index).index(idx)], sig, 
                    va='center', color='red' if sig != 'ns' else 'grey', weight='bold')

        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        pdf_path = os.path.join(save_dir, f"{dataset}_Fig_RQ3_Mediation_Forest_v1.pdf")
        plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ {dataset} 图表已保存: {pdf_path}")

if __name__ == "__main__":
    main()
    
      #. python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/plot_3_mediation_forest_amazon.py