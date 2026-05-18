#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DATASETS = ['amazon_appliances', 'amazon_beauty']
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"plot_5_gram_schmidt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def gram_schmidt_residual(y, X):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef

def main():
    for dataset in tqdm(DATASETS, desc="Plotting Gram-Schmidt Heatmaps"):
        # 寻找 DML 残差文件，假设存在于 DML_Results 目录下
        res_dir = f"/home/xzhe162/wh_workspace/casual/processed_data/{dataset}/DML_Results"
        save_dir = f"/home/xzhe162/wh_workspace/casual/processed_data/{dataset}/Final_Causal_Output/Figures"
        os.makedirs(save_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(res_dir, "*Residuals*.parquet"))
        if not files:
            logger.warning(f"⚠️ 找不到 {dataset} 的 Parquet 残差文件，跳过热力图绘制...")
            continue
            
        df = pd.read_parquet(files[-1]).replace([np.inf, -np.inf], np.nan).dropna()
        
        con_cols = ['res_T_con_mkt_calibrated', 'res_T_con_soc', 'res_T_con_rat']
        int_cols = ['res_T_int_fac_calibrated', 'res_T_int_vis_calibrated', 'res_T_int_sem']

        # 校验列名是否存在
        if not all(col in df.columns for col in con_cols + int_cols):
            logger.warning(f"⚠️ {dataset} 残差矩阵列名不匹配，跳过...")
            continue

        display_names = [r"$T_{con,mkt}$", r"$T_{con,soc}$", r"$T_{con,rat}$", r"$T_{int,fac}$", r"$T_{int,vis}$", r"$T_{int,sem}$"]
        corr_before = df[con_cols + int_cols].corr()

        X_con = df[con_cols].values
        pure_cols = []
        for col in int_cols:
            df[col + "_pure"] = gram_schmidt_residual(df[col].values, X_con)
            pure_cols.append(col + "_pure")

        corr_after = df[con_cols + pure_cols].corr()
        max_corr = np.max(np.abs(corr_after.iloc[3:, :3].values))

        sns.set_style("white")
        plt.rcParams["font.family"] = "serif"
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        mask = np.triu(np.ones_like(corr_before, dtype=bool))

        sns.heatmap(corr_before, ax=axes[0], cmap=cmap, mask=mask, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", cbar=False, xticklabels=display_names, yticklabels=display_names, linewidths=0.5)
        axes[0].set_title("(a) Before Orthogonalization\nSevere Cross-System Entanglement", fontsize=14, weight="bold", pad=15)
        axes[0].add_patch(patches.Rectangle((0, 3), 3, 3, fill=False, edgecolor='red', linewidth=3, linestyle='--'))

        display_names_after = display_names[:3] + [r"$T_{int,fac}^{\perp}$", r"$T_{int,vis}^{\perp}$", r"$T_{int,sem}^{\perp}$"]
        sns.heatmap(corr_after, ax=axes[1], cmap=cmap, mask=mask, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", cbar=True, xticklabels=display_names_after, yticklabels=display_names_after, linewidths=0.5)
        axes[1].set_title(f"(b) After Orthogonalization\nPerfect Decoupling (Max |Corr| = {max_corr:.3f})", fontsize=14, weight="bold", pad=15)
        axes[1].add_patch(patches.Rectangle((0, 3), 3, 3, fill=False, edgecolor='red', linewidth=3, linestyle='-'))

        for ax in axes:
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', rotation=0, labelsize=12)

        plt.tight_layout()
        pdf_path = os.path.join(save_dir, f"{dataset}_Fig_GramSchmidt_Clean_v1.pdf")
        plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ {dataset} 图表已保存: {pdf_path}")

if __name__ == "__main__":
    main()
    
     #. python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/plot_5_gram_schmidt_heatmap_amazon.py