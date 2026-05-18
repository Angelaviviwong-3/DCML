import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径配置
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
RES_PATH = os.path.join(BASE_DIR, "DML_Results/DCML_Residuals_3.parquet")
SAVE_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# Gram-Schmidt 函数
# ==========================================
def gram_schmidt_residual(y, X):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef

def main():
    df = pd.read_parquet(RES_PATH).replace([np.inf, -np.inf], np.nan).dropna()
    con_cols = ['res_T_con_mkt_calib', 'res_T_con_soc', 'res_T_con_rat']
    int_cols = ['res_T_int_fac_calib', 'res_T_int_vis_calib', 'res_T_int_sem']

    display_names = [r"$T_{con,mkt}$", r"$T_{con,soc}$", r"$T_{con,rat}$",
                     r"$T_{int,fac}$", r"$T_{int,vis}$", r"$T_{int,sem}$"]

    # --- Before ---
    corr_before = df[con_cols + int_cols].corr()

    # --- Gram-Schmidt ---
    X_con = df[con_cols].values
    pure_cols = []
    for col in int_cols:
        df[col + "_pure"] = gram_schmidt_residual(df[col].values, X_con)
        pure_cols.append(col + "_pure")

    corr_after = df[con_cols + pure_cols].corr()
    max_corr = np.max(np.abs(corr_after.iloc[3:, :3].values))

    # ==========================================
    # 绘图（去文字，保留重点红框）
    # ==========================================
    sns.set_style("white")
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr_before, dtype=bool))

    # --- 图 (a) ---
    sns.heatmap(corr_before, ax=axes[0], cmap=cmap, mask=mask, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", cbar=False, xticklabels=display_names, yticklabels=display_names, linewidths=0.5)
    axes[0].set_title("(a) Before Orthogonalization\nSevere Cross-System Entanglement", fontsize=14, weight="bold", pad=15)
    # 红色虚线框：高亮伪相关区域
    axes[0].add_patch(patches.Rectangle((0, 3), 3, 3, fill=False, edgecolor='red', linewidth=3, linestyle='--'))

    # --- 图 (b) ---
    display_names_after = display_names[:3] + [r"$T_{int,fac}^{\perp}$", r"$T_{int,vis}^{\perp}$", r"$T_{int,sem}^{\perp}$"]
    sns.heatmap(corr_after, ax=axes[1], cmap=cmap, mask=mask, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", cbar=True, xticklabels=display_names_after, yticklabels=display_names_after, linewidths=0.5)
    axes[1].set_title(f"(b) After Orthogonalization\nPerfect Decoupling (Max |Corr| = {max_corr:.3f})", fontsize=14, weight="bold", pad=15)
    # 红色实线框：高亮正交解耦区域
    axes[1].add_patch(patches.Rectangle((0, 3), 3, 3, fill=False, edgecolor='red', linewidth=3, linestyle='-'))

    for ax in axes:
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', rotation=0, labelsize=12)

    plt.tight_layout()
    pdf_path = os.path.join(SAVE_DIR, "Fig_GramSchmidt_Clean.pdf")
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"✅ 图表已保存至: {pdf_path}")

if __name__ == "__main__":
    main()
    

    
    # python /home/xzhe162/wh_workspace/casual/model/DML/gram_schmidt_heatmap.py
    
    # step1_