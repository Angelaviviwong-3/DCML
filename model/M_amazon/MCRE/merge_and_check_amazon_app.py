#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

# ==========================================
# 0. 参数解析与路径配置
# ==========================================
parser = argparse.ArgumentParser(description="Amazon Multimodal Split Merge & ECDF Plotting")
parser.add_argument("-d", "--dataset", default="amazon_appliances", choices=["amazon_appliances", "amazon_beauty"], help="选择数据集")
args = parser.parse_args()

DATASET_NAME = args.dataset

BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}/"
SAVE_DIR = os.path.join(BASE_DIR, "item_multimodal_scalars")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ⚠️ 核心修改 1：读取正式分片文件，而非 checkpoint
INPUT_PATTERN = os.path.join(SAVE_DIR, f"{DATASET_NAME}_item_multimodal_scalars_full_2_*.csv")

# ⚠️ 核心修改 2：给所有输出文件加上 _2 尾缀
# 最终合并后输出的全量纯净版
MERGED_OUTPUT_FILE = os.path.join(SAVE_DIR, f"{DATASET_NAME}_item_multimodal_scalars_merged_full_2.parquet")

# 绘图输出路径
RAW_MACRO_PLOT = os.path.join(SAVE_DIR, f"{DATASET_NAME}_dist_01_macro_raw_full_2.png")
CALIB_MACRO_PLOT = os.path.join(SAVE_DIR, f"{DATASET_NAME}_dist_02_macro_calib_full_2.png")
CALIB_MKT_ATOMIC_PLOT = os.path.join(SAVE_DIR, f"{DATASET_NAME}_dist_03_atomic_mkt_calib_full_2.png")
CALIB_VIS_ATOMIC_PLOT = os.path.join(SAVE_DIR, f"{DATASET_NAME}_dist_04_atomic_vis_calib_full_2.png")

# 日志配置 (名称也加上 _2)
log_filename = f"merge_ecdf_{DATASET_NAME}_full_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 完美平滑零值保留秩归一化
# ==========================================
def zero_preserved_rank_norm(x, seed=2025):
    x_arr = np.array(x, dtype=float)
    x_calibrated = np.zeros_like(x_arr)
    
    mask_pos = (x_arr > 0)
    num_pos = mask_pos.sum()
    
    if num_pos > 0:
        x_pos = x_arr[mask_pos]
        np.random.seed(seed)
        jitter = np.random.uniform(0, 1e-6, size=num_pos)
        x_pos_jittered = x_pos + jitter
        
        ranks = rankdata(x_pos_jittered, method='ordinal')
        x_calibrated[mask_pos] = ranks / float(num_pos)
        
    return x_calibrated

# ==========================================
# 2. 核心绘图与统计函数
# ==========================================
def generate_ecdf_comparison(df, columns, titles, colors, save_path, xlabel_text):
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "black", 
        "grid.linestyle": "--", 
        "grid.alpha": 0.5,
        "font.family": "serif"
    })
    
    n_cols = len(columns)
    plt.figure(figsize=(6 * n_cols, 5))
    
    for i, (col, title, color) in enumerate(zip(columns, titles, colors), 1):
        plt.subplot(1, n_cols, i)
        if col in df.columns:
            sns.ecdfplot(data=df, x=col, color=color, linewidth=2.5)
            plt.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
            plt.title(title, fontsize=14, pad=15, fontweight='bold')
            plt.xlabel(xlabel_text, fontsize=11)
            if i == 1:
                plt.ylabel("Empirical CDF (Cumulative Prob.)", fontsize=11)
            else:
                plt.ylabel("")
            plt.xlim(-0.05, 1.05)
            plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def print_data_diagnostics(df, columns, phase_name):
    logger.info(f"\n--- {phase_name} 变量分布诊断 ---")
    for col in columns:
        if col in df.columns:
            zero_pct = (df[col] == 0).mean() * 100
            mean_val = df[col].mean()
            max_val = df[col].max()
            logger.info(f"[{col:<25}]: 绝对0值占比(对照组) = {zero_pct:5.2f}% | 均值 = {mean_val:.4f} | 最大值 = {max_val:.4f}")

# ==========================================
# 3. 主流程：拼接 -> 校准 -> 绘图
# ==========================================
def main():
    logger.info(f"========== 启动亚马逊多模态合并与校验 ({DATASET_NAME}) ==========")
    
    # --- A. 合并分片文件 ---
    data_files = glob.glob(INPUT_PATTERN)
    if not data_files:
        logger.error(f"❌ 找不到任何数据文件，请检查路径: {INPUT_PATTERN}")
        return

    logger.info(f">>> 找到 {len(data_files)} 个分片文件，准备合并...")
    df_list = []
    for f in sorted(data_files):
        logger.info(f"   - 正在加载: {os.path.basename(f)}")
        df_list.append(pd.read_csv(f))
    
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged['item_id'] = df_merged['item_id'].astype(str)
    
    # 按照 item_id 去重（防止分片边缘或多次保存导致重叠）
    initial_len = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['item_id'])
    logger.info(f">>> 去重后全量数据量: {len(df_merged)} 条 (剔除了 {initial_len - len(df_merged)} 条重复数据)")

    # --- B. 重新执行全局连续性校准 (极度关键) ---
    logger.info(">>> 正在执行全局分布校准 (Zero-Preserved Rank Norm)...")
    cols_to_calibrate = [
        'T_con_mkt', 'T_int_vis', 'T_int_fac',
        'atomic_a_pri', 'atomic_a_gft', 'atomic_a_sub', 'atomic_a_urg', 
        'atomic_a_spec', 'atomic_a_str'
    ]
    
    for col in cols_to_calibrate:
        if col in df_merged.columns:
            df_merged[f'{col}_calibrated'] = zero_preserved_rank_norm(df_merged[col])

    # 导出合并且校准后的最终全量数据
    df_merged.to_parquet(MERGED_OUTPUT_FILE, index=False)
    csv_merged_file = MERGED_OUTPUT_FILE.replace('.parquet', '.csv')
    df_merged.to_csv(csv_merged_file, index=False)
    logger.info(f"💾 合并且校准后的全量文件已安全保存至:\n   {MERGED_OUTPUT_FILE}\n   {csv_merged_file}")

    # --- C. 数据诊断与输出 ---
    cols_raw = ['T_int_fac', 'T_int_vis', 'T_con_mkt']
    cols_calib = ['T_int_fac_calibrated', 'T_int_vis_calibrated', 'T_con_mkt_calibrated']
    
    print_data_diagnostics(df_merged, cols_raw, "Raw (原始LMM打分)")
    print_data_diagnostics(df_merged, cols_calib, "Calibrated (校准后打分)")

    # --- D. 绘图阶段 ---
    logger.info("\n>>> 1/4 绘制宏观变量原始 ECDF...")
    titles_raw = ['Raw: Factual Text Density', 'Raw: Functional Visual', 'Raw: Marketing Saliency']
    colors_raw = ['dimgray', 'dimgray', 'dimgray']
    generate_ecdf_comparison(df_merged, cols_raw, titles_raw, colors_raw, RAW_MACRO_PLOT, "Raw LMM Intensity Score")

    logger.info(">>> 2/4 绘制宏观变量校准后 ECDF...")
    titles_calib = ['Calibrated: Factual Density', 'Calibrated: Visual Specs', 'Calibrated: Marketing Saliency']
    colors_calib = ['#1f77b4', '#2ca02c', '#d62728']
    generate_ecdf_comparison(df_merged, cols_calib, titles_calib, colors_calib, CALIB_MACRO_PLOT, "Calibrated Intensity (Rank Norm)")

    logger.info(">>> 3/4 绘制 [营销] 原子变量校准后 ECDF (4 个组件)...")
    cols_mkt_atomic = [
        'atomic_a_pri_calibrated', 'atomic_a_gft_calibrated', 
        'atomic_a_sub_calibrated', 'atomic_a_urg_calibrated'
    ]
    titles_mkt_atomic = [
        'Calib: Price Shock (a_pri)', 'Calib: Gift Appeal (a_gft)', 
        'Calib: Subsidy Auth (a_sub)', 'Calib: Urgency (a_urg)'
    ]
    colors_mkt_atomic = ['#e377c2', '#ff7f0e', '#9467bd', '#8c564b']
    generate_ecdf_comparison(df_merged, cols_mkt_atomic, titles_mkt_atomic, colors_mkt_atomic, CALIB_MKT_ATOMIC_PLOT, "Calibrated Atomic Score")

    logger.info(">>> 4/4 绘制 [视觉] 原子变量校准后 ECDF (2 个组件)...")
    cols_vis_atomic = ['atomic_a_spec_calibrated', 'atomic_a_str_calibrated']
    titles_vis_atomic = ['Calib: Spec Overlay (a_spec)', 'Calib: Internal Structure (a_str)']
    colors_vis_atomic = ['#17becf', '#bcbd22']
    generate_ecdf_comparison(df_merged, cols_vis_atomic, titles_vis_atomic, colors_vis_atomic, CALIB_VIS_ATOMIC_PLOT, "Calibrated Atomic Score")

    logger.info(f"🎉 绘图与合并任务全部圆满完成！")
    logger.info("========================================================================\n")

if __name__ == "__main__":
    main()
    
       #. python  /home/xzhe162/wh_workspace/casual/model/M_amazon/MCRE/merge_and_check_amazon_app.py