#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASET_NAME = "amazon_appliances"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"

# 输入路径
RES_PATH = os.path.join(BASE_DIR, f"DML_Results/{DATASET_NAME}_DML_Residuals_Final_2.parquet")
ATE_PATH = os.path.join(BASE_DIR, f"Final_Causal_Output/{DATASET_NAME}_Final_ATE_v3.csv")

# 输出路径
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
FIG_DIR = os.path.join(SAVE_DIR, "Figures")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"rq5_placebo_test_{DATASET_NAME}_0511.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(f">>> [Section 5.6.2] 安慰剂检验 (全漏斗+12变量): {DATASET_NAME}")
    logger.info("="*80)

    # 1. 加载数据
    logger.info(">>> 1. 正在载入海量残差数据...")
    df_res = pd.read_parquet(RES_PATH)
    df_ate = pd.read_csv(ATE_PATH)
    
    # 针对海量数据抽样，以防内存崩溃
    if len(df_res) > 1000000:
        logger.info(f"   - 样本量巨大 ({len(df_res):,}), 正在抽样 1,000,000 条进行检验...")
        df_res = df_res.sample(n=1000000, random_state=42)

    # 2. 定义 12 个核心 Treatment 变量
    con_cols = ['res_T_con_mkt_calibrated', 'res_T_con_soc', 'res_T_con_rat']
    int_cols = [
        'res_T_int_fac_calibrated', 'res_T_int_vis_calibrated', 'res_T_int_sem',
        'res_atomic_a_pri_calibrated', 'res_atomic_a_gft_calibrated', 'res_atomic_a_sub_calibrated',
        'res_atomic_a_urg_calibrated', 'res_atomic_a_spec_calibrated', 'res_atomic_a_str_calibrated'
    ]
    all_treatments = con_cols + int_cols
    y_cols = ['res_y_click', 'res_y_cart', 'res_y_purchase']
    
    # 3. 安慰剂打乱 (Shuffling)
    logger.info(">>> 2. 正在随机打乱 Treatment 矩阵 (切断 Y 的因果链)...")
    np.random.seed(2025)
    shuffled_idx = np.random.permutation(len(df_res))
    df_placebo = df_res.copy()
    df_placebo[all_treatments] = df_res[all_treatments].values[shuffled_idx, :]
    
    # 4. 重新执行正交化投影 (复现 DCML 提纯逻辑)
    logger.info(">>> 3. 正在对打乱后的变量执行多元正交化提纯...")
    T_con_mat = df_placebo[con_cols].values
    
    int_cols_pure = []
    for col in int_cols:
        pure_col = col + "_pure"
        int_cols_pure.append(pure_col)
        # 将 System 2/Atomic 向 System 1 空间投射并取残差
        model_gs = sm.OLS(df_placebo[col], T_con_mat).fit()
        df_placebo[pure_col] = df_placebo[col] - model_gs.predict(T_con_mat)
            
    final_placebo_treatments = con_cols + int_cols_pure
    
    # 5. 执行全阶段安慰剂回归
    logger.info(">>> 4. 运行全漏斗阶段安慰剂回归分析...")
    placebo_results = []
    for stage, y_col in zip(['click', 'cart', 'purchase'], y_cols):
        model = sm.OLS(df_placebo[y_col], df_placebo[final_placebo_treatments]).fit()
        for i, col in enumerate(final_placebo_treatments):
            clean_name = col.replace('res_', '').replace('_calibrated', '').replace('_pure', '').replace('atomic_', '')
            placebo_results.append({
                'Stage': stage,
                'Component': clean_name,
                'Coef_Placebo': model.params[i]
            })
    df_placebo_df = pd.DataFrame(placebo_results)
    
    # 6. 合并真实 ATE 与安慰剂结果
    # 统一组件名称
    df_ate['Component'] = df_ate['Component'].str.replace('atomic_', '')
    df_true = df_ate[df_ate['Resolution'].isin(['Macro', 'Atomic'])].copy()
    
    final_output = pd.merge(
        df_true[['Stage', 'Component', 'Coefficient', 'Std_Error', 'P_Value']], 
        df_placebo_df, 
        on=['Stage', 'Component']
    )
    
    csv_out = os.path.join(SAVE_DIR, f"{DATASET_NAME}_RQ5_Placebo_Test_Results_Detailed_0511.csv")
    final_output.to_csv(csv_out, index=False)
    logger.info(f"💾 详细对比数据已保存至: {csv_out}")

    # ================= 绘图部分 (以 Purchase 阶段为代表) =================
    plot_df = final_output[final_output['Stage'] == 'purchase'].copy()
    
    # 标签映射
    name_map = {
        'T_con_mkt': 'Mkt Saliency', 'T_con_soc': 'Social Signal', 'T_con_rat': 'Heuristic Rat',
        'T_int_fac': 'Fact Density', 'T_int_vis': 'Vis Utility', 'T_int_sem': 'Sem Align',
        'a_pri': 'Price Shock', 'a_gft': 'Gift Appeal', 'a_sub': 'Subsidy Auth',
        'a_urg': 'Urgency/Disc', 'a_spec': 'Spec Overlay', 'a_str': 'Internal Str'
    }
    plot_df['Display_Name'] = plot_df['Component'].map(name_map)
    plot_df['System'] = plot_df['Component'].apply(lambda x: 'Sys2' if any(i in x for i in ['int', 'a_']) else 'Sys1')

    plt.figure(figsize=(12, 8.5))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "grid.linestyle": "--"})
    
    y_pos = np.arange(len(plot_df))
    
    # 绘制真实估计值
    plt.errorbar(plot_df['Coefficient'], y_pos - 0.15, 
                 xerr=1.96 * plot_df['Std_Error'], 
                 fmt='o', color='#c0392b', label='True ATE (DCML)', 
                 markersize=8, capsize=4, elinewidth=1.5)
                 
    # 绘制安慰剂估计点
    plt.scatter(plot_df['Coef_Placebo'], y_pos + 0.15, 
                marker='s', color='#7f8c8d', label='Placebo ATE (Shuffled)', 
                s=70, alpha=0.8, edgecolors='black', zorder=5)
                 
    # 显著性标注
    for idx, row in plot_df.iterrows():
        if row['P_Value'] < 0.01:
            plt.text(row['Coefficient'], idx - 0.3, '***', ha='center', fontweight='bold', color='#c0392b')

    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    plt.yticks(y_pos, plot_df['Display_Name'].values, fontsize=12)
    plt.xlabel('Estimated Average Treatment Effect (ATE)', fontsize=13, fontweight='bold')
    plt.title(f'Falsification Test: {DATASET_NAME.replace("_", " ").title()}\n(True vs. Shuffled Treatments)', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plot_out = os.path.join(FIG_DIR, f"Fig_RQ5_Placebo_Test_{DATASET_NAME}_0511.pdf")
    plt.savefig(plot_out, dpi=400, bbox_inches='tight')
    
    logger.info("="*80)
    logger.info(f"✅ 安慰剂检验流程结束。")
    logger.info(f"🖼️ 图像已保存至: {plot_out}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
    # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/placebo_test_amazon_app.py