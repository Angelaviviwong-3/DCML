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

warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径与环境配置
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
RES_PATH = os.path.join(BASE_DIR, "DML_Results/DCML_Residuals_3.parquet")
ATE_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Final_ATE_Results_2.csv")

SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
FIG_DIR = os.path.join(SAVE_DIR, "Figures")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"rq5_placebo_test_detailed_0511_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(">>> [Section 5.6.2] 伪造/安慰剂检验 (全漏斗 + 12变量 0511版)")
    logger.info("="*80)

    # 1. 加载数据
    df_res = pd.read_parquet(RES_PATH)
    df_ate = pd.read_csv(ATE_PATH)
    
    # 2. 定义 12 个核心变量 (根据 inspect_residuals.py 匹配列名)
    con_cols = ['res_T_con_mkt_calib', 'res_T_con_soc', 'res_T_con_rat']
    int_cols = [
        'res_T_int_fac_calib', 'res_T_int_vis_calib', 'res_T_int_sem',
        'res_atomic_a_pri_calib', 'res_atomic_a_gft_calib', 'res_atomic_a_sub_calib',
        'res_atomic_a_urg_calib', 'res_atomic_a_spec_calib', 'res_atomic_a_str_calib'
    ]
    all_treatments = con_cols + int_cols
    y_cols = ['res_y_click', 'res_y_cart', 'res_y_purchase']
    
    # 3. 随机打乱处理变量矩阵 (切断与 Y 的因果联系)
    np.random.seed(2025)
    shuffled_idx = np.random.permutation(len(df_res))
    df_placebo = df_res.copy()
    
    # 整体打乱特征矩阵，保留特征间的相关性，但打破与 Y 的关系
    df_placebo[all_treatments] = df_res[all_treatments].values[shuffled_idx, :]
    
    # 4. 执行 Gram-Schmidt 正交化 (复现 DCML 真实流程)
    logger.info(">>> 正在对打乱后的特征执行 Gram-Schmidt 正交化...")
    T_con_mat = df_placebo[con_cols].values
    
    int_cols_pure = []
    for col in int_cols:
        pure_col = col + "_pure"
        int_cols_pure.append(pure_col)
        # OLS 投影并取残差
        model_gs = sm.OLS(df_placebo[col], T_con_mat).fit()
        df_placebo[pure_col] = df_placebo[col] - model_gs.predict(T_con_mat)
        
    final_placebo_treatments = con_cols + int_cols_pure
    
    # 5. 执行阶段级 ATE 回归
    logger.info(">>> 正在计算 Placebo ATE...")
    placebo_results = []
    
    for stage, y_col in zip(['click', 'cart', 'purchase'], y_cols):
        # 无截距回归 (因已残差化)
        model = sm.OLS(df_placebo[y_col], df_placebo[final_placebo_treatments]).fit()
        
        for idx, col in enumerate(final_placebo_treatments):
            # 清洗名称以匹配原表
            clean_name = col.replace('res_', '').replace('_calib', '').replace('_pure', '').replace('atomic_', '')
            placebo_results.append({
                'Stage': stage,
                'Component': clean_name,
                'Coef_Placebo': model.params[idx],
                'PVal_Placebo': model.pvalues[idx]
            })
            
    df_placebo_res = pd.DataFrame(placebo_results)
    
    # 6. 合并真实结果与安慰剂结果
    df_ate_macro = df_ate[df_ate['Resolution'] == 'Macro']
    df_ate_atomic = df_ate[df_ate['Resolution'] == 'Atomic']
    df_true = pd.concat([df_ate_macro, df_ate_atomic])
    df_true['Component'] = df_true['Component'].str.replace('atomic_', '')
    
    final_output = pd.merge(
        df_true[['Stage', 'Component', 'Coefficient', 'Std_Error', 'P_Value']], 
        df_placebo_res, 
        on=['Stage', 'Component'],
        how='inner'
    )
    
    csv_out = os.path.join(SAVE_DIR, "RQ5_Placebo_Test_Results_Detailed_0511.csv")
    final_output.to_csv(csv_out, index=False)
    logger.info(f"💾 全漏斗安慰剂对比结果已保存至: {csv_out}")

    # ================= 绘图部分 (仅绘制 Purchase 阶段作为代表) =================
    plot_df = final_output[final_output['Stage'] == 'purchase'].copy()
    
    name_map = {
        'T_con_mkt': ('Mkt Saliency', 'Sys1'), 'T_con_soc': ('Social Signal', 'Sys1'), 'T_con_rat': ('Heuristic Rat', 'Sys1'),
        'T_int_fac': ('Fact Density', 'Sys2'), 'T_int_vis': ('Vis Utility', 'Sys2'), 'T_int_sem': ('Sem Align', 'Sys2'),
        'a_pri': ('Price Shock', 'Atomic'), 'a_gft': ('Gift Appeal', 'Atomic'), 'a_sub': ('Subsidy Auth', 'Atomic'),
        'a_urg': ('Urgency/Disc', 'Atomic'), 'a_spec': ('Spec Overlay', 'Atomic'), 'a_str': ('Internal Str', 'Atomic')
    }
    
    plot_df['Display_Name'] = plot_df['Component'].apply(lambda x: name_map[x][0])
    plot_df['Group'] = plot_df['Component'].apply(lambda x: name_map[x][1])
    plot_df = plot_df.sort_values(by=['Group', 'Component']).reset_index(drop=True)

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "grid.linestyle": "--"})
    
    y_pos = np.arange(len(plot_df))
    
    # 真实估计值
    plt.errorbar(plot_df['Coefficient'], y_pos - 0.15, 
                 xerr=1.96 * plot_df['Std_Error'], 
                 fmt='o', color='#c0392b', label='True ATE (DCML)', 
                 markersize=8, capsize=4, elinewidth=1.5)
                 
    # 安慰剂估计值
    plt.scatter(plot_df['Coef_Placebo'], y_pos + 0.15, 
                marker='s', color='#7f8c8d', label='Placebo ATE (Shuffled)', 
                s=60, alpha=0.8, edgecolors='black', zorder=5)
                 
    plt.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
    plt.yticks(y_pos, plot_df['Display_Name'].values, fontsize=12)
    plt.xlabel('Estimated Average Treatment Effect (ATE) at Purchase', fontsize=13, fontweight='bold')
    plt.title('Placebo Test: True vs. Shuffled Treatments (Suning Dataset)', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plot_out = os.path.join(FIG_DIR, "Fig_RQ5_Placebo_Test_Unified_0511.pdf")
    plt.savefig(plot_out, dpi=400, bbox_inches='tight')
    
    logger.info("="*80)
    logger.info(f"✅ 安慰剂检验绘图完成！图表: {plot_out}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
#. python /home/xzhe162/wh_workspace/casual/model/M_suning/DML/placebo_test.py