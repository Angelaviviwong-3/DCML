#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASET_NAME = "amazon_beauty"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"

# 对接 Stage III 生成的最新 ATE 文件 (v1 版本)
ATE_PATH = os.path.join(BASE_DIR, f"Final_Causal_Output/{DATASET_NAME}_Final_ATE_v1.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 【修改点 1】：更新日志文件尾缀为 0510
log_filename = f"rq5_robustness_value_{DATASET_NAME}_0510_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def calculate_rv(t_stat, dof):
    """
    根据论文公式 (21) 计算 Robustness Value (RV)
    衡量需要多强的未观测混淆变量才能推翻现有结论。
    """
    f2 = (t_stat ** 2) / dof
    rv = 0.5 * (np.sqrt(f2**2 + 4*f2) - f2)
    return rv

def main():
    logger.info("="*80)
    logger.info(f">>> [Section 5.6.1] 敏感性分析：计算多阶段全变量 Robustness Value (RV) - {DATASET_NAME}")
    logger.info("="*80)

    if not os.path.exists(ATE_PATH):
        logger.error(f"❌ 找不到 ATE 结果文件: {ATE_PATH}")
        return
        
    df_ate = pd.read_csv(ATE_PATH)
    
    # ⚠️ 关键参数：亚马逊 Beauty Set C 的准确样本量 (自由度)
    N_DOF = 6568021  
    logger.info(f">>> 设定计算自由度 (N_DOF) = {N_DOF:,}")
    
    # 【修改点 2】：提取全阶段、全变量的 ATE 结果，不再强制过滤 P < 0.05
    all_effects = df_ate[df_ate['H_level'] == 'All_ATE'].copy()
    
    if all_effects.empty:
        logger.warning("⚠️ 未发现 H_level 为 'All_ATE' 的记录，请检查 ATE 结果文件。")
        return

    # 【修改点 3】：新增显著性判定列
    all_effects['Significant_05'] = all_effects['P_Value'].apply(lambda p: 'Yes' if p < 0.05 else 'No')

    # 计算 RV
    all_effects['RV'] = all_effects['T_Stat'].apply(lambda t: calculate_rv(t, N_DOF))
    
    # 整理输出格式，加入 Significant_05
    rv_table = all_effects[['Resolution', 'Stage', 'Component', 'Coefficient', 'T_Stat', 'P_Value', 'Significant_05', 'RV']].copy()
    
    # 【修改点 4】：将 RV 格式化为百分比形式并保留4位小数
    rv_table['RV_Formatted'] = rv_table['RV'].apply(lambda x: f"{x*100:.4f}%")
    
    # 分阶段打印日志以便预览
    stages = ['click', 'cart', 'purchase']
    for stage in stages:
        stage_df = rv_table[rv_table['Stage'] == stage]
        if not stage_df.empty:
            logger.info(f"\n🎯 [核心结论 - {stage.upper()} 阶段] 所有变量的 RV 值:")
            # 打印时，显著的优先展示在前面，其次按宏观/微观排序
            stage_df_sorted = stage_df.sort_values(by=['Significant_05', 'Resolution'], ascending=[False, False])
            logger.info("\n" + stage_df_sorted[['Resolution', 'Component', 'Coefficient', 'T_Stat', 'Significant_05', 'RV_Formatted']].to_string(index=False))
    
    # 【修改点 5】：保存结果加上尾缀 _0510
    out_csv_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_RQ5_Robustness_Value_Table_0510.csv")
    rv_table.to_csv(out_csv_path, index=False)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ RV 全变量鲁棒性分析结果已成功保存至: \n   {out_csv_path}")
    logger.info("="*80)
    
    # 学术解读指导
    logger.info("\n💡 [学术分析提示]:")
    logger.info("RV 值代表了结果对‘遗漏变量偏误’的抵抗力。")
    logger.info(f"在 {DATASET_NAME} 数据集中，对于显著变量（Significant_05=Yes），特别是社会信号 (T_con_soc)，")
    logger.info("极高的 RV 值（如 > 10%）强有力地证明了从众效应作为美妆决策核心驱动力的稳健性，无法被未观测的混淆变量推翻。")

if __name__ == "__main__":
    main()
    
    # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/robustness_value_amazon_beauty.py