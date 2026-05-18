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
DATASET_NAME = "amazon_appliances"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"

# 对接 Stage III 生成的最新 ATE 文件 (v3 版本)
ATE_PATH = os.path.join(BASE_DIR, f"Final_Causal_Output/{DATASET_NAME}_Final_ATE_v3.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 更新日志文件尾缀为 0510
log_filename = f"rq5_robustness_value_{DATASET_NAME}_0510_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def calculate_rv(t_stat, dof):
    """
    根据论文公式 (21) 计算 Robustness Value (RV)
    RV 衡量的是：一个未被观测到的混淆变量，需要多强的解释力（方差占比），
    才能使当前显著的因果效应变为不显著 (即推翻我们的结论)。
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
    
    # 亚马逊 Appliances Set C 的准确样本量 (自由度)
    N_DOF = 21152537  
    logger.info(f">>> 设定计算自由度 (N_DOF) = {N_DOF:,}")
    
    # 【修改点 1】：提取全阶段、全变量的 ATE 结果，不再过滤 P_Value
    all_effects = df_ate[df_ate['H_level'] == 'All_ATE'].copy()
    
    # 【修改点 2】：新增显著性判定列
    all_effects['Significant_05'] = all_effects['P_Value'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
    
    # 计算 RV
    all_effects['RV'] = all_effects['T_Stat'].apply(lambda t: calculate_rv(t, N_DOF))
    
    # 整理输出列
    rv_table = all_effects[['Resolution', 'Stage', 'Component', 'Coefficient', 'T_Stat', 'P_Value', 'Significant_05', 'RV']].copy()
    
    # 【修改点 3】：格式化 RV 为百分比形式，保留4位小数
    rv_table['RV_Formatted'] = rv_table['RV'].apply(lambda x: f"{x*100:.4f}%")
    
    # 分阶段打印日志
    stages = ['click', 'cart', 'purchase']
    for stage in stages:
        stage_df = rv_table[rv_table['Stage'] == stage]
        if not stage_df.empty:
            logger.info(f"\n🎯 [核心结论 - {stage.upper()} 阶段] 所有变量的 RV 值:")
            # 打印时，显著的优先展示在前面，其次按宏观/微观排序
            stage_df_sorted = stage_df.sort_values(by=['Significant_05', 'Resolution'], ascending=[False, False])
            logger.info("\n" + stage_df_sorted[['Resolution', 'Component', 'Coefficient', 'T_Stat', 'Significant_05', 'RV_Formatted']].to_string(index=False))
    
    # 【修改点 4】：保存结果文件加上尾缀 _0510
    out_csv_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_RQ5_Robustness_Value_Table_0510.csv")
    rv_table.to_csv(out_csv_path, index=False)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ RV 全变量鲁棒性分析结果已成功保存至: \n   {out_csv_path}")
    logger.info("="*80)
    
    # 学术解读指导
    logger.info("\n💡 [学术解读指南]:")
    logger.info("如果 RV 值较大（例如 > 0.05 ~ 0.15 即 5% ~ 15%），意味着即使存在一个极强的未观测混淆变量，")
    logger.info("它也必须解释超过对应比例的剩余方差，才能推翻你发现的因果关系！")
    logger.info("对于不显著的变量(Significant_05=No)，其 RV 值将非常接近于 0，这在理论上是符合预期的边界条件体现。")

if __name__ == "__main__":
    main()
    
  # python  /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/robustness_value_amazon_app.py