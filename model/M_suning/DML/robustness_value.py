import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# ==========================================
# 0. 路径与环境配置
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
ATE_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Final_ATE_Results_2.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 更新日志尾缀为 0510
log_filename = f"rq5_robustness_value_0510_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def calculate_rv(t_stat, dof):
    """根据论文公式 (21) 计算 Robustness Value (RV)"""
    f2 = (t_stat ** 2) / dof
    rv = 0.5 * (np.sqrt(f2**2 + 4*f2) - f2)
    return rv

def main():
    logger.info("="*80)
    logger.info(">>> [Section 5.6.1] 敏感性分析：计算多阶段全变量 Robustness Value (RV)")
    logger.info("="*80)

    if not os.path.exists(ATE_PATH):
        logger.error(f"❌ 找不到 ATE 结果文件: {ATE_PATH}")
        return
        
    df_ate = pd.read_csv(ATE_PATH)
    
    # 苏宁数据集样本量约 83.6K (自由度)
    N_DOF = 83600  
    
    # 【修改点 1】：提取全阶段、全变量 (包含 3行为 x (6宏观 + 6微观))
    # 不再强制过滤 P < 0.05，而是保留所有变量，新增显著性标签
    all_effects = df_ate[df_ate['H_level'] == 'All (ATE)'].copy()
    
    # 标记是否显著 (P < 0.05)
    all_effects['Significant_05'] = all_effects['P_Value'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
    
    # 计算 RV
    all_effects['RV'] = all_effects['T_Stat'].apply(lambda t: calculate_rv(t, N_DOF))
    
    # 整理输出格式
    rv_table = all_effects[['Resolution', 'Stage', 'Component', 'Coefficient', 'T_Stat', 'P_Value', 'Significant_05', 'RV']].copy()
    rv_table['RV_Formatted'] = rv_table['RV'].apply(lambda x: f"{x*100:.4f}%") # 格式化为百分比便于阅读
    
    # 分阶段打印日志
    stages = ['click', 'cart', 'purchase']
    for stage in stages:
        stage_df = rv_table[rv_table['Stage'] == stage]
        if not stage_df.empty:
            logger.info(f"\n🎯 [核心结论 - {stage.upper()} 阶段] 所有变量的 RV 值:")
            # 在控制台打印时优先展示显著的变量在前面
            stage_df_sorted = stage_df.sort_values(by=['Significant_05', 'Resolution'], ascending=[False, False])
            logger.info("\n" + stage_df_sorted[['Resolution', 'Component', 'Coefficient', 'T_Stat', 'Significant_05', 'RV_Formatted']].to_string(index=False))
    
    # 【修改点 2】：保存结果加上尾缀 _0510
    out_csv_path = os.path.join(SAVE_DIR, "RQ5_Robustness_Value_Table_0510.csv")
    rv_table.to_csv(out_csv_path, index=False)
    logger.info("="*80)
    logger.info(f"✅ RV 全变量结果表已成功保存至: {out_csv_path}")

if __name__ == "__main__":
    main()
    
#. python /home/xzhe162/wh_workspace/casual/model/M_suning/DML/robustness_value.py