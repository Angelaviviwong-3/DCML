#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================
# 修复并行溢出
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import statsmodels.api as sm
import lightgbm as lgb
import scipy.stats as stats
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASET_NAME = "amazon_beauty"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"
# ⚠️ 注意这里指向 Beauty 数据集构建产生的 _1 结尾文件
DATA_PATH = os.path.join(BASE_DIR, f"build_dataset/{DATASET_NAME}_DCML_Set_C_1.parquet")

SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"baseline_comparison_{DATASET_NAME}_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(f">>> [Section 5.3.2] 运行传统基线估计器对比 (Naive OLS & IPW) - {DATASET_NAME} (V1)")
    logger.info("="*80)

    # 1. 载入推理集数据
    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ 找不到数据文件: {DATA_PATH}")
        return
    logger.info(f">>> 正在加载 {DATASET_NAME} 数据集 (样本量: 656万)...")
    df = pd.read_parquet(DATA_PATH)
    
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    
    # 定义因果处理变量 T (12个)
    T_macro = [
        'T_int_fac_calibrated', 'T_int_vis_calibrated', 'T_con_mkt_calibrated', 
        'T_int_sem', 'T_con_soc', 'T_con_rat'
    ]
    T_atomic = [
        'atomic_a_pri_calibrated', 'atomic_a_gft_calibrated', 
        'atomic_a_sub_calibrated', 'atomic_a_urg_calibrated', 
        'atomic_a_spec_calibrated', 'atomic_a_str_calibrated',
        'T_int_fac_calibrated' # 事实密度原子
    ]

    # 定义混淆变量 X
    X_cols = ['x_pri_log', 'x_pos_freq', 'H_level', 'user_review_count_scaled', 'user_avg_rating_scaled']
    
    # 清洗变量名用于输出
    def clean_name(t):
        return t.replace('_calibrated', '').replace('atomic_', '')
    
    results = []

    # ---------------------------------------------------------
    # Baseline 1: Naive OLS (忽略所有控制变量)
    # ---------------------------------------------------------
    logger.info("\n>>> 1. 正在运行 Naive OLS (模拟原始相关性)...")
    
    for level_name, t_list in [('Macro', T_macro), ('Atomic', T_atomic)]:
        X_naive = sm.add_constant(df[t_list])
        for y_col in Y_cols:
            stage = y_col.replace('y_', '')
            logger.info(f"   [Naive OLS] 正在计算 {level_name} 组对 {stage} 的关联...")
            model_naive = sm.OLS(df[y_col], X_naive).fit()
            
            for t_col in t_list:
                results.append({
                    'Estimator': 'Naive OLS',
                    'Resolution': level_name,
                    'Stage': stage,
                    'Component': clean_name(t_col),
                    'Coefficient': model_naive.params[t_col],
                    'Std_Error': model_naive.bse[t_col],
                    'P_Value': model_naive.pvalues[t_col],
                    'T_Stat': model_naive.tvalues[t_col]
                })

    # ---------------------------------------------------------
    # Baseline 2: Statistical-Only IPW (基于倾向得分加权的估计)
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 正在运行 IPW (验证广义倾向得分 GPS)...")
    
    # 提取唯一的处理变量
    all_unique_t = list(set(T_macro + T_atomic))
    
    for t_col in tqdm(all_unique_t, desc=f"IPW Progress for {DATASET_NAME}"):
        # 1. 拟合倾向得分模型 E[T|X]
        model_t = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        model_t.fit(df[X_cols], df[t_col])
        t_hat = model_t.predict(df[X_cols])
        
        # 2. 计算连续型倾向得分权重 (Generalized Propensity Score)
        # 权重公式: W = f(T) / f(T|X)
        std_resid = np.std(df[t_col] - t_hat)
        numerator = stats.norm.pdf(df[t_col], np.mean(df[t_col]), np.std(df[t_col]))
        denominator = stats.norm.pdf(df[t_col], t_hat, std_resid)
        
        # 归一化权重并截断 99 分位数防止方差爆炸
        weights = numerator / (denominator + 1e-8)
        weights = np.clip(weights, 0, np.percentile(weights, 99))
        
        # 3. 运行 WLS 回归
        belong_to = []
        if t_col in T_macro: belong_to.append('Macro')
        if t_col in T_atomic: belong_to.append('Atomic')
            
        for level_name in belong_to:
            for y_col in Y_cols:
                stage = y_col.replace('y_', '')
                X_wls = sm.add_constant(df[t_col])
                model_ipw = sm.WLS(df[y_col], X_wls, weights=weights).fit()
                
                results.append({
                    'Estimator': 'IPW',
                    'Resolution': level_name,
                    'Stage': stage,
                    'Component': clean_name(t_col),
                    'Coefficient': model_ipw.params[t_col],
                    'Std_Error': model_ipw.bse[t_col],
                    'P_Value': model_ipw.pvalues[t_col],
                    'T_Stat': model_ipw.tvalues[t_col]
                })

    # ---------------------------------------------------------
    # 结果导出
    # ---------------------------------------------------------
    df_final = pd.DataFrame(results)
    out_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_Baseline_Comparison_v1.csv")
    df_final.to_csv(out_path, index=False)

    logger.info("\n" + "="*80)
    logger.info(f"🎉 {DATASET_NAME} 基线对比实验圆满完成！")
    logger.info(f"💾 结果保存路径: {out_path}")
    logger.info("========================================================\n")

if __name__ == "__main__":
    main()
    
  # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/causal_baseline_comparison_amazon_beauty.py