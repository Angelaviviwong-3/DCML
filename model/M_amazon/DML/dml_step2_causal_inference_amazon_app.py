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
import argparse
from datetime import datetime
import statsmodels.api as sm
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASET_NAME = "amazon_appliances"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"
# 读取 Stage II 生成的残差文件
RES_PATH = os.path.join(BASE_DIR, f"DML_Results/{DATASET_NAME}_DML_Residuals_Final_2.parquet")

SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置
log_filename = f"causal_inference_amazon_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(f">>> [DCML Stage III] 亚马逊 {DATASET_NAME} 最终因果效应估计 (V3)")
    logger.info(">>> [目标]: 包含 Std_Error 字段，版本尾缀 v2")
    logger.info("="*80)

    # 1. 载入残差数据
    logger.info(f">>> 1. 载入 2100万行残差矩阵...")
    df = pd.read_parquet(RES_PATH)
    Y_cols = ['res_y_click', 'res_y_cart', 'res_y_purchase']
    
    # ---------------------------------------------------------
    # STEP 1: Multivariate Gram-Schmidt 正交化
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 执行 Multivariate Gram-Schmidt 正交化投影...")
    
    # 【1.1 宏观层面 (Macro)】
    con_macro = ['res_T_con_mkt_calibrated', 'res_T_con_soc', 'res_T_con_rat']
    int_macro = ['res_T_int_fac_calibrated', 'res_T_int_vis_calibrated', 'res_T_int_sem']
    
    macro_pure_cols = []
    for target in tqdm(int_macro, desc="Macro Orthogonalization"):
        model = sm.OLS(df[target], df[con_macro]).fit()
        pure_name = f"{target}_pure"
        df[pure_name] = df[target] - model.predict(df[con_macro])
        macro_pure_cols.append(pure_name)
        
    T_star_macro = con_macro + macro_pure_cols

    # 【1.2 原子层面 (Atomic)】
    con_atomic = [
        'res_atomic_a_pri_calibrated', 'res_atomic_a_gft_calibrated', 
        'res_atomic_a_sub_calibrated', 'res_atomic_a_urg_calibrated'
    ]
    int_atomic = ['res_atomic_a_spec_calibrated', 'res_atomic_a_str_calibrated']
    
    atomic_pure_cols = []
    for target in tqdm(int_atomic, desc="Atomic Orthogonalization"):
        model = sm.OLS(df[target], df[con_atomic]).fit()
        pure_name = f"{target}_pure"
        df[pure_name] = df[target] - model.predict(df[con_atomic])
        atomic_pure_cols.append(pure_name)

    # 事实原子 a_fact 复用提纯后的事实残差
    T_star_atomic = con_atomic + atomic_pure_cols + ['res_T_int_fac_calibrated_pure']

    # ---------------------------------------------------------
    # 核心因果回归引擎 (增加 Std_Error 提取)
    # ---------------------------------------------------------
    def run_causal_engine(treatment_cols, res_label):
        results = []
        
        # A. ATE 计算
        logger.info(f"   - 正在计算 {res_label} 组 ATE...")
        for y_col in Y_cols:
            model = sm.OLS(df[y_col], df[treatment_cols]).fit()
            for i, t_col in enumerate(treatment_cols):
                clean_name = t_col.replace('res_', '').replace('_calibrated', '').replace('_pure', '').replace('atomic_', '')
                results.append({
                    'Resolution': res_label,
                    'H_level': 'All_ATE',
                    'Stage': y_col.replace('res_y_', ''),
                    'Component': clean_name,
                    'Coefficient': model.params.iloc[i],
                    'Std_Error': model.bse.iloc[i],   # ✅ 新增字段
                    'P_Value': model.pvalues.iloc[i],
                    'T_Stat': model.tvalues.iloc[i]
                })

        # B. CATE 计算
        logger.info(f"   - 正在计算 {res_label} 组 CATE...")
        h_levels = sorted(df['H_level'].unique())
        for h in tqdm(h_levels, desc=f"{res_label} CATE"):
            df_h = df[df['H_level'] == h]
            for y_col in Y_cols:
                model = sm.OLS(df_h[y_col], df_h[treatment_cols]).fit()
                for i, t_col in enumerate(treatment_cols):
                    clean_name = t_col.replace('res_', '').replace('_calibrated', '').replace('_pure', '').replace('atomic_', '')
                    results.append({
                        'Resolution': res_label,
                        'H_level': f"Involvement_{h}",
                        'Stage': y_col.replace('res_y_', ''),
                        'Component': clean_name,
                        'Coefficient': model.params.iloc[i],
                        'Std_Error': model.bse.iloc[i], # ✅ 新增字段
                        'P_Value': model.pvalues.iloc[i],
                        'T_Stat': model.tvalues.iloc[i]
                    })
            del df_h; gc.collect()
        return results

    # ---------------------------------------------------------
    # 执行回归并保存结果
    # ---------------------------------------------------------
    macro_res = run_causal_engine(T_star_macro, "Macro")
    atomic_res = run_causal_engine(T_star_atomic, "Atomic")

    df_final = pd.DataFrame(macro_res + atomic_res)
    
    # 定义保存路径
    ate_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_Final_ATE_v3.csv")
    cate_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_Final_CATE_v3.csv")
    
    # 分别保存 ATE 和 CATE
    df_final[df_final['H_level'] == 'All_ATE'].to_csv(ate_path, index=False)
    df_final[df_final['H_level'] != 'All_ATE'].to_csv(cate_path, index=False)

    logger.info("="*80)
    logger.info(f"🎉 亚马逊 Appliances 因果估计 V3 完成！")
    logger.info(f"💾 ATE 结果 (含 Std_Error): {ate_path}")
    logger.info(f"💾 CATE 结果 (含 Std_Error): {cate_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()