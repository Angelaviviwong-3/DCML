#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from econml.dml import LinearDML, CausalForestDML
warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASET_NAME = "amazon_beauty"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"
DATA_PATH = os.path.join(BASE_DIR, f"build_dataset/{DATASET_NAME}_DCML_Set_C_1.parquet")

SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"baseline_comparison_{DATASET_NAME}_v1_0509_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(f">>> [Section 5.3.2] 运行全量基线估计器对比 (OLS, IPW, DR, CF, DML) - {DATASET_NAME}")
    logger.info("="*80)

    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ 找不到数据文件: {DATA_PATH}")
        return
    logger.info(f">>> 正在加载数据集...")
    df = pd.read_parquet(DATA_PATH)
    
    # 🌟 核心优化：全局随机下采样，加速计算并防止 OOM
    GLOBAL_SAMPLE_SIZE = 200000
    if len(df) > GLOBAL_SAMPLE_SIZE:
        logger.info(f"   ⚠️ 数据量过大 ({len(df)} 行)，全局随机下采样至 {GLOBAL_SAMPLE_SIZE} 行...")
        df = df.sample(n=GLOBAL_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        logger.info(f"   - 数据量: {len(df)} 行")
    
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    T_macro = ['T_int_fac_calibrated', 'T_int_vis_calibrated', 'T_con_mkt_calibrated', 'T_int_sem', 'T_con_soc', 'T_con_rat']
    T_atomic = ['atomic_a_pri_calibrated', 'atomic_a_gft_calibrated', 'atomic_a_sub_calibrated', 'atomic_a_urg_calibrated', 'atomic_a_spec_calibrated', 'atomic_a_str_calibrated', 'T_int_fac_calibrated']
    X_cols = ['x_pri_log', 'x_pos_freq', 'H_level', 'user_review_count_scaled', 'user_avg_rating_scaled']
    
    def clean_name(t):
        return t.replace('_calibrated', '').replace('atomic_', '')
    
    results = []

    # ==========================================
    # Pipeline 1/3: Naive OLS (全量)
    # ==========================================
    logger.info("\n>>> 1. 正在运行 Naive OLS (Pipeline 1/3)...")
    for level_name, t_list in [('Macro', T_macro), ('Atomic', T_atomic)]:
        X_naive = sm.add_constant(df[t_list])
        for y_col in Y_cols:
            stage = y_col.replace('y_', '')
            logger.info(f"   [Naive OLS] 计算 {level_name} 组对 {stage} 的影响...")
            model_naive = sm.OLS(df[y_col], X_naive).fit()
            for t_col in t_list:
                results.append({'Estimator': 'Naive OLS', 'Resolution': level_name, 'Stage': stage, 'Component': clean_name(t_col),
                                'Coefficient': model_naive.params[t_col], 'P_Value': model_naive.pvalues[t_col], 'T_Stat': model_naive.tvalues[t_col]})

    # ==========================================
    # Pipeline 2: Statistical-Only IPW, DR, CF, DML
    # ==========================================
    logger.info("\n>>> 2. 正在运行高级因果推断模型 (Pipeline 2)...")
    all_unique_t = list(set(T_macro + T_atomic))
    X_vals = df[X_cols].values
    
    for t_col in tqdm(all_unique_t, desc=f"Estimating Models for {DATASET_NAME}"):
        belong_to = []
        if t_col in T_macro: belong_to.append('Macro')
        if t_col in T_atomic: belong_to.append('Atomic')

        T_vals = df[t_col].values

        # --- A. IPW ---
        model_t = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        model_t.fit(df[X_cols], df[t_col])
        t_hat = model_t.predict(df[X_cols])
        
        std_resid = np.std(df[t_col] - t_hat)
        num = stats.norm.pdf(df[t_col], np.mean(df[t_col]), np.std(df[t_col]))
        den = stats.norm.pdf(df[t_col], t_hat, std_resid)
        weights = np.clip(num / (den + 1e-8), 0, np.percentile(num / (den + 1e-8), 99))
        
        for level_name in belong_to:
            for y_col in Y_cols:
                stage = y_col.replace('y_', '')
                Y_vals = df[y_col].values

                # A. IPW 写入
                model_ipw = sm.WLS(df[y_col], sm.add_constant(df[t_col]), weights=weights).fit()
                results.append({'Estimator': 'IPW', 'Resolution': level_name, 'Stage': stage, 'Component': clean_name(t_col),
                                'Coefficient': model_ipw.params[t_col], 'P_Value': model_ipw.pvalues[t_col], 'T_Stat': model_ipw.tvalues[t_col]})

                # B. Standard DML
                try:
                    dml_est = LinearDML(model_y=lgb.LGBMRegressor(n_estimators=30, n_jobs=-1, verbose=-1), 
                                        model_t=lgb.LGBMRegressor(n_estimators=30, n_jobs=-1, verbose=-1), discrete_treatment=False, cv=2)
                    dml_est.fit(Y_vals, T_vals, X=X_vals)
                    results.append({'Estimator': 'Standard DML', 'Resolution': level_name, 'Stage': stage, 'Component': clean_name(t_col),
                                    'Coefficient': dml_est.ate(X_vals), 'P_Value': dml_est.ate__inference().pvalue(), 'T_Stat': np.nan})
                except Exception:
                    pass

                # C. Causal Forest
                try:
                    cf_est = CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=30, n_jobs=-1, verbose=-1), 
                                             model_t=lgb.LGBMRegressor(n_estimators=30, n_jobs=-1, verbose=-1), discrete_treatment=False, n_estimators=50, cv=2)
                    cf_est.fit(Y_vals, T_vals, X=X_vals)
                    results.append({'Estimator': 'Causal Forest', 'Resolution': level_name, 'Stage': stage, 'Component': clean_name(t_col),
                                    'Coefficient': cf_est.ate(X_vals), 'P_Value': cf_est.ate__inference().pvalue(), 'T_Stat': np.nan})
                except Exception:
                    pass
                
                # D. DR Estimator
                try:
                    y_hat = lgb.LGBMRegressor(n_estimators=30, n_jobs=-1, verbose=-1).fit(X_vals, Y_vals).predict(X_vals)
                    y_res, t_res = Y_vals - y_hat, T_vals - t_hat
                    den_samp = stats.norm.pdf(T_vals, t_hat, np.std(t_res) + 1e-8)
                    w_dr = np.clip(1.0 / (den_samp + 1e-8), 0, np.percentile(1.0 / (den_samp + 1e-8), 99))
                    model_dr = sm.WLS(y_res, sm.add_constant(t_res), weights=w_dr).fit()
                    results.append({'Estimator': 'DR Estimator', 'Resolution': level_name, 'Stage': stage, 'Component': clean_name(t_col),
                                    'Coefficient': model_dr.params[1], 'P_Value': model_dr.pvalues[1], 'T_Stat': model_dr.tvalues[1]})
                except Exception:
                    pass

    df_final = pd.DataFrame(results)
    out_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_Baseline_Comparison_v1_0509.csv")
    df_final.to_csv(out_path, index=False)

    logger.info("\n" + "="*80)
    logger.info(f"🎉 {DATASET_NAME} 基线对比实验圆满完成！")
    logger.info(f"💾 结果保存路径: {out_path}")
    logger.info("========================================================\n")

if __name__ == "__main__":
    main()
    
    
    # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/causal_baseline_comparison_amazon_beauty_0509.py