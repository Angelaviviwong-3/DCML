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
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
DATA_PATH = os.path.join(BASE_DIR, "build_dataset/DCML_C_5.parquet")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"baseline_estimators_suning_0509_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(">>> [Section 5.3.2] 运行全量传统基线估计器对比 (OLS, IPW, DR, CF, DML) - Suning")
    logger.info("="*80)

    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ 找不到数据文件: {DATA_PATH}")
        return
    
    logger.info(">>> 正在加载数据集...")
    df = pd.read_parquet(DATA_PATH)
    
    # 🌟 核心优化：全局随机下采样，加速计算并防止 OOM
    GLOBAL_SAMPLE_SIZE = 200000
    if len(df) > GLOBAL_SAMPLE_SIZE:
        logger.info(f"   ⚠️ 数据量过大 ({len(df)} 行)，全局随机下采样至 {GLOBAL_SAMPLE_SIZE} 行...")
        df = df.sample(n=GLOBAL_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        logger.info(f"   - 数据量: {len(df)} 行")

    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    T_macro_raw = ['T_int_fac_calib', 'T_int_vis_calib', 'T_con_mkt_calib', 'T_int_sem', 'T_con_soc', 'T_con_rat']
    T_atomic_raw = ['atomic_mkt_price_calib', 'atomic_a_gft_calib', 'atomic_a_sub_calib', 'atomic_a_urg_calib', 
                    'atomic_a_spec_calib', 'atomic_a_str_calib']
    
    T_macro_exist = [c for c in T_macro_raw if c in df.columns]
    T_atomic_exist = []
    for c in T_atomic_raw:
        if c in df.columns:
            T_atomic_exist.append(c)
        elif c.replace('mkt_price', 'a_pri') in df.columns:
            T_atomic_exist.append(c.replace('mkt_price', 'a_pri'))
            
    if 'T_int_fac_calib' in df.columns and 'T_int_fac_calib' not in T_atomic_exist:
        T_atomic_exist.append('T_int_fac_calib')

    # 定义混淆变量 X
    exclude_cols = ['user_id', 'item_id', 'timestamp', 'H_level', 'M_norm'] + Y_cols + T_macro_exist + T_atomic_exist
    redundant_cols = [c for c in df.columns if '_calibrated_calib' in c]
    X_cols = [col for col in df.columns if col not in exclude_cols and col not in redundant_cols]

    def clean_name(t):
        return t.replace('res_', '').replace('_calib', '').replace('_pure', '').replace('atomic_', '')
    
    results = []
    all_unique_treatments = list(set(T_macro_exist + T_atomic_exist))

    # ==========================================
    # Pipeline 1/3: Naive OLS / MCRE + OLS
    # ==========================================
    logger.info("\n>>> [Pipeline 1 & 3] 正在运行 MCRE + OLS (忽略环境与选择偏差)...")
    for res_level, t_list in [('Macro', T_macro_exist), ('Atomic', T_atomic_exist)]:
        X_naive = sm.add_constant(df[t_list])
        for y_col in Y_cols:
            stage = y_col.replace('y_', '')
            model_naive = sm.OLS(df[y_col], X_naive).fit()
            for t_col in t_list:
                results.append({'Estimator': 'Naive OLS', 'Resolution': res_level, 'Stage': stage, 'Component': clean_name(t_col),
                                'Coefficient': model_naive.params[t_col], 'P_Value': model_naive.pvalues[t_col], 'T_Stat': model_naive.tvalues[t_col]})

    # ==========================================
    # Pipeline 2: Statistical-Only (IPW, DR, CF, DML)
    # ==========================================
    logger.info("\n>>> [Pipeline 2] 正在运行高级因果估计器 (IPW, DR, Causal Forest, DML)...")
    
    X_vals = df[X_cols].values

    for t_col in tqdm(all_unique_treatments, desc="Running Advanced Estimators"):
        belong_to = []
        if t_col in T_macro_exist: belong_to.append('Macro')
        if t_col in T_atomic_exist: belong_to.append('Atomic')

        T_vals = df[t_col].values

        # --- A. IPW ---
        model_t_ipw = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        model_t_ipw.fit(df[X_cols], df[t_col])
        t_hat_ipw = model_t_ipw.predict(df[X_cols])
        
        std_resid = np.std(df[t_col] - t_hat_ipw)
        num = stats.norm.pdf(df[t_col], np.mean(df[t_col]), np.std(df[t_col]))
        den = stats.norm.pdf(df[t_col], t_hat_ipw, std_resid)
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
                    y_res, t_res = Y_vals - y_hat, T_vals - t_hat_ipw
                    den_samp = stats.norm.pdf(T_vals, t_hat_ipw, np.std(t_res) + 1e-8)
                    w_dr = np.clip(1.0 / (den_samp + 1e-8), 0, np.percentile(1.0 / (den_samp + 1e-8), 99))
                    model_dr = sm.WLS(y_res, sm.add_constant(t_res), weights=w_dr).fit()
                    results.append({'Estimator': 'DR Estimator', 'Resolution': level_name, 'Stage': stage, 'Component': clean_name(t_col),
                                    'Coefficient': model_dr.params[1], 'P_Value': model_dr.pvalues[1], 'T_Stat': model_dr.tvalues[1]})
                except Exception:
                    pass

    df_baselines = pd.DataFrame(results)
    out_path = os.path.join(SAVE_DIR, "Baseline_Comparison_Results_0509.csv")
    df_baselines.to_csv(out_path, index=False)

    logger.info("\n" + "="*80)
    logger.info("🎉 传统基线结果已生成！")
    logger.info(f"保存路径: {out_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
    
    # python /home/xzhe162/wh_workspace/casual/model/M_suning/DML/causal_baseline_comparison_suning_0509.py