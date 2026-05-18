#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================
# 修复 OpenBLAS 多核并行溢出的段错误
# ⚠️ 必须放在所有其他 import 之前！
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import pandas as pd
import numpy as np
import logging
import time
import joblib
import gc
from datetime import datetime
import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 参数解析与路径配置
# ==========================================
parser = logging.getLogger(__name__)
DATASET_NAME = "amazon_beauty"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"

# ⚠️ 输入路径：对接 Beauty 数据集构建产生的 _1 结尾文件
SET_B_PATH = os.path.join(BASE_DIR, f"build_dataset/{DATASET_NAME}_DCML_Set_B_1.parquet")
SET_C_PATH = os.path.join(BASE_DIR, f"build_dataset/{DATASET_NAME}_DCML_Set_C_1.parquet")

# ✅ 输出路径
SAVE_DIR = os.path.join(BASE_DIR, "DML_Results")
MODEL_DIR = os.path.join(SAVE_DIR, "nuisance_models")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"dml_nuisance_{DATASET_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(f">>> [Stage II: Nuisance Modeling] 亚马逊 {DATASET_NAME} 因果残差计算")
    logger.info("="*80)

    # ---------------------------------------------------------
    # STEP 1: 载入数据
    # ---------------------------------------------------------
    logger.info(f">>> 1. 正在载入 {DATASET_NAME} 数据集 (Set B & Set C)...")
    df_b = pd.read_parquet(SET_B_PATH)
    df_c = pd.read_parquet(SET_C_PATH)

    # 1. 结果变量 (Outcomes)
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    
    # 2. 处理变量 (Treatments) - 严格匹配 6宏观 + 6原子
    T_macro = [
        'T_int_fac_calibrated', 'T_int_vis_calibrated', 'T_con_mkt_calibrated', 
        'T_int_sem', 'T_con_soc', 'T_con_rat'
    ]
    T_atomic = [
        'atomic_a_pri_calibrated', 'atomic_a_gft_calibrated', 'atomic_a_sub_calibrated', 
        'atomic_a_urg_calibrated', 'atomic_a_spec_calibrated', 'atomic_a_str_calibrated'
    ]
    T_cols = T_macro + T_atomic
    
    # 3. 混淆变量 (Confounders X)
    X_cols = ['x_pri_log', 'x_pos_freq', 'H_level', 'user_review_count_scaled', 'user_avg_rating_scaled']
    
    logger.info(f"   - 🎯 结果变量 (Y): {Y_cols}")
    logger.info(f"   - 💊 处理变量 (T): {len(T_cols)}个核心因果处理变量")
    logger.info(f"   - 🧬 混淆变量 (X): {len(X_cols)}维混淆特征")

    X_train = df_b[X_cols]
    X_test = df_c[X_cols]
    
    # 初始化残差表 (保留主键、调节变量 H_level, 中介变量 M_norm)
    df_residuals = df_c[['user_id', 'item_id', 'timestamp', 'H_level', 'M_norm']].copy()
    
    # 存一份 T 的原始值，用于后续 Stage III 正交化回归
    for t_name in T_cols:
        df_residuals[f'raw_{t_name}'] = df_c[t_name]

    # LGBM 参数配置
    lgb_params = {
        'n_estimators': 100,      
        'learning_rate': 0.1, 
        'max_depth': 8, 
        'num_leaves': 63, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'n_jobs': -1,             
        'random_state': 2025,
        'verbose': -1
    }

    # ---------------------------------------------------------
    # STEP 2: 拟合 Treatment 残差 \tilde{T} = T - E[T|X]
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 正在计算多模态处理变量 (Treatments) 残差...")
    for t_name in tqdm(T_cols, desc=f"Processing T for {DATASET_NAME}"):
        model_t = lgb.LGBMRegressor(**lgb_params)
        model_t.fit(X_train, df_b[t_name])
        
        t_hat = model_t.predict(X_test)
        
        # 截断处理
        if t_name != 'T_con_soc':
            t_hat = np.clip(t_hat, 0.0, 1.0)
        else:
            t_hat = np.clip(t_hat, 0.0, None)
            
        df_residuals[f'res_{t_name}'] = df_c[t_name] - t_hat
        
        # 保存模型并释放资源
        joblib.dump(model_t, os.path.join(MODEL_DIR, f"model_T_{t_name}_beauty.pkl"))
        del model_t
        gc.collect()

    # ---------------------------------------------------------
    # STEP 3: 拟合 Outcome 残差 \tilde{Y} = Y - E[Y|X]
    # ---------------------------------------------------------
    logger.info("\n>>> 3. 正在计算转化结果 (Outcomes) 残差...")
    for y_name in tqdm(Y_cols, desc=f"Processing Y for {DATASET_NAME}"):
        model_y = lgb.LGBMClassifier(objective='binary', **lgb_params)
        model_y.fit(X_train, df_b[y_name])
        
        y_prob = model_y.predict_proba(X_test)[:, 1]
        df_residuals[f'res_{y_name}'] = df_c[y_name] - y_prob
        
        joblib.dump(model_y, os.path.join(MODEL_DIR, f"model_Y_{y_name}_beauty.pkl"))
        del model_y
        gc.collect()

    # ---------------------------------------------------------
    # STEP 4: 结果保存
    # ---------------------------------------------------------
    logger.info("\n>>> 4. 正在保存 Beauty 最终残差矩阵...")
    final_res_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_DML_Residuals_Final.parquet")
    df_residuals.to_parquet(final_res_path, index=False)

    logger.info("="*80)
    logger.info(f"🎉 {DATASET_NAME} DML Nuisance 建模完成！")
    logger.info(f"   - 最终残差矩阵路径: {final_res_path}")
    logger.info("========================================================================\n")

if __name__ == "__main__":
    main()
    
   # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/dml_step1_nuisance_amazon_beauty.py