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
# 0. 路径与环境配置
# ==========================================
DATASET_NAME = "amazon_appliances"
BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"

# ⚠️ 输入路径：对接 dataset_builder 生成的 v3 数据集
SET_B_PATH = os.path.join(BASE_DIR, f"build_dataset/{DATASET_NAME}_DCML_Set_B_3.parquet")
SET_C_PATH = os.path.join(BASE_DIR, f"build_dataset/{DATASET_NAME}_DCML_Set_C_3.parquet")

# ✅ 输出路径
SAVE_DIR = os.path.join(BASE_DIR, "DML_Results")
MODEL_DIR = os.path.join(SAVE_DIR, "nuisance_models_2")
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
    logger.info(">>> 1. 正在载入千万级数据集 (Set B & Set C)...")
    # 为了节省内存，读取时指定必要字段
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
    # 根据你的预处理结果，X 包含价格、位置、卷入度以及用户历史行为指标
    X_cols = ['x_pri_log', 'x_pos_freq', 'H_level', 'user_review_count_scaled', 'user_avg_rating_scaled']
    
    logger.info(f"   - 🎯 结果变量 (Y): {Y_cols}")
    logger.info(f"   - 💊 处理变量 (T): {len(T_cols)}个核心变量")
    logger.info(f"   - 🧬 混淆变量 (X): {len(X_cols)}维基础特征")

    X_train = df_b[X_cols]
    X_test = df_c[X_cols]
    
    # 初始化残差表 (保留主键和后续需要的调节变量 H_level, 中介变量 M_norm)
    df_residuals = df_c[['user_id', 'item_id', 'timestamp', 'H_level', 'M_norm']].copy()
    
    # 存一份 T 的原始值，用于后续 Stage III 的 Gram-Schmidt 投影
    for t_name in T_cols:
        df_residuals[f'raw_{t_name}'] = df_c[t_name]

    # LGBM 超参数优化 (针对 A100 服务器, 开启多线程加速)
    lgb_params = {
        'n_estimators': 100,      # 千万级数据，100个树已足够捕捉非线性，防止过拟合
        'learning_rate': 0.1, 
        'max_depth': 8, 
        'num_leaves': 63, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'n_jobs': -1,             # 使用所有 CPU 核心
        'random_state': 2025,
        'verbose': -1
    }

    # ---------------------------------------------------------
    # STEP 2: 拟合 Treatment 模型 \hat{T} = E[T|X]
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 正在计算 Treatment 残差 (正交化关键步)...")
    for t_name in tqdm(T_cols, desc="Residualizing Treatments"):
        model_t = lgb.LGBMRegressor(**lgb_params)
        model_t.fit(X_train, df_b[t_name])
        
        t_hat = model_t.predict(X_test)
        
        # 边界修剪：确保预测值符合物理含义 [0, 1]
        if t_name != 'T_con_soc': # soc 是对数热度，不设上限
            t_hat = np.clip(t_hat, 0.0, 1.0)
        else:
            t_hat = np.clip(t_hat, 0.0, None)
            
        df_residuals[f'res_{t_name}'] = df_c[t_name] - t_hat
        
        # 释放模型内存并存盘
        joblib.dump(model_t, os.path.join(MODEL_DIR, f"model_T_{t_name}.pkl"))
        del model_t
        gc.collect()

    # ---------------------------------------------------------
    # STEP 3: 拟合 Outcome 模型 \hat{Y} = E[Y|X]
    # ---------------------------------------------------------
    logger.info("\n>>> 3. 正在计算 Outcome 残差 (漏斗各阶段)...")
    for y_name in tqdm(Y_cols, desc="Residualizing Outcomes"):
        # Y 是二分类，使用 predict_proba 预测概率
        model_y = lgb.LGBMClassifier(objective='binary', **lgb_params)
        model_y.fit(X_train, df_b[y_name])
        
        y_prob = model_y.predict_proba(X_test)[:, 1]
        df_residuals[f'res_{y_name}'] = df_c[y_name] - y_prob
        
        joblib.dump(model_y, os.path.join(MODEL_DIR, f"model_Y_{y_name}.pkl"))
        del model_y
        gc.collect()

    # ---------------------------------------------------------
    # STEP 4: 保存结果
    # ---------------------------------------------------------
    logger.info("\n>>> 4. 正在保存最终残差矩阵...")
    final_res_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_DML_Residuals_Final_2.parquet")
    df_residuals.to_parquet(final_res_path, index=False)

    logger.info("="*80)
    logger.info("🎉 因果 Nuisance 建模完成！")
    logger.info(f"   - 样本量: {len(df_residuals):,}")
    logger.info(f"   - 残差矩阵已保存至: {final_res_path}")
    logger.info("========================================================================\n")

if __name__ == "__main__":
    main()
    
    
    # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/dml_step1_nuisance_amazon_app.py