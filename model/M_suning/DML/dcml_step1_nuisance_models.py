import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import lightgbm as lgb
import joblib
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径与环境配置
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
# 对应你 log 里的文件命名
SET_B_PATH = os.path.join(BASE_DIR, "build_dataset/DCML_B_5.parquet")
SET_C_PATH = os.path.join(BASE_DIR, "build_dataset/DCML_C_5.parquet")

SAVE_DIR = os.path.join(BASE_DIR, "DML_Results")
MODEL_DIR = os.path.join(SAVE_DIR, "nuisance_models_3")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"dml_phase1_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(">>> [DCML Phase I & II] 训练 Nuisance Models 并计算因果残差 (v3 宏观+原子版)")
    logger.info("="*80)

    # ---------------------------------------------------------
    # STEP 1: 载入数据并严格定义变量空间
    # ---------------------------------------------------------
    logger.info(">>> 1. 载入 Set B (训练集) 和 Set C (推理集)...")
    df_b = pd.read_parquet(SET_B_PATH)
    df_c = pd.read_parquet(SET_C_PATH)

    # 1. 结果变量 (Outcomes)
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    
    # 2. 处理变量 (Treatments) - 手动精准定义，避开 redundancy 列
    T_macro = ['T_int_fac_calib', 'T_int_vis_calib', 'T_con_mkt_calib', 'T_int_sem', 'T_con_soc', 'T_con_rat']
    T_atomic = ['atomic_mkt_price_calib', 'atomic_a_gft_calib', 'atomic_a_sub_calib', 'atomic_a_urg_calib', 
                'atomic_a_spec_calib', 'atomic_a_str_calib']
    
    # 检查列名以防拼写不一致（适配你的实际列名）
    T_cols = []
    for t in T_macro + T_atomic:
        if t in df_b.columns: T_cols.append(t)
        # 兼容你的表头命名
        elif t.replace('mkt_price', 'a_pri') in df_b.columns: T_cols.append(t.replace('mkt_price', 'a_pri'))
        
    # 3. 严格剔除不能进入 X 的列（主键、目标、治疗、调节、中介、以及带 _calibrated_calib 的废列）
    exclude_cols = ['user_id', 'item_id', 'timestamp', 'H_level', 'M_norm'] + Y_cols + T_cols
    redundant_cols = [c for c in df_b.columns if '_calibrated_calib' in c]
    
    X_cols = [col for col in df_b.columns if col not in exclude_cols and col not in redundant_cols]
    
    logger.info(f"   - 🎯 结果变量 (Y): {Y_cols}")
    logger.info(f"   - 💊 处理变量 (T): {T_cols} (共 {len(T_cols)} 个)")
    logger.info(f"   - 🧬 混淆变量 (X): {len(X_cols)} 维 (已排除 H_level 和 M_norm 防止数据泄漏)")

    X_train, X_test = df_b[X_cols], df_c[X_cols]
    
    # 初始化残差表 (需要保留主键, H_level 算 CATE, M_norm 算中介)
    df_residuals = df_c[['user_id', 'item_id', 'timestamp', 'H_level', 'M_norm']].copy()
    
    # 存一份 T 的原始值，论文 Eq. 17 的 Gram-Schmidt 投影必须用到它
    for t_name in T_cols:
        df_residuals[f'raw_{t_name}'] = df_c[t_name]

    # LGBM 超参数 (防过拟合)
    lgb_params_reg = {'objective': 'regression', 'n_estimators': 200, 'learning_rate': 0.05, 
                      'max_depth': 6, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 
                      'random_state': 2025, 'n_jobs': -1, 'verbose': -1}
    lgb_params_clf = lgb_params_reg.copy()
    lgb_params_clf['objective'] = 'binary'

    # ---------------------------------------------------------
    # STEP 2: 拟合 Treatment 模型 \hat{T} = E[T|X] 
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 拟合 Treatment 倾向得分模型...")
    for t_name in tqdm(T_cols, desc="Training T Models"):
        model_t = lgb.LGBMRegressor(**lgb_params_reg)
        model_t.fit(X_train, df_b[t_name])
        
        # 预测：除了 T_con_soc 是对数热度外，其余 T 均在 [0, 1] 之间
        t_hat_test = model_t.predict(X_test)
        if t_name != 'T_con_soc':
            t_hat_test = np.clip(t_hat_test, 0.0, 1.0)
        else:
            t_hat_test = np.clip(t_hat_test, 0.0, None) # 热度不能为负
            
        df_residuals[f'res_{t_name}'] = df_c[t_name] - t_hat_test
        joblib.dump(model_t, os.path.join(MODEL_DIR, f"model_{t_name}.pkl"))

    # ---------------------------------------------------------
    # STEP 3: 拟合 Outcome 模型 \hat{Y} = E[Y|X]
    # ---------------------------------------------------------
    logger.info("\n>>> 3. 拟合 Outcome 预测模型...")
    for y_name in tqdm(Y_cols, desc="Training Y Models"):
        model_y = lgb.LGBMClassifier(**lgb_params_clf)
        model_y.fit(X_train, df_b[y_name])
        
        y_hat_test = model_y.predict_proba(X_test)[:, 1]
        df_residuals[f'res_{y_name}'] = df_c[y_name] - y_hat_test
        joblib.dump(model_y, os.path.join(MODEL_DIR, f"model_{y_name}.pkl"))

    # ---------------------------------------------------------
    # STEP 4: 保存核心残差矩阵
    # ---------------------------------------------------------
    logger.info("\n>>> 4. 正在保存 DML 核心残差矩阵...")
    res_path = os.path.join(SAVE_DIR, "DCML_Residuals_3.parquet")
    df_residuals.to_parquet(res_path, index=False)

    logger.info("="*80)
    logger.info("🎉 DML 第一阶段 (干扰模型训练与残差化) 圆满结束！")
    logger.info(f"   - 残差矩阵已保存至: {res_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
 #. python /home/xzhe162/wh_workspace/casual/model/DML/dcml_step1_nuisance_models.py