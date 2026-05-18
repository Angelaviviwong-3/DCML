import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import statsmodels.api as sm
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径配置
# ==========================================
BASE_WORKSPACE = "/home/xzhe162/wh_workspace/casual"
DATASET_NAME = "amazon_beauty"
BASE_DIR = os.path.join(BASE_WORKSPACE, "processed_data", DATASET_NAME)

# 输入路径
SET_C_PATH = os.path.join(BASE_DIR, "build_dataset", f"{DATASET_NAME}_DCML_Set_C_1.parquet")
RES_PATH = os.path.join(BASE_DIR, "DML_Results", f"{DATASET_NAME}_DML_Residuals_Final.parquet")

# 输出路径
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = os.path.join(BASE_WORKSPACE, "log")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"causal_mediation_{DATASET_NAME}_full_0510.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ==========================================
# 1. Bootstrap 中介效应核心引擎
# ==========================================
def bootstrap_mediation(df, t_col, m_col, y_col, n_iterations=1000, seed=42):
    """
    执行中介效应检验 (针对去混淆后的残差信号)
    Path a: T -> M
    Path b: M -> Y (controlled for T)
    Indirect Effect = a * b
    """
    np.random.seed(seed)
    deltas = []
    
    T = df[t_col].values
    M = df[m_col].values
    Y = df[y_col].values
    n_samples = len(df)
    
    # === 点估计 (Point Estimate) ===
    # alpha_1: T 对 M 的回归系数 (a)
    alpha_1 = np.linalg.lstsq(sm.add_constant(T), M, rcond=None)[0][1]
    # beta_2: M 对 Y 的回归系数 (控制 T) (b)
    beta_2 = np.linalg.lstsq(sm.add_constant(np.column_stack((T, M))), Y, rcond=None)[0][2]
    delta_point = alpha_1 * beta_2

    # === Bootstrap ===
    for i in range(n_iterations):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        T_b, M_b, Y_b = T[idx], M[idx], Y[idx]
        a = np.linalg.lstsq(sm.add_constant(T_b), M_b, rcond=None)[0][1]
        b = np.linalg.lstsq(sm.add_constant(np.column_stack((T_b, M_b))), Y_b, rcond=None)[0][2]
        deltas.append(a * b)
        
    deltas = np.array(deltas)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    p_val_approx = min((deltas > 0).mean(), (deltas < 0).mean()) * 2.0
    
    return alpha_1, beta_2, delta_point, ci_lower, ci_upper, p_val_approx

# ==========================================
# 2. 主流程
# ==========================================
def main():
    logger.info("="*80)
    logger.info(f">>> [DCML Mediation] 开始亚马逊 {DATASET_NAME} 全变量中介效应检验")
    logger.info("="*80)

    # 1. 载入数据并合并 (需要 Set C 里的混淆变量 X 来提纯 M)
    logger.info(">>> 1. 正在载入数据并对齐残差矩阵...")
    df_raw = pd.read_parquet(SET_C_PATH)
    df_res = pd.read_parquet(RES_PATH)
    
    # 定义控制变量 X
    X_cols = ['x_pri_log', 'x_pos_freq', 'user_review_count_scaled', 'user_avg_rating_scaled']
    
    df = pd.merge(
        df_res, 
        df_raw[['item_id', 'user_id', 'timestamp'] + X_cols], 
        on=['item_id', 'user_id', 'timestamp'], 
        how='inner'
    )
    logger.info(f"   - 样本对齐完成: {len(df)} 行")
    del df_raw, df_res; gc.collect()

    # 千万级数据抽样 100万行以保证 Bootstrap 在合理时间内完成
    if len(df) > 1000000:
        logger.info("   - 样本量巨大，正在抽样 1,000,000 条进行 Bootstrap...")
        df = df.sample(n=1000000, random_state=42)

    # 2. 中介变量 M_norm 残差化提纯 (排除 X 干扰)
    logger.info(">>> 2. 正在执行中介变量 M_norm 的去混淆残差化...")
    model_m = sm.OLS(df['M_norm'], sm.add_constant(df[X_cols])).fit()
    df['M_tilde'] = df['M_norm'] - model_m.predict(sm.add_constant(df[X_cols]))

    # 3. 变量清单 (6 Macro + 6 Atomic，严格对应体检报告中的列名)
    macro_base = ['T_con_mkt', 'T_con_soc', 'T_con_rat', 'T_int_fac', 'T_int_vis', 'T_int_sem']
    atomic_base = ['atomic_a_pri', 'atomic_a_gft', 'atomic_a_sub', 'atomic_a_urg', 'atomic_a_spec', 'atomic_a_str']
    all_target_base_names = macro_base + atomic_base
    
    # 动态匹配 res_ 前缀的残差列
    valid_cols = []
    for base in all_target_base_names:
        found = False
        for col in df.columns:
            # 匹配例如 res_T_con_mkt_calibrated 或 res_T_con_soc
            if col.startswith('res_' + base):
                valid_cols.append(col)
                found = True
                break
        if not found:
            logger.warning(f"⚠️ 警告: 无法识别变量 {base} 的残差列")

    Y_cols = ['res_y_click', 'res_y_cart', 'res_y_purchase']
    results = []

    logger.info(f"\n>>> 3. 开始执行 36 组路径的中介检验 (12 变量 x 3 阶段)...")
    
    for t_col in tqdm(valid_cols, desc="Components"):
        for y_col in Y_cols:
            # 清洗名称用于最终结果显示
            clean_t = t_col.replace('res_', '').replace('_calibrated', '').replace('atomic_', '')
            clean_y = y_col.replace('res_y_', '')
            
            a, b, delta, ci_l, ci_u, p = bootstrap_mediation(
                df=df, t_col=t_col, m_col='M_tilde', y_col=y_col, n_iterations=1000
            )
            
            sig = "Yes" if (ci_l * ci_u > 0) else "No"
            
            results.append({
                'Heuristic_Cue': clean_t,
                'Funnel_Stage': clean_y,
                'Path_a_Coef': a,
                'Path_b_Coef': b,
                'Indirect_Effect': delta,
                '95%_CI_Lower': ci_l,
                '95%_CI_Upper': ci_u,
                'P_Value': p,
                'Significant': sig
            })

    # 4. 统计校正与保存
    df_results = pd.DataFrame(results)
    # Bonferroni 校正
    df_results['P_Value_Bonferroni'] = np.minimum(df_results['P_Value'] * len(df_results), 1.0)
    df_results['Sig_Bonferroni'] = np.where(
        (df_results['95%_CI_Lower'] * df_results['95%_CI_Upper'] > 0) & (df_results['P_Value_Bonferroni'] < 0.05),
        "Yes", "No"
    )

    out_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_Mediation_Results_0510.csv")
    df_results.to_csv(out_path, index=False)

    logger.info("\n" + "="*80)
    logger.info(f"🎉 任务完成！{DATASET_NAME} 全变量中介结果已保存至:\n{out_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
    # python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/dcml_step3_mediation_amazon_beauty.py
    