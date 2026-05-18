import pandas as pd
import numpy as np
import os
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
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
DATA_PATH = os.path.join(BASE_DIR, "build_dataset/DCML_C_5.parquet")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"baseline_estimators_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(">>> [Section 5.3.2] 运行传统基线估计器 (Naive OLS & IPW) 全量变量版")
    logger.info(">>> [研究设计对齐]: 分别回归 Macro 组与 Atomic 组")
    logger.info("="*80)

    # 1. 载入推理集数据
    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ 找不到数据文件: {DATA_PATH}")
        return
    df = pd.read_parquet(DATA_PATH)
    
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    T_macro_raw = ['T_int_fac_calib', 'T_int_vis_calib', 'T_con_mkt_calib', 'T_int_sem', 'T_con_soc', 'T_con_rat']
    T_atomic_raw = ['atomic_mkt_price_calib', 'atomic_a_gft_calib', 'atomic_a_sub_calib', 'atomic_a_urg_calib', 
                    'atomic_a_spec_calib', 'atomic_a_str_calib']
    
    # 动态匹配存在的列名 (处理 mkt_price -> a_pri 的命名变化)
    T_macro_exist = [c for c in T_macro_raw if c in df.columns]
    T_atomic_exist = []
    for c in T_atomic_raw:
        if c in df.columns:
            T_atomic_exist.append(c)
        elif c.replace('mkt_price', 'a_pri') in df.columns:
            T_atomic_exist.append(c.replace('mkt_price', 'a_pri'))
            
    # 此外，原子层面也需要加入事实密度 (论文设定)
    if 'T_int_fac_calib' in df.columns and 'T_int_fac_calib' not in T_atomic_exist:
        T_atomic_exist.append('T_int_fac_calib')

    logger.info(f"   - 识别到 Macro 特征 ({len(T_macro_exist)} 个): {T_macro_exist}")
    logger.info(f"   - 识别到 Atomic 特征 ({len(T_atomic_exist)} 个): {T_atomic_exist}")

    # 定义混淆变量 X
    exclude_cols = ['user_id', 'item_id', 'timestamp', 'H_level', 'M_norm'] + Y_cols + T_macro_exist + T_atomic_exist
    redundant_cols = [c for c in df.columns if '_calibrated_calib' in c]
    X_cols = [col for col in df.columns if col not in exclude_cols and col not in redundant_cols]
    
    # 清洗变量名 (对齐 DCML 输出格式)
    def clean_name(t):
        return t.replace('res_', '').replace('_calib', '').replace('_pure', '').replace('atomic_', '')
    
    results = []

    # ---------------------------------------------------------
    # Baseline 1: Naive OLS (完全忽略 X)
    # ---------------------------------------------------------
    logger.info("\n>>> 1. 正在运行 Naive OLS (忽略环境与选择偏差)...")
    
    for res_level, t_list in [('Macro', T_macro_exist), ('Atomic', T_atomic_exist)]:
        X_naive = sm.add_constant(df[t_list])
        for y_col in Y_cols:
            stage = y_col.replace('y_', '')
            model_naive = sm.OLS(df[y_col], X_naive).fit()
            
            for t_col in t_list:
                results.append({
                    'Estimator': 'Naive OLS',
                    'Resolution': res_level,
                    'H_level': 'All (ATE)',
                    'Stage': stage,
                    'Component': clean_name(t_col),
                    'Coefficient': model_naive.params[t_col],
                    'Std_Error': model_naive.bse[t_col],
                    'P_Value': model_naive.pvalues[t_col],
                    'T_Stat': model_naive.tvalues[t_col]
                })

    # ---------------------------------------------------------
    # Baseline 2: Statistical-Only IPW (连续变量倾向得分加权)
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 正在运行 IPW (验证连续倾向得分权重爆炸)...")
    # 抽取 Top 50 混淆变量以防全维 X 训练极度缓慢
    X_sample = df[X_cols].sample(n=min(50, len(X_cols)), axis=1, random_state=42)
    
    # 获取所有的独特特征集合
    all_unique_treatments = list(set(T_macro_exist + T_atomic_exist))
    
    for t_col in tqdm(all_unique_treatments, desc="Computing GPS & IPW for all treatments"):
        # 拟合 E[T|X]
        model_t = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        model_t.fit(X_sample, df[t_col])
        t_hat = model_t.predict(X_sample)
        
        # 计算广义连续倾向得分权重 W = f(T) / f(T|X)
        std_resid = np.std(df[t_col] - t_hat)
        num = stats.norm.pdf(df[t_col], np.mean(df[t_col]), np.std(df[t_col]))
        den = stats.norm.pdf(df[t_col], t_hat, std_resid)
        weights = num / (den + 1e-8)
        
        # 为了让模型能够吐出结果，进行截断 (防止无穷大，但依然保留巨大方差)
        weights = np.clip(weights, 0, np.percentile(weights, 99.5))
        
        # 判断当前特征属于哪个层级（如果同时存在于两组，记录两次）
        belong_to = []
        if t_col in T_macro_exist: belong_to.append('Macro')
        if t_col in T_atomic_exist: belong_to.append('Atomic')
            
        for res_level in belong_to:
            for y_col in Y_cols:
                stage = y_col.replace('y_', '')
                # 这里针对单个变量做 IPW (标准连续型 IPW 操作)
                X_wls = sm.add_constant(df[t_col])
                model_ipw = sm.WLS(df[y_col], X_wls, weights=weights).fit()
                
                results.append({
                    'Estimator': 'IPW',
                    'Resolution': res_level,
                    'H_level': 'All (ATE)',
                    'Stage': stage,
                    'Component': clean_name(t_col),
                    'Coefficient': model_ipw.params[t_col],
                    'Std_Error': model_ipw.bse[t_col],
                    'P_Value': model_ipw.pvalues[t_col],
                    'T_Stat': model_ipw.tvalues[t_col]
                })

    # ---------------------------------------------------------
    # 保存结果
    # ---------------------------------------------------------
    df_baselines = pd.DataFrame(results)
    
    # 确保保存路径正确且不冲突
    out_path = os.path.join(SAVE_DIR, "Baseline_Comparison_Results.csv")
    df_baselines.to_csv(out_path, index=False)

    logger.info("\n" + "="*80)
    logger.info("🎉 传统基线结果已全量生成！")
    logger.info(f"保存路径: {out_path}")
    logger.info(f"记录数量: {len(df_baselines)} 行")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
    #. python /home/xzhe162/wh_workspace/casual/model/DML/causal_baseline_comparison.py