import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径配置 (对接 v4 残差矩阵, 输出加 _2)
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/DML_Results"
# ⚠️ 读取 Phase I 跑出来的最新残差矩阵
RES_PATH = os.path.join(BASE_DIR, "DCML_Residuals_3.parquet")

SAVE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/Final_Causal_Output"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置
log_filename = f"causal_inference_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(">>> [DCML Phase III & IV] 最终残差正交化与因果效应(ATE/CATE)估计")
    logger.info(">>> [研究设计对齐]: 6大宏观组件 + 6大原子组件 (事实密度复用宏观表示)")
    logger.info("="*80)

    # 1. 载入残差数据
    if not os.path.exists(RES_PATH):
        logger.error(f"❌ 找不到残差文件: {RES_PATH}")
        return
    df = pd.read_parquet(RES_PATH)
    Y_cols = ['res_y_click', 'res_y_cart', 'res_y_purchase']
    
    # ---------------------------------------------------------
    # STEP 1: Multivariate Gram-Schmidt 正交化 (Phase III)
    # ---------------------------------------------------------
    # 理论逻辑 (Eq.17)：System 1 (Conformity) 先行，System 2 (Interest) 在其正交补空间上。
    logger.info("\n>>> 1. 执行 Multivariate Gram-Schmidt 正交化投影 (Eq. 17)...")
    
    # 【1.1 宏观层面 (Macro) 正交化】
    # System 1: Conformity Path (营销视觉、社会信号、启发式评分)
    con_macro_cols = ['res_T_con_mkt_calib', 'res_T_con_soc', 'res_T_con_rat']
    # System 2: Interest Path (事实密度、功能视觉、语义对齐)
    int_macro_cols = ['res_T_int_fac_calib', 'res_T_int_vis_calib', 'res_T_int_sem']
    
    macro_pure_cols = []
    for target in int_macro_cols:
        model = sm.OLS(df[target], df[con_macro_cols]).fit()
        pure_name = f"{target}_pure"
        df[pure_name] = df[target] - model.predict(df[con_macro_cols])
        macro_pure_cols.append(pure_name)
        logger.info(f"   [诊断-Macro] {target} 被 System1 解释的 R-squared: {model.rsquared:.4f}")
        
    T_star_macro = con_macro_cols + macro_pure_cols

    # 【1.2 原子层面 (Atomic) 正交化】
    # System 1 (营销原子): 价格, 赠品, 补贴, 紧迫感
    con_atomic_cols = [
        'res_atomic_a_pri_calib', 'res_atomic_a_gft_calib', 
        'res_atomic_a_sub_calib', 'res_atomic_a_urg_calib'
    ]
    # System 2 (视觉原子): 规格, 结构
    int_atomic_cols = ['res_atomic_a_spec_calib', 'res_atomic_a_str_calib']
    
    atomic_pure_cols = []
    for target in int_atomic_cols:
        model = sm.OLS(df[target], df[con_atomic_cols]).fit()
        pure_name = f"{target}_pure"
        df[pure_name] = df[target] - model.predict(df[con_atomic_cols])
        atomic_pure_cols.append(pure_name)
        logger.info(f"   [诊断-Atomic] {target} 被 营销原子 解释的 R-squared: {model.rsquared:.4f}")

    # 组装原子层面的最终 Treatment 向量
    # 论文中提到 a_fact 是唯一的事实原子，所以我们直接将宏观层面上已经被提纯过的事实残差放入原子模型中
    T_star_atomic = con_atomic_cols + atomic_pure_cols + ['res_T_int_fac_calib_pure']

    # ---------------------------------------------------------
    # 核心回归引擎函数
    # ---------------------------------------------------------
    def run_causal_regression(treatment_cols, resolution_level):
        results = []
        
        # 1. 估计 ATE (全样本)
        for y_col in Y_cols:
            model = sm.OLS(df[y_col], df[treatment_cols]).fit()
            for i, t_col in enumerate(treatment_cols):
                # 清洗组件名称，使其与论文 Table 2 符号完全一致
                clean_name = t_col.replace('res_', '').replace('_calib', '').replace('_pure', '').replace('atomic_', '')
                results.append({
                    'Resolution': resolution_level,
                    'H_level': 'All (ATE)',
                    'Stage': y_col.replace('res_y_', ''),
                    'Component': clean_name,
                    'Coefficient': model.params.iloc[i],
                    'Std_Error': model.bse.iloc[i],
                    'P_Value': model.pvalues.iloc[i],
                    'T_Stat': model.tvalues.iloc[i]
                })
                
        # 2. 估计 CATE (按卷入度 H_level 分层)
        h_levels = sorted(df['H_level'].unique())
        for h in h_levels:
            df_sub = df[df['H_level'] == h]
            for y_col in Y_cols:
                model = sm.OLS(df_sub[y_col], df_sub[treatment_cols]).fit()
                for i, t_col in enumerate(treatment_cols):
                    clean_name = t_col.replace('res_', '').replace('_calib', '').replace('_pure', '').replace('atomic_', '')
                    results.append({
                        'Resolution': resolution_level,
                        'H_level': h,
                        'Stage': y_col.replace('res_y_', ''),
                        'Component': clean_name,
                        'Coefficient': model.params.iloc[i],
                        'Std_Error': model.bse.iloc[i],
                        'P_Value': model.pvalues.iloc[i],
                        'T_Stat': model.tvalues.iloc[i]
                    })
        return results

    # ---------------------------------------------------------
    # STEP 2 & 3: 估计 ATE 和 CATE
    # ---------------------------------------------------------
    logger.info("\n>>> 2. 估计宏观组 (Macro) ATE & CATE ...")
    macro_results = run_causal_regression(T_star_macro, "Macro")
    
    logger.info(">>> 3. 估计原子组 (Atomic) ATE & CATE ...")
    atomic_results = run_causal_regression(T_star_atomic, "Atomic")

    # ---------------------------------------------------------
    # STEP 4: 保存结果 (尾缀 _2)
    # ---------------------------------------------------------
    df_final = pd.DataFrame(macro_results + atomic_results)
    
    # 拆分 ATE 和 CATE 方便后续画图
    df_ate = df_final[df_final['H_level'] == 'All (ATE)']
    df_cate = df_final[df_final['H_level'] != 'All (ATE)']
    
    ate_path = os.path.join(SAVE_DIR, "Final_ATE_Results_2.csv")
    cate_path = os.path.join(SAVE_DIR, "Final_CATE_Results_2.csv")
    
    df_ate.to_csv(ate_path, index=False)
    df_cate.to_csv(cate_path, index=False)

    logger.info("\n" + "="*80)
    logger.info("🎉 [Congratulation!] DCML 核心因果推断流程圆满完成！")
    logger.info(f"   - ATE 结果已保存至: {ate_path}")
    logger.info(f"   - CATE 结果已保存至: {cate_path}")
    
    # 终端预览核心结论 (因果反转)
    purchase_ate = df_ate[df_ate['Stage'] == 'purchase']
    mkt_macro = purchase_ate[purchase_ate['Component'] == 'T_con_mkt']
    pri_atomic = purchase_ate[purchase_ate['Component'] == 'a_pri']
    
    logger.info(f"\n💡 [验证 Hypothesis 4c] 营销显著性在最终【购买阶段】的因果效应:")
    if not mkt_macro.empty:
        logger.info(f"   - 宏观营销 (T_con_mkt) ATE: {mkt_macro['Coefficient'].values[0]:.6f} (P-val: {mkt_macro['P_Value'].values[0]:.4f})")
    if not pri_atomic.empty:
        logger.info(f"   - 原子价格刺激 (a_pri) ATE: {pri_atomic['Coefficient'].values[0]:.6f} (P-val: {pri_atomic['P_Value'].values[0]:.4f})")
    logger.info("="*80)

if __name__ == "__main__":
    main()
    
    #. python /home/xzhe162/wh_workspace/casual/model/DML/dcml_step2_causal_inference.py