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
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 路径配置 (尾缀改为 _5)
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"

# ⚠️ 输入路径：读取 full_3 数据
Y_PATH = os.path.join(BASE_DIR, "Y/y_behavior.parquet")
USER_PATH = os.path.join(BASE_DIR, "Confounder_user/confounder_user_matrix.parquet")
ITEM_PATH = os.path.join(BASE_DIR, "item_feature&Confounder_price/item_text_price_matrix.parquet")
RATING_PATH = os.path.join(BASE_DIR, "rating/user_item_rating.parquet")
TREATMENT_PATH = os.path.join(BASE_DIR, "item_multimodal_scalars/item_multimodal_scalars_full_3.parquet")

# ✅ 输出路径：DCML v5 数据集
SAVE_DIR = os.path.join(BASE_DIR, "build_dataset")
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"dataset_builder_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ==========================================
# 核心函数定义
# ==========================================

def zero_preserved_rank_norm(x, seed=2025):
    """ 建议1: 添加全零列防御的零值保留秩归一化 """
    x_arr = np.array(x, dtype=float)
    x_calibrated = np.zeros_like(x_arr)
    mask_pos = (x_arr > 0)
    num_pos = mask_pos.sum()
    
    if num_pos == 0:  # 🔹 全零列直接返回，防止报错
        return x_calibrated
        
    if num_pos > 0:
        x_pos = x_arr[mask_pos]
        np.random.seed(seed)
        jitter = np.random.uniform(0, 1e-6, size=num_pos)
        ranks = rankdata(x_pos + jitter, method='ordinal')
        x_calibrated[mask_pos] = ranks / float(num_pos)
    return x_calibrated

def compute_semantic_alignment(df_y, df_item, category_col='category'):
    """ 建议3: 封装 T_int_sem (语义对齐度) 计算，支持扩展 """
    # 提取类目
    df_item_copy = df_item.copy()
    df_item_copy[category_col] = df_item_copy['text_W'].str.split('|').str[1].str.replace('类目：', '').str.strip()
    
    # 仅统计产生过点击的行为
    user_cat_pref = pd.merge(df_y[df_y['y_click']==1], df_item_copy[['item_id', category_col]], on='item_id')
    
    # 计算比例
    user_cat_total = user_cat_pref.groupby(['user_id', category_col]).size().reset_index(name='cat_clicks')
    user_total = user_cat_pref.groupby('user_id').size().reset_index(name='user_total_clicks')
    user_cat_ratio = pd.merge(user_cat_total, user_total, on='user_id')
    user_cat_ratio['T_int_sem'] = user_cat_ratio['cat_clicks'] / user_cat_ratio['user_total_clicks']
    
    return df_item_copy[['item_id', category_col]], user_cat_ratio[['user_id', category_col, 'T_int_sem']]

# ==========================================
# 主流程
# ==========================================
def main():
    try:
        logger.info("="*80)
        logger.info(">>> 开始构建 DCML v5 数据集 (全量对齐论文实验设计)")
        logger.info("="*80)

        # ---------------------------------------------------------
        # STEP 1: 载入数据
        # ---------------------------------------------------------
        logger.info(">>> [1/6] 载入各大预处理模块数据...")
        df_y = pd.read_parquet(Y_PATH)
        df_user = pd.read_parquet(USER_PATH)
        df_item = pd.read_parquet(ITEM_PATH)
        df_t = pd.read_parquet(TREATMENT_PATH)
        df_rating = pd.read_parquet(RATING_PATH)

        # ---------------------------------------------------------
        # STEP 2: 构造 M_norm (中介) 和 T_int_sem (语义对齐)
        # ---------------------------------------------------------
        logger.info(">>> [2/6] 构造中介变量 (M_norm) 与语义对齐特征 (T_int_sem)...")
        # 中介变量：评论热度对数 (代理 Subjective Norms)
        m_norm_data = df_rating.groupby('item_id').size().reset_index(name='M_norm_raw')
        m_norm_data['M_norm'] = np.log1p(m_norm_data['M_norm_raw'])

        # 语义对齐特征计算
        df_item_cat, user_cat_ratio = compute_semantic_alignment(df_y, df_item)

        # ---------------------------------------------------------
        # STEP 3: 数据全局 Inner Join
        # ---------------------------------------------------------
        logger.info(">>> [3/6] 执行全局主表合并...")
        master_df = pd.merge(df_y, df_user, on="user_id", how="inner")
        # 拼入含有 category 的 item 表
        master_df = pd.merge(master_df, pd.merge(df_item[['item_id', 'x_pri_log']], df_item_cat, on='item_id'), on="item_id", how="inner")
        master_df = pd.merge(master_df, df_t, on="item_id", how="inner")
        
        # 拼入中介变量与语义对齐变量
        master_df = pd.merge(master_df, m_norm_data[['item_id', 'M_norm']], on='item_id', how='left').fillna({'M_norm': 0})
        master_df = pd.merge(master_df, user_cat_ratio, on=['user_id', 'category'], how='left').fillna({'T_int_sem': 0.01})
        
        # 拼入 T_con_soc (热度) 和 T_con_rat (评分)
        item_social = df_y.groupby('item_id').size().reset_index(name='N_signal')
        item_social['T_con_soc'] = np.log1p(item_social['N_signal'])
        item_rating = df_rating.groupby('item_id')['rating'].mean().reset_index()
        item_rating['T_con_rat'] = (item_rating['rating'] - 1) / 4.0
        
        master_df = pd.merge(master_df, item_social[['item_id', 'T_con_soc']], on='item_id', how='left').fillna({'T_con_soc': 0})
        master_df = pd.merge(master_df, item_rating[['item_id', 'T_con_rat']], on='item_id', how='left').fillna({'T_con_rat': 0.5})

        # ---------------------------------------------------------
        # STEP 4: GMM 计算认知卷入度 (H_level)
        # ---------------------------------------------------------
        logger.info(">>> [4/6] 执行 GMM 计算用户认知卷入度 (H_level)...")
        user_activity = df_y.groupby('user_id').size().reset_index(name='b_u')
        user_activity['b_u_log'] = np.log1p(user_activity['b_u'])
        
        gmm = GaussianMixture(n_components=4, random_state=2025, n_init=3)
        user_activity['H_raw'] = gmm.fit_predict(user_activity[['b_u_log']])
        
        # 排序确保 0=低卷入, 3=高卷入
        order = user_activity.groupby('H_raw')['b_u_log'].mean().sort_values().index
        mapping = {old: new for new, old in enumerate(order)}
        user_activity['H_level'] = user_activity['H_raw'].map(mapping)
        
        master_df = pd.merge(master_df, user_activity[['user_id', 'H_level']], on='user_id', how='inner')

        # ---------------------------------------------------------
        # STEP 5: 变量校准与诊断 (Zero-Preserved Rank Norm)
        # ---------------------------------------------------------
        logger.info(">>> [5/6] 提取 LMM 特征执行校准，并输出诊断日志...")
        # 自动识别 df_t 里所有的 LMM 处理变量（宏观 + 原子）
        lmm_vars = [col for col in df_t.columns if col != 'item_id']
        calibrated_vars = []
        
        for v in lmm_vars:
            calib_name = f"{v}_calib"
            master_df[calib_name] = zero_preserved_rank_norm(master_df[v])
            calibrated_vars.append(calib_name)
            
            # 建议2: 实时诊断输出
            zero_pct = (master_df[calib_name] == 0).mean() * 100
            unique_vals = master_df[calib_name].nunique()
            logger.info(f"   ✓ {calib_name:<30} | 零值比例: {zero_pct:>6.2f}% | 唯一值数: {unique_vals}")

        # 混淆变量 X_pos 编码
        pos_map = master_df['position'].value_counts(normalize=True).to_dict()
        master_df['x_pos_freq'] = master_df['position'].map(pos_map)

        # ---------------------------------------------------------
        # STEP 6: 严谨的三集切分与输出
        # ---------------------------------------------------------
        logger.info(">>> [6/6] 整理论文矩阵并执行三集切分...")
        
        # 汇集所有 Treatment
        all_treatments = calibrated_vars + ['T_int_sem', 'T_con_soc', 'T_con_rat']
        user_conf_cols = [c for c in df_user.columns if c != 'user_id']

        final_cols = (
            ['user_id', 'item_id', 'timestamp'] + 
            ['y_click', 'y_cart', 'y_purchase'] + 
            all_treatments + 
            ['M_norm', 'H_level', 'x_pri_log', 'x_pos_freq'] + 
            user_conf_cols
        )
        master_df = master_df[final_cols]
        
        # 按照论文 4.1 划分 Set A(20%), Set B(40%), Set C(40%)
        set_A, remain = train_test_split(master_df, test_size=0.8, random_state=2025, stratify=master_df['H_level'])
        set_B, set_C = train_test_split(remain, test_size=0.5, random_state=42, stratify=remain['H_level'])

        # 保存
        for name, data in zip(['Table_5', 'A_5', 'B_5', 'C_5'], [master_df, set_A, set_B, set_C]):
            data.to_parquet(os.path.join(SAVE_DIR, f"DCML_{name}.parquet"), index=False)

        logger.info("="*80)
        logger.info(f"🎉 DCML v5 构建成功！")
        logger.info(f"   - 包含的因果处理变量 ({len(all_treatments)}个): {all_treatments}")
        logger.info(f"   - Set B (Nuisance) 样本量: {len(set_B)}")
        logger.info(f"   - Set C (Inference) 样本量: {len(set_C)}")
        logger.info("="*80)

    except Exception as e:
        logger.exception(f"❌ 运行失败: {e}")

if __name__ == "__main__":
    main()
    
    #.  python /home/xzhe162/wh_workspace/casual/model/MCRE/dataset_builder.py
    
    