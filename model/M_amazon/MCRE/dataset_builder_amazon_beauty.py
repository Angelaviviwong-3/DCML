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
import argparse
import time
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 参数解析与路径配置
# ==========================================
parser = argparse.ArgumentParser(description="Amazon DCML Dataset Builder (Beauty V1)")
# ✅ 默认值修改为 amazon_beauty
parser.add_argument("-d", "--dataset", default="amazon_beauty", choices=["amazon_appliances", "amazon_beauty"])
args = parser.parse_args()

DATASET_NAME = args.dataset

BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}"

# ⚠️ 输入路径：分别指向各模块最后生成的文件
Y_PATH = os.path.join(BASE_DIR, "Y/y_behavior.parquet")
USER_PATH = os.path.join(BASE_DIR, "Confounder_user/confounder_user_matrix.parquet")
ITEM_PATH = os.path.join(BASE_DIR, "item_feature&Confounder_price/item_text_price_matrix.parquet")
RATING_PATH = os.path.join(BASE_DIR, "rating/user_item_rating.parquet")

# ⚠️ 使用上一阶段最终合并好的、已经过校准的特征
TREATMENT_PATH = os.path.join(BASE_DIR, f"item_multimodal_scalars/{DATASET_NAME}_item_multimodal_scalars_merged_full.parquet")

# ✅ 输出路径：DCML v2 数据集
SAVE_DIR = os.path.join(BASE_DIR, "build_dataset")
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"dataset_builder_{DATASET_NAME}_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', 
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, log_filename)), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ==========================================
# 核心函数定义
# ==========================================
def compute_semantic_alignment(df_y, df_item, category_col='cat_name'):
    """ 封装 T_int_sem (语义对齐度) 计算 """
    df_item_copy = df_item.copy()
    
    if category_col not in df_item_copy.columns:
        df_item_copy[category_col] = df_item_copy['text_W'].str.extract(r'类目：(.*?)\s*\|')[0].fillna('Unknown')
    
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
        logger.info(f">>> 开始构建亚马逊 {DATASET_NAME} DCML 数据集 (v1)")
        logger.info("="*80)

        # ✅ 加入全局进度条
        pbar = tqdm(total=6, desc=f"构建 {DATASET_NAME} 矩阵")

        # ---------------------------------------------------------
        # STEP 1: 载入数据
        # ---------------------------------------------------------
        logger.info(">>> [1/6] 载入各大预处理模块数据...")
        df_y = pd.read_parquet(Y_PATH)
        df_user = pd.read_parquet(USER_PATH)
        df_item = pd.read_parquet(ITEM_PATH)
        df_t = pd.read_parquet(TREATMENT_PATH)
        df_rating = pd.read_parquet(RATING_PATH)

        # 确保 id 的数据类型一致
        df_y['item_id'] = df_y['item_id'].astype(str)
        df_y['user_id'] = df_y['user_id'].astype(str)
        df_user['user_id'] = df_user['user_id'].astype(str)
        df_item['item_id'] = df_item['item_id'].astype(str)
        df_t['item_id'] = df_t['item_id'].astype(str)
        df_rating['item_id'] = df_rating['item_id'].astype(str)
        
        pbar.update(1)

        # ---------------------------------------------------------
        # STEP 2: 构造 M_norm (中介) 和 T_int_sem (语义对齐)
        # ---------------------------------------------------------
        logger.info(">>> [2/6] 构造中介变量 (M_norm) 与语义对齐特征 (T_int_sem)...")
        # 中介变量：评论热度对数 (代理 Subjective Norms)
        m_norm_data = df_rating.groupby('item_id').size().reset_index(name='M_norm_raw')
        m_norm_data['M_norm'] = np.log1p(m_norm_data['M_norm_raw'])

        # 语义对齐特征计算
        df_item_cat, user_cat_ratio = compute_semantic_alignment(df_y, df_item)
        
        pbar.update(1)

        # ---------------------------------------------------------
        # STEP 3: 数据全局 Inner Join
        # ---------------------------------------------------------
        logger.info(">>> [3/6] 执行全局主表合并...")
        master_df = pd.merge(df_y, df_user, on="user_id", how="inner")
        master_df = pd.merge(master_df, pd.merge(df_item[['item_id', 'x_pri_log']], df_item_cat, on='item_id'), on="item_id", how="inner")
        
        # 只取 df_t 中 _calibrated 结尾的列作为多模态特征，保持极度干净
        lmm_cols = [c for c in df_t.columns if c.endswith('_calibrated')]
        master_df = pd.merge(master_df, df_t[['item_id'] + lmm_cols], on="item_id", how="inner")
        
        # 拼入中介变量与语义对齐变量
        master_df = pd.merge(master_df, m_norm_data[['item_id', 'M_norm']], on='item_id', how='left').fillna({'M_norm': 0})
        master_df = pd.merge(master_df, user_cat_ratio, on=['user_id', 'cat_name'], how='left').fillna({'T_int_sem': 0.01})
        
        # 拼入 T_con_soc (热度) 和 T_con_rat (评分)
        item_social = df_y.groupby('item_id').size().reset_index(name='N_signal')
        item_social['T_con_soc'] = np.log1p(item_social['N_signal'])
        item_rating = df_rating.groupby('item_id')['rating'].mean().reset_index()
        item_rating['T_con_rat'] = (item_rating['rating'] - 1) / 4.0
        
        master_df = pd.merge(master_df, item_social[['item_id', 'T_con_soc']], on='item_id', how='left').fillna({'T_con_soc': 0})
        master_df = pd.merge(master_df, item_rating[['item_id', 'T_con_rat']], on='item_id', how='left').fillna({'T_con_rat': 0.5})

        pbar.update(1)

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

        pbar.update(1)

        # ---------------------------------------------------------
        # STEP 5: 混淆变量编码与提取
        # ---------------------------------------------------------
        logger.info(">>> [5/6] 混淆变量位置编码...")
        pos_map = master_df['position'].value_counts(normalize=True).to_dict()
        master_df['x_pos_freq'] = master_df['position'].map(pos_map)

        pbar.update(1)

        # ---------------------------------------------------------
        # STEP 6: 严谨的三集切分与输出
        # ---------------------------------------------------------
        logger.info(">>> [6/6] 整理论文矩阵并执行三集切分...")
        
        # 核心：确保恰好包含 12 个因果处理变量！
        all_treatments = lmm_cols + ['T_int_sem', 'T_con_soc', 'T_con_rat']
        
        user_conf_cols = [c for c in df_user.columns if c != 'user_id']

        final_cols = (
            ['user_id', 'item_id', 'timestamp'] + 
            ['y_click', 'y_cart', 'y_purchase'] + 
            all_treatments + 
            ['M_norm', 'H_level', 'x_pri_log', 'x_pos_freq'] + 
            user_conf_cols
        )
        
        # 筛选最终列，防止出现奇怪的多余列
        master_df = master_df[final_cols]
        
        # 按照论文 4.1 划分 Set A(20%), Set B(40%), Set C(40%)
        set_A, remain = train_test_split(master_df, test_size=0.8, random_state=2025, stratify=master_df['H_level'])
        set_B, set_C = train_test_split(remain, test_size=0.5, random_state=42, stratify=remain['H_level'])

        # 保存
        for name, data in zip(['Table_Master', 'Set_A', 'Set_B', 'Set_C'], [master_df, set_A, set_B, set_C]):
            out_path = os.path.join(SAVE_DIR, f"{DATASET_NAME}_DCML_{name}_2.parquet")
            data.to_parquet(out_path, index=False)
            logger.info(f"   - 已保存 {name} -> {out_path}")

        pbar.update(1)
        pbar.close()

        logger.info("="*80)
        logger.info(f"🎉 亚马逊数据集 ({DATASET_NAME}) DCML v1 构建成功！")
        logger.info(f"   - 包含的因果处理变量 ({len(all_treatments)}个): {all_treatments}")
        logger.info(f"   - 主表全量数据: {len(master_df):,} 条")
        logger.info(f"   - Set A (Representation) 样本量: {len(set_A):,}")
        logger.info(f"   - Set B (Nuisance Model) 样本量: {len(set_B):,}")
        logger.info(f"   - Set C (Causal Inference) 样本量: {len(set_C):,}")
        logger.info("="*80)

    except Exception as e:
        logger.exception(f"❌ 运行失败: {e}")

if __name__ == "__main__":
    main()
    
    
 #.  python /home/xzhe162/wh_workspace/casual/model/M_amazon/MCRE/dataset_builder_amazon_beauty.py