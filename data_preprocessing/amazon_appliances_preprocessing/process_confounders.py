#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配 DCML 框架的亚马逊数据集 Confounder & Item Text 构建脚本
包含：
1. User Confounder (X_user): 历史活跃度、平均打分 (Z-Score)
2. Item Feature (W & X_pri): LMM自然语言拼接、价格清洗与对数化、提取社会评价信号
"""

import os
import json
import re
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASETS = ["amazon_appliances", "amazon_beauty"]

BASE_RAW_DIR = "/home/xzhe162/wh_workspace/casual/raw_data"
BASE_OUT_DIR = "/home/xzhe162/wh_workspace/casual/processed_data"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
log_filename = f"process_amazon_confounders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_price(price_val):
    """鲁棒的亚马逊价格解析函数"""
    if price_val is None:
        return np.nan
    if isinstance(price_val, (int, float)):
        return float(price_val)
    # 处理字符串，例如 "$19.99", "£19.99 - $25.00" (取第一个)
    match = re.search(r'[\d\.]+', str(price_val))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan

def process_dataset(dataset_name):
    logger.info(f"========== 开始处理数据集: {dataset_name} ==========")
    
    # 路径映射
    if dataset_name == "amazon_appliances":
        meta_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "meta_Appliances.jsonl")
        review_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "reviews_Appliances.jsonl")
    else:
        meta_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "meta_All_Beauty.jsonl")
        review_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "All_Beauty.jsonl")

    out_user_dir = os.path.join(BASE_OUT_DIR, dataset_name, "Confounder_user")
    out_item_dir = os.path.join(BASE_OUT_DIR, dataset_name, "item_feature&Confounder_price")
    os.makedirs(out_user_dir, exist_ok=True)
    os.makedirs(out_item_dir, exist_ok=True)

    # ==========================================
    # PART 1: 构建 User Confounder (X_user)
    # ==========================================
    logger.info(">>> [PART 1] 开始构建 User Confounder...")
    user_records = []
    
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                user_id = data.get("user_id")
                rating = data.get("rating")
                if user_id and rating is not None:
                    user_records.append({"user_id": user_id, "rating": float(rating)})
            except json.JSONDecodeError:
                continue

    df_users = pd.DataFrame(user_records)
    
    # 按照论文设计：提取 historical engagement scales
    df_user_agg = df_users.groupby("user_id").agg(
        user_review_count=('rating', 'count'),  # 活跃度
        user_avg_rating=('rating', 'mean')      # 苛刻程度/偏好基线
    ).reset_index()

    # Z-Score 标准化
    scaler = StandardScaler()
    df_user_agg[['user_review_count_scaled', 'user_avg_rating_scaled']] = scaler.fit_transform(
        df_user_agg[['user_review_count', 'user_avg_rating']]
    )
    
    # 提取最终需要的用户特征列
    x_user_final = df_user_agg[['user_id', 'user_review_count_scaled', 'user_avg_rating_scaled']]
    
    parquet_path_user = os.path.join(out_user_dir, "confounder_user_matrix.parquet")
    csv_path_user = os.path.join(out_user_dir, "confounder_user_matrix.csv")
    x_user_final.to_parquet(parquet_path_user, index=False)
    x_user_final.to_csv(csv_path_user, index=False)
    logger.info(f"User Confounder 完成! 独立用户数: {len(x_user_final)}")

    # ==========================================
    # PART 2: 构建 Item Feature & Price Confounder
    # ==========================================
    logger.info(">>> [PART 2] 开始构建 Item Feature (W) 与 Price Confounder (X_pri)...")
    item_records = []
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取 Meta JSONL"):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                asin = data.get("parent_asin")
                if not asin: continue
                
                # 提取文本组件
                brand = str(data.get("store") or "未知品牌").strip()
                category = str(data.get("main_category") or "未知类目").strip()
                title = str(data.get("title") or "").strip()
                
                # 特征列表截断，防止 LMM context 过长
                features_list = data.get("features", [])
                features_str = " ".join(features_list)[:300] if features_list else "无"
                
                # 拼接给大模型的自然语言 W
                text_W = f"品牌：{brand} | 类目：{category} | 标题：{title} | 商品特征：{features_str}"
                
                # 价格与社会信号
                raw_price = data.get("price")
                rating_number = data.get("rating_number", 0)
                avg_rating = data.get("average_rating", 0.0)
                
                item_records.append({
                    "item_id": asin,
                    "cat_name": category,
                    "text_W": text_W,
                    "raw_price": raw_price,
                    "rating_number": rating_number if rating_number else 0,
                    "average_rating": avg_rating if avg_rating else 0.0
                })
            except json.JSONDecodeError:
                continue

    df_items = pd.DataFrame(item_records)
    # 去重
    df_items = df_items.drop_duplicates(subset=['item_id'])

    # 1. 价格清洗与填补
    df_items['price'] = df_items['raw_price'].apply(parse_price)
    
    # 策略：同类目中位数填补 -> 全局中位数填补
    cat_median = df_items.groupby('cat_name')['price'].transform('median')
    df_items['price'] = df_items['price'].fillna(cat_median)
    df_items['price'] = df_items['price'].fillna(df_items['price'].median())
    
    # 价格对数变换（因果推断必须）
    df_items['x_pri_log'] = np.log1p(df_items['price'])

    # 2. 社会从众信号变换 (论文中 T_con,soc = log(N_signal + 1))
    df_items['social_signal_log'] = np.log1p(df_items['rating_number'])

    # 3. 提取最终矩阵
    final_item_cols = ['item_id', 'text_W', 'x_pri_log', 'social_signal_log', 'average_rating']
    df_item_output = df_items[final_item_cols]

    parquet_path_item = os.path.join(out_item_dir, "item_text_price_matrix.parquet")
    csv_path_item = os.path.join(out_item_dir, "item_text_price_matrix.csv")
    df_item_output.to_parquet(parquet_path_item, index=False)
    df_item_output.to_csv(csv_path_item, index=False, encoding='utf-8-sig')

    logger.info(f"Item Feature 处理完成! 独立商品数: {len(df_item_output)}")
    logger.info(f"文本 W 示例: {df_item_output['text_W'].iloc[0][:150]}...")
    logger.info("========================================================\n")


if __name__ == "__main__":
    for ds in DATASETS:
        try:
            process_dataset(ds)
        except Exception as e:
            logger.exception(f"处理数据集 {ds} 时发生异常: {e}")
            
#. python /home/xzhe162/wh_workspace/casual/data_preprocessing/amazon_appliances_preprocessing/process_confounders.py