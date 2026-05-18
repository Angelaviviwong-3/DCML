#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配 DCML 框架的亚马逊数据集行为漏斗 (Y) 构建脚本
包含：解析 JSONL、构建漏斗负采样、生成位置代理变量。
"""

import os
import json
import time
import logging
import random
import pandas as pd
import numpy as np

# ===============================
# 1. 路径与环境配置
# ===============================
DATASETS = ["amazon_appliances", "amazon_beauty"]

BASE_RAW_DIR = "/home/xzhe162/wh_workspace/casual/raw_data"
BASE_OUT_DIR = "/home/xzhe162/wh_workspace/casual/processed_data"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置
log_file = os.path.join(LOG_DIR, "process_amazon_y_funnel.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# 2. 漏斗负采样配置 (为 DML 创造方差)
# 比例设定 (基于常见电商转化漏斗)
# ===============================
# 假设 1 个 purchase 背后对应：
# - 2 个 仅加购 (Cart=1, Purchase=0)
# - 7 个 仅点击 (Click=1, Cart=0)
# - 20 个 仅曝光 (Click=0)
RATIO_CART_ABANDON = 2
RATIO_CLICK_BOUNCE = 7
RATIO_IGNORE = 20

def process_dataset(dataset_name):
    logger.info(f"========== 开始处理数据集: {dataset_name} ==========")
    start_time = time.time()
    
    # 路径映射
    # 兼容命名不一致的情况：Appliances vs All_Beauty
    if dataset_name == "amazon_appliances":
        meta_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "meta_Appliances.jsonl")
        review_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "reviews_Appliances.jsonl")
    else:
        meta_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "meta_All_Beauty.jsonl")
        review_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "All_Beauty.jsonl")

    output_dir = os.path.join(BASE_OUT_DIR, dataset_name, "Y")
    os.makedirs(output_dir, exist_ok=True)
    
    # ===============================
    # Step 1: 解析 Meta 数据构建 Item 池和位置代理
    # ===============================
    logger.info("Step 1: 读取 Meta 文件获取 item 池及热度...")
    item_pool = []
    item_popularity = {}
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                asin = data.get("parent_asin")
                if not asin: continue
                
                # 用 rating_number 作为热度指标，用于模拟 Position Bias
                pop = data.get("rating_number", 0)
                if pop is None: pop = 0
                
                item_popularity[asin] = pop
                item_pool.append(asin)
            except json.JSONDecodeError:
                continue
                
    item_pool = list(set(item_pool)) # 去重
    logger.info(f"共提取了 {len(item_pool)} 个独立商品。")

    # 根据热度生成模拟的 Position (分为 Top, Mid, Tail 模拟排位)
    pop_series = pd.Series(item_popularity)
    pop_quantiles = pop_series.quantile([0.33, 0.66])
    
    def get_proxy_position(asin):
        pop = item_popularity.get(asin, 0)
        if pop > pop_quantiles[0.66]: return "Page_1_Top"
        elif pop > pop_quantiles[0.33]: return "Page_2_Mid"
        else: return "Page_3_Tail"

    # ===============================
    # Step 2: 解析 Reviews 提取真实的转化数据
    # ===============================
    logger.info("Step 2: 读取 Review 文件提取正样本...")
    positives = []
    
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                user_id = data.get("user_id")
                item_id = data.get("parent_asin")
                timestamp = data.get("timestamp")
                
                if user_id and item_id:
                    positives.append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "y_click": 1,
                        "y_cart": 1,
                        "y_purchase": 1,
                        "position": get_proxy_position(item_id)
                    })
            except json.JSONDecodeError:
                continue
                
    pos_df = pd.DataFrame(positives)
    pos_df = pos_df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    logger.info(f"提取了 {len(pos_df)} 条真实的购买转化记录。")

    # ===============================
    # Step 3: 漏斗负采样 (核心修改)
    # ===============================
    logger.info("Step 3: 进行漏斗负采样生成伪行为记录 (保障 DML 方差)...")
    
    negatives = []
    # 为了加速，我们将循环操作转为批量化思维
    # 1. 抽取所有需要的随机 item
    total_negs_needed = len(pos_df) * (RATIO_CART_ABANDON + RATIO_CLICK_BOUNCE + RATIO_IGNORE)
    random_items = np.random.choice(item_pool, size=total_negs_needed, replace=True)
    
    idx = 0
    # 模拟用户在购买正样本前，浏览过其他商品
    for _, row in pos_df.iterrows():
        uid = row['user_id']
        ts = row['timestamp']
        
        # 1. 仅加购未买 (Cart Abandonment)
        for _ in range(RATIO_CART_ABANDON):
            item = random_items[idx]
            idx += 1
            negatives.append({"user_id": uid, "item_id": item, "timestamp": ts - 1000, 
                              "y_click": 1, "y_cart": 1, "y_purchase": 0, "position": get_proxy_position(item)})
            
        # 2. 仅点击未加购 (Click Bounce)
        for _ in range(RATIO_CLICK_BOUNCE):
            item = random_items[idx]
            idx += 1
            negatives.append({"user_id": uid, "item_id": item, "timestamp": ts - 5000, 
                              "y_click": 1, "y_cart": 0, "y_purchase": 0, "position": get_proxy_position(item)})
            
        # 3. 仅曝光未点击 (Ignore)
        for _ in range(RATIO_IGNORE):
            item = random_items[idx]
            idx += 1
            negatives.append({"user_id": uid, "item_id": item, "timestamp": ts - 10000, 
                              "y_click": 0, "y_cart": 0, "y_purchase": 0, "position": get_proxy_position(item)})

    neg_df = pd.DataFrame(negatives)
    
    # ===============================
    # Step 4: 合并与清洗
    # ===============================
    logger.info("Step 4: 合并数据与最终清洗...")
    y_df = pd.concat([pos_df, neg_df], ignore_index=True)
    
    # 确保漏斗逻辑严格性： Purchase=1 必须 Cart=1，Cart=1 必须 Click=1
    mask_purchase = (y_df['y_purchase'] == 1)
    y_df.loc[mask_purchase, ['y_cart', 'y_click']] = 1
    
    mask_cart = (y_df['y_cart'] == 1)
    y_df.loc[mask_cart, 'y_click'] = 1

    # 排序
    y_df = y_df.sort_values(by=["user_id", "timestamp"])
    
    # 去除极少数采样碰撞导致的重复 (保留转化最深的那个，即最大值)
    y_df = y_df.groupby(["user_id", "item_id", "timestamp"], as_index=False).max()

    # ===============================
    # Step 5: 保存文件
    # ===============================
    logger.info("Step 5: 写入本地文件...")
    parquet_path = os.path.join(output_dir, "y_behavior.parquet")
    csv_path = os.path.join(output_dir, "y_behavior.csv")

    y_df.to_parquet(parquet_path, index=False)
    y_df.to_csv(csv_path, index=False)

    logger.info(f"保存成功: {parquet_path}")
    
    # ===============================
    # 数据统计摘要
    # ===============================
    logger.info("--- Data Summary ---")
    logger.info(f"总记录数: {len(y_df)}")
    logger.info(f"总点击 (y_click=1): {y_df['y_click'].sum()}")
    logger.info(f"总加购 (y_cart=1): {y_df['y_cart'].sum()}")
    logger.info(f"总购买 (y_purchase=1): {y_df['y_purchase'].sum()}")
    
    end_time = time.time()
    logger.info(f"[{dataset_name}] 处理耗时: {round(end_time - start_time, 2)} 秒")
    logger.info("========================================================\n")


if __name__ == "__main__":
    for ds in DATASETS:
        try:
            process_dataset(ds)
        except Exception as e:
            logger.exception(f"处理数据集 {ds} 时发生异常!")
            
            
            
          # python /home/xzhe162/wh_workspace/casual/data_preprocessing/amazon_appliances_preprocessing/build_y_funnel_amazon.py