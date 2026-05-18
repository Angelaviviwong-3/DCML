#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配 DCML 框架的亚马逊评论与评分 (Reviews & Rating) 构建脚本
包含：
1. 提取启发式评分信号 (Rating) 用于 Path 2 (System 1)
2. 深度清洗并提取分析型文本 (Pure Reviews) 用于 Path 1 (System 2)
3. 统一时间戳格式为 YYYYMMDD
"""

import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# ==========================================
# 0. 路径与环境配置
# ==========================================
DATASETS = ["amazon_appliances", "amazon_beauty"]

BASE_RAW_DIR = "/home/xzhe162/wh_workspace/casual/raw_data"
BASE_OUT_DIR = "/home/xzhe162/wh_workspace/casual/processed_data"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
log_filename = f"process_amazon_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def convert_timestamp(ts):
    """
    转换亚马逊时间戳为 YYYYMMDD
    亚马逊的 timestamp 通常是 13 位毫秒级整数
    """
    if ts is None:
        return "19700101"
    try:
        # 如果是 13 位时间戳（毫秒）转换为秒
        if isinstance(ts, int) and ts > 1e11:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts).strftime('%Y%m%d')
    except Exception:
        return "19700101"

def clean_review_text(text):
    """文本深度清洗，判定是否为有效评论"""
    if not text or not isinstance(text, str):
        return False, ""
    
    text = text.strip()
    
    # 无效特征匹配 (亚马逊常见的无效词)
    garbage_texts = [
        "none", "na", "n/a", "good", "great", "ok", "okay", "nice",
        "love it", "works great", "five stars", "five star", "as expected"
    ]
    
    # 规则 1：长度过短 (少于10个字符难以支持 System 2 的深度分析)
    if len(text) < 10:
        return False, text
        
    # 规则 2：纯废话过滤
    if text.lower() in garbage_texts:
        return False, text
        
    return True, text

def process_dataset(dataset_name):
    logger.info(f"========== 开始处理数据集: {dataset_name} 的评论数据 ==========")
    
    # 路径映射
    if dataset_name == "amazon_appliances":
        review_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "reviews_Appliances.jsonl")
    else:
        review_path = os.path.join(BASE_RAW_DIR, f"{dataset_name}_raw", "All_Beauty.jsonl")

    # 输出路径设计
    out_rating_dir = os.path.join(BASE_OUT_DIR, dataset_name, "rating")
    out_reviews_dir = os.path.join(BASE_OUT_DIR, dataset_name, "pure_reviews")
    os.makedirs(out_rating_dir, exist_ok=True)
    os.makedirs(out_reviews_dir, exist_ok=True)

    ratings_list = []
    pure_reviews_list = []
    
    logger.info(">>> 正在逐行解析并清洗 JSONL 评论文件...")
    
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"解析 {dataset_name}"):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                user_id = data.get("user_id")
                item_id = data.get("parent_asin")
                rating = data.get("rating")
                raw_text = data.get("text", "")
                ts = data.get("timestamp")
                
                # 提取图片/视频标记
                has_images = len(data.get("images", [])) > 0
                
                if not user_id or not item_id or rating is None:
                    continue
                
                # 统一时间戳
                fmt_date = convert_timestamp(ts)
                
                # ------------------------------------------
                # 构建用于 Path 2 的 Rating 记录
                # ------------------------------------------
                ratings_list.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": float(rating),
                    "timestamp": fmt_date
                })
                
                # ------------------------------------------
                # 构建用于 Path 1 的 Pure Review 记录
                # ------------------------------------------
                is_valid, cleaned_text = clean_review_text(raw_text)
                if is_valid:
                    pure_reviews_list.append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "review_text": cleaned_text,
                        "is_image_review": int(has_images),
                        "timestamp": fmt_date
                    })

            except json.JSONDecodeError:
                continue

    # 转为 DataFrame
    logger.info(">>> 转换为 DataFrame 并去重...")
    df_rating = pd.DataFrame(ratings_list)
    df_reviews = pd.DataFrame(pure_reviews_list)
    
    # 数据去重（同一用户对同一商品在同一天只保留一条）
    if not df_rating.empty:
        df_rating = df_rating.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])
    if not df_reviews.empty:
        df_reviews = df_reviews.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])

    # 保存数据
    logger.info(">>> 正在写入本地文件...")
    # Rating
    df_rating.to_parquet(os.path.join(out_rating_dir, "user_item_rating.parquet"), index=False)
    df_rating.to_csv(os.path.join(out_rating_dir, "user_item_rating.csv"), index=False)
    
    # Pure Reviews
    df_reviews.to_parquet(os.path.join(out_reviews_dir, "user_item_pure_reviews.parquet"), index=False)
    df_reviews.to_csv(os.path.join(out_reviews_dir, "user_item_pure_reviews.csv"), index=False, encoding='utf-8-sig')

    logger.info("=========================================")
    logger.info(f"🎉 评论与评分预处理完成！ [{dataset_name}]")
    logger.info(f"   - 提取的星级评分记录 (Path 2): {len(df_rating)}")
    logger.info(f"   - 提取的有效深度文本记录 (Path 1): {len(df_reviews)}")
    logger.info("=========================================\n")


if __name__ == "__main__":
    for ds in DATASETS:
        try:
            process_dataset(ds)
        except Exception as e:
            logger.exception(f"处理数据集 {ds} 时发生异常: {e}")


#. python /home/xzhe162/wh_workspace/casual/data_preprocessing/amazon_appliances_preprocessing/process_item_reviews_amazon.py