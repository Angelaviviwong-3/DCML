#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配亚马逊 Amazon Review Data 的多线程图片下载器
针对 meta_*.jsonl 文件，提取每个 ASIN 的高清主图并下载。
支持自定义数据集名称，并规范化了日志和存储路径。
"""

import os
import json
import time
import logging
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from tqdm import tqdm

# ----------------------------- 基础配置 -----------------------------
DATASET_NAME = "amazon_appliances"  # 明确指明当前处理的数据集名称

DEFAULT_WORKERS = 16           
DEFAULT_RETRIES = 3            
DEFAULT_TIMEOUT = 30           
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
PROXIES = {"http": None, "https": None}

# ----------------------------- 路径配置 -----------------------------
DEFAULT_JSONL = f"/home/xzhe162/wh_workspace/casual/raw_data/{DATASET_NAME}_raw/meta_Appliances.jsonl"
DEFAULT_OUTPUT = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}/pictures"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"download_{DATASET_NAME}_images.log")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

stats_lock = Lock()
stats = {"success": 0, "failed": 0, "skipped": 0, "no_image": 0}

def get_file_extension(url, default=".jpg"):
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        ext = default
    return ext

def download_one(task, output_dir, retries=DEFAULT_RETRIES):
    """下载单张图片"""
    asin = task['asin']
    url = task['url']

    if not url:
        return (asin, "no_image")

    ext = get_file_extension(url)
    filename = f"{asin}_MAIN{ext}"
    filepath = os.path.join(output_dir, filename)

    # 断点续传判断
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return (asin, "skipped")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return (asin, "success")
        except Exception as e:
            if attempt == retries:
                logger.debug(f"[{DATASET_NAME}] 下载失败（重试{retries}次）: {asin} | {url} | 错误: {e}")
                return (asin, "failed")
            else:
                time.sleep(1)

def extract_best_image(images_list):
    """
    从亚马逊的 images 列表中提取最佳的主图 URL
    优先级: MAIN 变体的 hi_res -> MAIN 变体的 large -> 列表第一张的 hi_res -> 列表第一张的 large
    """
    if not images_list:
        return None
        
    for img in images_list:
        if img.get("variant") == "MAIN":
            return img.get("hi_res") or img.get("large")
            
    # 如果没有标明 MAIN，退化选取第一张
    first_img = images_list[0]
    return first_img.get("hi_res") or first_img.get("large")

def download_amazon_images(jsonl_path, output_dir, dataset_name, workers=DEFAULT_WORKERS):
    logger.info(f"========== 开始处理数据集: {dataset_name} ==========")
    logger.info(f"读取路径: {jsonl_path}")
    logger.info(f"存储路径: {output_dir}")
    logger.info(f"日志路径: {LOG_FILE}")
    
    if not os.path.exists(jsonl_path):
        logger.error(f"文件不存在: {jsonl_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. 逐行读取文件，构建下载任务列表 (节省内存)
    tasks = []
    logger.info(f"正在扫描 {dataset_name} 的 jsonl 文件提取图片 URL...")
    
    # 记录已经提取过的 parent_asin，防止同一个变体重复下载
    seen_asins = set() 
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                asin = data.get("parent_asin") # 推荐使用 parent_asin 聚合变体
                images = data.get("images", [])
                
                if asin and asin not in seen_asins:
                    best_url = extract_best_image(images)
                    if best_url:
                        tasks.append({"asin": asin, "url": best_url})
                        seen_asins.add(asin)
                    else:
                        stats["no_image"] += 1
            except json.JSONDecodeError:
                continue

    total = len(tasks)
    logger.info(f"扫描完毕！{dataset_name} 找到 {total} 个有效商品的图片，准备启动 {workers} 个线程下载...")

    # 2. 启动线程池下载
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, task, output_dir): task for task in tasks}

        # 进度条增加数据集前缀
        with tqdm(total=total, desc=f"[{dataset_name}] 下载进度", unit="张") as pbar:
            for future in as_completed(futures):
                asin, status = future.result()
                
                with stats_lock:
                    stats[status] += 1
                
                pbar.update(1)
                pbar.set_postfix(success=stats["success"], skipped=stats["skipped"], failed=stats["failed"])

    logger.info(f"[{dataset_name}] 下载完成！成功: {stats['success']}, 跳过(已存在): {stats['skipped']}, 失败: {stats['failed']}, 无图商品: {stats['no_image']}")
    logger.info("========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多线程批量下载亚马逊商品主图")
    
    parser.add_argument("jsonl", nargs="?", default=DEFAULT_JSONL, help="Amazon meta JSONL 文件路径")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="图片输出目录")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    parser.add_argument("-d", "--dataset", default=DATASET_NAME, help="数据集名称标识")
    args = parser.parse_args()
    
    download_amazon_images(args.jsonl, args.output, args.dataset, args.workers)
    
    # python /home/xzhe162/wh_workspace/casual/data_preprocessing/amazon_appliances_preprocessing/download_images_amazon_app.py