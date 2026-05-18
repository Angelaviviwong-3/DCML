#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程批量下载图片（支持断点续传、重试、日志）
用法1 (使用默认路径): python download_images.py
用法2 (指定文件路径): python download_images.py /你的路径/data.xlsx -o /你的输出目录/
"""

import os
import time
import logging
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------------- 配置 -----------------------------
DEFAULT_WORKERS = 16           # 并发线程数
DEFAULT_RETRIES = 3            # 失败重试次数
DEFAULT_TIMEOUT = 30           # 请求超时（秒）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
PROXIES = {"http": None, "https": None}   # 禁用代理

# 【重要】请确保这里的列名与你 Excel 表格的第一行表头完全一致！
COL_ID = "item_id"          # 商品ID列名
COL_LOC = "pic_loc"    # 图片位置列名
COL_URL = "full_picture_url"      # 图片URL列名

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("download_images.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 线程锁与全局统计
stats_lock = Lock()
stats = {"success": 0, "failed": 0, "skipped": 0}


def get_file_extension(url, default=".jpg"):
    """从 URL 提取扩展名，若没有则返回默认"""
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        ext = default
    return ext


def sanitize_filename(filename):
    """移除文件名中的非法字符"""
    invalid_chars = r'\/:*?"<>|'
    for ch in invalid_chars:
        filename = filename.replace(ch, '_')
    return filename


def download_one(row, output_dir, retries=DEFAULT_RETRIES):
    """下载单张图片，返回 (item_id, pic_loc, 状态字符串)"""
    item_id = row[COL_ID]
    pic_loc = row[COL_LOC]
    url = str(row[COL_URL]).strip()

    # 跳过空 URL
    if pd.isna(url) or url == 'nan' or not url:
        logger.warning(f"空URL: {item_id}_{pic_loc}")
        return (item_id, pic_loc, "failed")

    # 确定扩展名和保存路径
    ext = get_file_extension(url)
    filename = f"{item_id}_{pic_loc}{ext}"
    filename = sanitize_filename(filename)
    filepath = os.path.join(output_dir, filename)

    # 断点续传：如果文件已存在且大小 > 0，跳过
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logger.debug(f"跳过已存在: {filepath}")
        return (item_id, pic_loc, "skipped")

    # 重试下载
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            # 保存文件
            with open(filepath, "wb") as f:
                f.write(resp.content)
            logger.debug(f"下载成功: {filepath}")
            return (item_id, pic_loc, "success")
        except Exception as e:
            if attempt == retries:
                logger.error(f"下载失败（重试{retries}次）: {item_id}_{pic_loc} | {url} | 错误: {e}")
                return (item_id, pic_loc, "failed")
            else:
                logger.warning(f"下载失败，{attempt}/{retries}次重试: {item_id}_{pic_loc} | {e}")
                time.sleep(1)


def download_images_from_excel(excel_path, output_dir="figs", workers=DEFAULT_WORKERS):
    logger.info(f"准备读取 Excel: {excel_path}")
    
    if not os.path.exists(excel_path):
        logger.error(f"Excel文件不存在: {excel_path}")
        logger.error("请检查路径是否正确，或者在命令行中输入正确的绝对路径。")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取 Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"读取Excel失败: {e}")
        return

    # 检查必要列
    required_cols = [COL_ID, COL_LOC, COL_URL]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Excel缺少列: '{col}'。当前Excel包含的列有: {list(df.columns)}")
            return

    total = len(df)
    logger.info(f"共 {total} 条记录，使用 {workers} 个线程下载到目录: {output_dir}")

    # 使用线程池下载
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, row, output_dir): row for _, row in df.iterrows()}

        # 进度条
        with tqdm(total=total, desc="下载进度", unit="张") as pbar:
            for future in as_completed(futures):
                item_id, pic_loc, status = future.result()
                
                # 统计状态更新 (status 的值是 "success", "skipped" 或 "failed")
                with stats_lock:
                    stats[status] += 1
                
                pbar.update(1)
                pbar.set_postfix(success=stats["success"], skipped=stats["skipped"], failed=stats["failed"])

    # 最终统计
    logger.info(f"下载完成！成功: {stats['success']}, 跳过已存在: {stats['skipped']}, 失败: {stats['failed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多线程批量下载图片")
    
    # 将默认路径直接填在这里
    default_excel = "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/7_picture_url.xlsx"
    default_output = "/home/xzhe162/wh_workspace/casual/processed_data/suning/picture"
    
    parser.add_argument("excel", nargs="?", default=default_excel, help=f"Excel文件路径（默认: {default_excel}）")
    parser.add_argument("-o", "--output", default=default_output, help=f"输出目录（默认: {default_output}）")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help=f"并发线程数（默认 {DEFAULT_WORKERS}）")
    
    args = parser.parse_args()

    download_images_from_excel(args.excel, args.output, args.workers)