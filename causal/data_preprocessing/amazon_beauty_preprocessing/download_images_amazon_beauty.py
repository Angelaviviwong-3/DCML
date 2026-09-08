#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent image downloader for Amazon Reviews data.
Read meta_All_Beauty.jsonl and download the high-resolution main image for each ASIN.
"""

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


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

# ----------------------------- Basic configuration -----------------------------
DATASET_NAME = "amazon_beauty"  # Dataset identifier

DEFAULT_WORKERS = 16
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 30
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
PROXIES = {"http": None, "https": None}

# ----------------------------- Path configuration -----------------------------
# All Beauty metadata input
DEFAULT_JSONL = str(_dcml_paths.project_path('raw_data/amazon_beauty_raw/meta_All_Beauty.jsonl'))
# Image output path
DEFAULT_OUTPUT = str(_dcml_paths.project_path('processed_data/amazon_beauty/pictures'))
# Log output path
LOG_DIR = str(_dcml_paths.project_path('log'))

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"download_{DATASET_NAME}_images.log")

# Set up logging
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
    """Download one image."""
    asin = task['asin']
    url = task['url']

    if not url:
        return (asin, "no_image")

    ext = get_file_extension(url)
    filename = f"{asin}_MAIN{ext}"
    filepath = os.path.join(output_dir, filename)

    # Resume: skip existing nonempty files
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
                logger.debug(f"[{DATASET_NAME}] Download failed after {retries} attempts: {asin} | {url} | Error: {e}")
                return (asin, "failed")
            else:
                time.sleep(1)

def extract_best_image(images_list):
    """
    Select the best main-image URL from the Amazon images list.
    Priority: MAIN hi_res, MAIN large, first-image hi_res, first-image large.
    """
    if not images_list:
        return None

    for img in images_list:
        if img.get("variant") == "MAIN":
            return img.get("hi_res") or img.get("large")

    # Fall back to the first image when no MAIN variant is available
    first_img = images_list[0]
    return first_img.get("hi_res") or first_img.get("large")

def download_amazon_images(jsonl_path, output_dir, dataset_name, workers=DEFAULT_WORKERS):
    logger.info(f"========== Processing dataset: {dataset_name} ==========")
    logger.info(f"Input path: {jsonl_path}")
    logger.info(f"Output path: {output_dir}")
    logger.info(f"Log path: {LOG_FILE}")

    if not os.path.exists(jsonl_path):
        logger.error(f"File not found: {jsonl_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Read line by line to build download tasks without loading the entire JSONL
    tasks = []
    logger.info(f"Scanning {dataset_name} JSONL for image URLs...")

    # Track parent_asin values to avoid duplicate downloads across variants
    seen_asins = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                asin = data.get("parent_asin") # Group variants by parent_asin
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
    logger.info(f"Scan complete! Found {total} product images in {dataset_name}; starting {workers} download workers...")

    # 2. Start threaded downloads
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, task, output_dir): task for task in tasks}

        # Prefix the progress bar with the dataset name
        with tqdm(total=total, desc=f"[{dataset_name}] Downloading", unit="image") as pbar:
            for future in as_completed(futures):
                asin, status = future.result()

                with stats_lock:
                    stats[status] += 1

                pbar.update(1)
                pbar.set_postfix(success=stats["success"], skipped=stats["skipped"], failed=stats["failed"])

    logger.info(f"[{dataset_name}] Downloads complete! Succeeded: {stats['success']}, Skipped (existing): {stats['skipped']}, Failed: {stats['failed']}, Products without images: {stats['no_image']}")
    logger.info("========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Amazon All Beauty main product images concurrently")

    parser.add_argument("jsonl", nargs="?", default=DEFAULT_JSONL, help="Amazon metadata JSONL input path")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Image output directory")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help="Worker threads")
    parser.add_argument("-d", "--dataset", default=DATASET_NAME, help="Dataset identifier")
    args = parser.parse_args()

    download_amazon_images(args.jsonl, args.output, args.dataset, args.workers)

    # python causal/data_preprocessing/amazon_beauty_preprocessing/download_images_amazon_beauty.py
