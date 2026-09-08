#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download images concurrently with resume support, retries, and logging.
Default paths: python download_images_suning.py
Custom paths: python download_images_suning.py /path/to/data.xlsx -o /path/to/images/
"""

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths


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

# ----------------------------- Configuration -----------------------------
DEFAULT_WORKERS = 16           # Worker threads
DEFAULT_RETRIES = 3            # Download retries
DEFAULT_TIMEOUT = 30           # Request timeout (seconds)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
PROXIES = {"http": None, "https": None}   # Disable proxies

# These column names must match the headers in the source workbook.
COL_ID = "item_id"          # Product ID column
COL_LOC = "pic_loc"    # Image-position column
COL_URL = "full_picture_url"      # Image URL column

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("download_images.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread lock and global counters
stats_lock = Lock()
stats = {"success": 0, "failed": 0, "skipped": 0}


def get_file_extension(url, default=".jpg"):
    """Extract a file extension from the URL, or return the default."""
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        ext = default
    return ext


def sanitize_filename(filename):
    """Remove invalid filename characters."""
    invalid_chars = r'\/:*?"<>|'
    for ch in invalid_chars:
        filename = filename.replace(ch, '_')
    return filename


def download_one(row, output_dir, retries=DEFAULT_RETRIES):
    """Download one image and return (item_id, pic_loc, status)."""
    item_id = row[COL_ID]
    pic_loc = row[COL_LOC]
    url = str(row[COL_URL]).strip()

    # Skip empty URLs
    if pd.isna(url) or url == 'nan' or not url:
        logger.warning(f"Empty URL: {item_id}_{pic_loc}")
        return (item_id, pic_loc, "failed")

    # Determine the file extension and output path
    ext = get_file_extension(url)
    filename = f"{item_id}_{pic_loc}{ext}"
    filename = sanitize_filename(filename)
    filepath = os.path.join(output_dir, filename)

    # Resume: skip an existing nonempty file
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logger.debug(f"Skipped existing: {filepath}")
        return (item_id, pic_loc, "skipped")

    # Retry the download
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            # Save the file
            with open(filepath, "wb") as f:
                f.write(resp.content)
            logger.debug(f"Download succeeded: {filepath}")
            return (item_id, pic_loc, "success")
        except Exception as e:
            if attempt == retries:
                logger.error(f"Download failed after {retries} attempts: {item_id}_{pic_loc} | {url} | Error: {e}")
                return (item_id, pic_loc, "failed")
            else:
                logger.warning(f"Download failed, attempt {attempt}/{retries}: {item_id}_{pic_loc} | {e}")
                time.sleep(1)


def download_images_from_excel(excel_path, output_dir="figs", workers=DEFAULT_WORKERS):
    logger.info(f"Reading Excel: {excel_path}")

    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found: {excel_path}")
        logger.error("Check the path or supply the correct absolute path on the command line.")
        return

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return

    # Check required columns
    required_cols = [COL_ID, COL_LOC, COL_URL]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing Excel column: '{col}'. Available columns: {list(df.columns)}")
            return

    total = len(df)
    logger.info(f"{total} records; downloading with {workers} workers to: {output_dir}")

    # Download using a thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, row, output_dir): row for _, row in df.iterrows()}

        # Progress bar
        with tqdm(total=total, desc="Downloading", unit="image") as pbar:
            for future in as_completed(futures):
                item_id, pic_loc, status = future.result()

                # Update counters for success, skipped, or failed status
                with stats_lock:
                    stats[status] += 1

                pbar.update(1)
                pbar.set_postfix(success=stats["success"], skipped=stats["skipped"], failed=stats["failed"])

    # Final statistics
    logger.info(f"Downloads complete! Succeeded: {stats['success']}, Skipped existing: {stats['skipped']}, Failed: {stats['failed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images concurrently")

    # Default input and output paths
    default_excel = str(_dcml_paths.project_path('raw_data/suning_raw/7_picture_url.xlsx'))
    default_output = str(_dcml_paths.project_path('processed_data/suning/picture'))

    parser.add_argument("excel", nargs="?", default=default_excel, help=f"Excel input path (default: {default_excel})")
    parser.add_argument("-o", "--output", default=default_output, help=f"Output directory (default: {default_output})")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help=f"Worker threads (default: {DEFAULT_WORKERS})")

    args = parser.parse_args()

    download_images_from_excel(args.excel, args.output, args.workers)
