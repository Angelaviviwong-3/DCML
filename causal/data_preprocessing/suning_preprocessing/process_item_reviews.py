
# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[2]))
import project_paths as _dcml_paths

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 0. Paths and environment
# ==========================================
RAW_REVIEWS_PATH = str(_dcml_paths.project_path('raw_data/suning_raw/6_reviews&rating.xlsx'))
SAVE_DIR_RATING = str(_dcml_paths.project_path('processed_data/suning/rating'))
SAVE_DIR_REVIEWS = str(_dcml_paths.project_path('processed_data/suning/pure_reviews'))
LOG_DIR = str(_dcml_paths.project_path('raw_data/log'))

os.makedirs(SAVE_DIR_RATING, exist_ok=True)
os.makedirs(SAVE_DIR_REVIEWS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_filename = f"reviews_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info(">>> Loading user reviews and ratings from Excel...")
        # 1. Load data
        df = pd.read_excel(RAW_REVIEWS_PATH)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # 2. Map the original Suning workbook headers to English field names
        # Field order: user ID, item ID, rating, review text, text-only flag, video flag, image flag, timestamp
        mapping = {
            '\u4f1a\u5458\u7f16\u7801': 'user_id',
            '\u5546\u54c1\u7f16\u7801': 'item_id',
            '\u8bc4\u4ef7\u661f\u7ea7': 'rating',
            '\u8bc4\u4ef7\u5185\u5bb9': 'review_text',
            '\u662f\u5426\u7eaf\u6587\u672c\u8bc4\u8bba': 'is_text_only',
            '\u662f\u5426\u89c6\u9891\u8bc4\u8bba': 'is_video_review',
            '\u662f\u5426\u5e26\u56fe\u8bc4\u8bba': 'is_image_review',
            '\u8bc4\u8bba\u521b\u5efa\u65f6\u95f4': 'timestamp'
        }
        df = df.rename(columns=mapping)

        # Define progress stages
        steps = ["Clean keys and timestamps", "Clean review text (System 2)", "Build ratings (System 1)", "Build reviews (System 2)", "Export data"]
        pbar = tqdm(total=len(steps), desc="[Reviews Data preprocessing]")

        # ------------------------------------------
        # STEP 1: Clean keys and align timestamps
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[0]}")
        # Remove leading zeros from item_id
        df['item_id'] = df['item_id'].astype(str).str.strip().str.lstrip('0')
        df['user_id'] = df['user_id'].astype(str).str.strip()

        # Convert timestamps to YYYYMMDD strings
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d')
        pbar.update(1)

        # ------------------------------------------
        # STEP 2: Clean text for the analytical pathway
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[1]}")
        garbage_texts = ["\u6b64\u7528\u6237\u6ca1\u6709\u586b\u5199\u8bc4\u4ef7\u5185\u5bb9", "\u8be5\u7528\u6237\u6ca1\u6709\u586b\u5199\u8bc4\u4ef7\u5185\u5bb9", "\u9ed8\u8ba4\u597d\u8bc4", "\u65e0\u8bc4\u4ef7\u5185\u5bb9", "nan", "None"]

        df['review_text'] = df['review_text'].astype(str).str.strip()

        # Initialize validity flags
        df['is_valid_review'] = True

        # Filter system-generated empty-review messages (original literals retained)
        df.loc[df['review_text'].isin(garbage_texts), 'is_valid_review'] = False

        # Use .str.len() to measure review length
        df.loc[df['review_text'].str.len() < 2, 'is_valid_review'] = False
        pbar.update(1)

        # ------------------------------------------
        # STEP 3: Build the ratings dataset (Path 2: Heuristics)
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[2]}")
        df_rating_final = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()
        df_rating_final['rating'] = pd.to_numeric(df_rating_final['rating'], errors='coerce')
        pbar.update(1)

        # ------------------------------------------
        # STEP 4: Build the review dataset (Path 1: Analytical)
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[3]}")
        # Select valid reviews
        df_pure_reviews = df[df['is_valid_review'] == True].copy()

        # Keep the columns required by the experiment
        keep_cols = ['user_id', 'item_id', 'review_text', 'timestamp']
        # Retain optional auxiliary columns when present
        for col in ['is_text_only', 'is_video_review', 'is_image_review']:
            if col in df_pure_reviews.columns:
                keep_cols.append(col)

        df_pure_reviews = df_pure_reviews[keep_cols]
        pbar.update(1)

        # ------------------------------------------
        # STEP 5: Save and export data
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[4]}")

        # Save ratings
        df_rating_final.to_parquet(os.path.join(SAVE_DIR_RATING, "user_item_rating.parquet"), index=False)
        df_rating_final.to_csv(os.path.join(SAVE_DIR_RATING, "user_item_rating.csv"), index=False)

        # Save review text
        df_pure_reviews.to_parquet(os.path.join(SAVE_DIR_REVIEWS, "user_item_pure_reviews.parquet"), index=False)
        df_pure_reviews.to_csv(os.path.join(SAVE_DIR_REVIEWS, "user_item_pure_reviews.csv"), index=False, encoding='utf-8-sig')

        pbar.update(1)
        pbar.close()

        logger.info("=========================================")
        logger.info(f"🎉 Review and rating preprocessing complete!")
        logger.info(f"   - Total source records: {len(df)}")
        logger.info(f"   - Valid rating records: {len(df_rating_final)}")
        logger.info(f"   - Valid analytical-text records: {len(df_pure_reviews)}")
        logger.info(f"   - Results saved to: {SAVE_DIR_RATING} and {SAVE_DIR_REVIEWS}")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"❌ Review processing error: {e}")
        raise

if __name__ == "__main__":
    main()

    #. python causal/data_preprocessing/suning_preprocessing/process_item_reviews.py
