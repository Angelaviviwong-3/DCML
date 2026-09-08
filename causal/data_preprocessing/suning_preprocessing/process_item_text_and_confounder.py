
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
RAW_PARAMS_PATH = str(_dcml_paths.project_path('raw_data/suning_raw/4_item_features.xlsx'))
RAW_PRICE_PATH = str(_dcml_paths.project_path('raw_data/suning_raw/5_item_price.xlsx'))
SAVE_DIR = str(_dcml_paths.project_path('processed_data/suning/item_feature&Confounder_price'))
LOG_DIR = str(_dcml_paths.project_path('raw_data/log'))

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_filename = f"item_text_price_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        # Define progress stages
        steps = [
            "Load product attributes and prices",
            "Clean product IDs (remove leading zeros)",
            "Build product text (W) for the LMM",
            "Prepare the price confounder (x_pri)",
            "Merge features and export"
        ]
        pbar = tqdm(total=len(steps), desc="[Item Text & Price preprocessing]")

        # ------------------------------------------
        # STEP 1: Load data
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[0]}")
        df_params = pd.read_excel(RAW_PARAMS_PATH)
        df_price = pd.read_excel(RAW_PRICE_PATH)

        # Strip whitespace from column names
        df_params.columns = df_params.columns.str.strip()
        df_price.columns = df_price.columns.str.strip()
        pbar.update(1)

        # ------------------------------------------
        # STEP 2: Clean product IDs
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[1]}")
        # Normalize item_id as a stripped string without leading zeros
        df_params['item_id'] = df_params['item_id'].astype(str).str.strip().str.lstrip('0')
        df_price['item_id'] = df_price['item_id'].astype(str).str.strip().str.lstrip('0')
        pbar.update(1)

        # ------------------------------------------
        # STEP 3: Build natural-language product text (W) for the LMM
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[2]}")

        # Fill missing values before concatenating strings
        fill_cols = ['brand_name', 'cat1_name', 'cat2_name', 'cat3_name', 'item_name']
        for col in fill_cols:
            if col in df_params.columns:
                df_params[col] = df_params[col].fillna("Unknown")

        # Combine product attributes into a description for the language model
        # Format: Brand: <brand> | Category: <category hierarchy> | Product name: <name>
        df_params['text_W'] = (
            "Brand: " + df_params['brand_name'].astype(str) + " | " +
            "Category: " + df_params['cat1_name'].astype(str) + "-" +
                       df_params['cat2_name'].astype(str) + "-" +
                       df_params['cat3_name'].astype(str) + " | " +
            "Product name: " + df_params['item_name'].astype(str)
        )

        # Each row represents one product; deduplicate without grouped aggregation
        df_text_agg = df_params.drop_duplicates(subset=['item_id'])[['item_id', 'text_W', 'cat3_name']]
        pbar.update(1)

        # ------------------------------------------
        # STEP 4: Prepare the price confounder (x_pri)
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[3]}")
        # Deduplicate prices to keep one record per product
        df_price_clean = df_price.drop_duplicates(subset=['item_id'])[['item_id', 'price']]
        df_price_clean['price'] = pd.to_numeric(df_price_clean['price'], errors='coerce')

        # Merge prices into the product text table
        df_final = pd.merge(df_text_agg, df_price_clean, on='item_id', how='left')

        # Impute missing prices
        # Use the cat3_name median first, then the global median
        cat_median = df_final.groupby('cat3_name')['price'].transform('median')
        df_final['price'] = df_final['price'].fillna(cat_median)
        df_final['price'] = df_final['price'].fillna(df_final['price'].median())

        # Apply log(x + 1) to the long-tailed price variable
        df_final['x_pri_log'] = np.log1p(df_final['price'])
        pbar.update(1)

        # ------------------------------------------
        # STEP 5: Save the final feature matrix
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[4]}")
        # Select the fields required by the research model
        final_cols = ['item_id', 'text_W', 'x_pri_log']
        df_output = df_final[final_cols]

        parquet_path = os.path.join(SAVE_DIR, "item_text_price_matrix.parquet")
        csv_path = os.path.join(SAVE_DIR, "item_text_price_matrix.csv")

        # Save CSV with UTF-8 BOM for multilingual spreadsheet compatibility
        df_output.to_parquet(parquet_path, index=False)
        df_output.to_csv(csv_path, index=False, encoding='utf-8-sig')

        pbar.update(1)
        pbar.close()

        # Log the output
        logger.info("=========================================")
        logger.info(f"🎉 Product text and price-confounder processing complete!")
        logger.info(f"   - Unique products: {len(df_output)}")
        logger.info(f"   - Example product text (W): \n     {df_output['text_W'].iloc[0]}")
        logger.info(f"   - Output columns: {list(df_output.columns)}")
        logger.info(f"   - Results saved to: {parquet_path}")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise

if __name__ == "__main__":
    main()

#. python causal/data_preprocessing/suning_preprocessing/process_item_text\&confounder.py
