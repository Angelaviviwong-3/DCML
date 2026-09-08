
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
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. Paths and environment
# ==========================================
RAW_DATA_PATH = str(_dcml_paths.project_path('raw_data/suning_raw/3_user_confounder.xlsx'))
SAVE_DIR = str(_dcml_paths.project_path('processed_data/suning/Confounder_user'))
LOG_DIR = str(_dcml_paths.project_path('raw_data/log'))

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_filename = f"user_confounder_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        logger.info(">>> Loading raw user data from Excel...")
        # 1. Load data
        df = pd.read_excel(RAW_DATA_PATH)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Assign the seven column names expected by the source schema
        # Field order：user_id, tier, reg_year, gende, age, city_tie, spending
        expected_cols = ['user_id', 'tier', 'reg_year', 'gende', 'age', 'city_tie', 'spending']

        if len(df.columns) == len(expected_cols):
            df.columns = expected_cols
        else:
            logger.warning(f"Found {len(df.columns)} columns; expected {len(expected_cols)}. Keeping the original headers.")

        # Define progress stages
        steps = ["Clean IDs and missing values", "Transform tenure", "One-hot encode categorical features", "Build the final feature matrix", "Save and export data"]
        pbar = tqdm(total=len(steps), desc="[User Confounder preprocessing]")

        # ------------------------------------------
        # STEP 1: Clean IDs and missing values
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[0]}")
        df['user_id'] = df['user_id'].astype(str).str.strip()

        # Impute missing categorical values
        fill_unknown_cols = ['tier', 'gende', 'age', 'city_tie']
        for col in fill_unknown_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()

        # Fill missing spending with the original no-spending category
        df['spending'] = df['spending'].fillna('0\u5143_\u672a\u6d88\u8d39').astype(str).str.strip()
        pbar.update(1)

        # ------------------------------------------
        # STEP 2: Transform account tenure
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[1]}")
        current_year = 2025
        df['reg_year'] = pd.to_numeric(df['reg_year'], errors='coerce')
        # Calculate account tenure
        df['user_tenure'] = current_year - df['reg_year']
        # Impute missing tenure with its median
        df['user_tenure'] = df['user_tenure'].fillna(df['user_tenure'].median())

        # Z-score standardization
        scaler = StandardScaler()
        df['user_tenure_scaled'] = scaler.fit_transform(df[['user_tenure']])
        pbar.update(1)

        # ------------------------------------------
        # STEP 3: One-hot encoding
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[2]}")
        # Select categorical variables for one-hot encoding
        # One-hot encode spending bands as categorical covariates
        cat_features = ['tier', 'gende', 'age', 'city_tie', 'spending']

        df_encoded = pd.get_dummies(df[cat_features], prefix=cat_features)

        # Convert True/False to 0/1
        df_encoded = df_encoded.astype(int)
        pbar.update(1)

        # ------------------------------------------
        # STEP 4: Build the final feature matrix
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[3]}")
        # Combine user_id, scaled tenure, and encoded categorical features
        x_user_final = pd.concat([
            df[['user_id', 'user_tenure_scaled']], 
            df_encoded
        ], axis=1)
        pbar.update(1)

        # ------------------------------------------
        # STEP 5: Save and export data
        # ------------------------------------------
        pbar.set_description(f"Running: {steps[4]}")
        parquet_path = os.path.join(SAVE_DIR, "confounder_user_matrix.parquet")
        csv_path = os.path.join(SAVE_DIR, "confounder_user_matrix.csv")

        # Export
        x_user_final.to_parquet(parquet_path, index=False)
        x_user_final.to_csv(csv_path, index=False)

        pbar.update(1)
        pbar.close()

        # Log a summary
        logger.info("=========================================")
        logger.info(f"🎉 Processing complete! Final user feature matrix:")
        logger.info(f"   - Sample rows: {len(x_user_final)}")
        logger.info(f"   - Source fields: {list(df.columns)}")
        logger.info(f"   - Number of features: {x_user_final.shape[1] - 1}")
        logger.info(f"   - Output path: {parquet_path}")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()

# python causal/data_preprocessing/suning_preprocessing/process_confounder_user.py
