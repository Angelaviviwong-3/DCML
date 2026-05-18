import pandas as pd
import numpy as np
import os
import logging
import time

# ===============================
# 1. 路径与环境配置
# ===============================
CLICK_PATH = "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/1_suning_y_click.xlsx"

CART_PURCHASE_PATHS = [
    "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/2_suning_y_cart&y_purchase_M4.xlsx",
    "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/2_suning_y_cart&y_purchase_M5.xlsx",
    "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/2_suning_y_cart&y_purchase_M6.xlsx",
]

OUTPUT_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/Y"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/raw_data/log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===============================
# 2. 日志配置
# ===============================
log_file = os.path.join(LOG_DIR, "process_suning_y_updated.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
start_time = time.time()

try:
    # ===============================
    # Step 1: 处理点击数据 (y_click 基准)
    # ===============================
    logger.info("Step 1: Loading click data...")
    click_df = pd.read_excel(CLICK_PATH, engine="openpyxl")

    # 重命名与去重
    click_df = click_df.rename(columns={
        "module_ID": "module_id",
        "slot_ID": "slot_id"
    })
    
    # 只要在点击表里，y_click 初始设为 1
    click_df["y_click"] = (click_df["click_quantity"].fillna(0) > 0).astype(int)
    
    # 构造位置变量 position
    click_df["position"] = click_df["module_id"].astype(str) + "_" + click_df["slot_id"].astype(str)

    click_df = click_df[["user_id", "item_id", "timestamp", "y_click", "position"]]
    click_df = click_df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])

    logger.info(f"Click records: {len(click_df)}")

    # ===============================
    # Step 2: 处理加购 & 成交数据
    # ===============================
    logger.info("Step 2: Merging M4-M6 cart & purchase data...")
    dfs = []
    for path in CART_PURCHASE_PATHS:
        logger.info(f"Reading: {path}")
        df = pd.read_excel(path, engine="openpyxl")
        dfs.append(df)

    cp_df = pd.concat(dfs, ignore_index=True)

    # 填充缺失值
    for col in ["y_cart", "y_purchase", "addcart_quantity", "purchase_quantity"]:
        cp_df[col] = cp_df[col].fillna(0)

    # 构造标签逻辑：数量大于0或标志位为1，则 y = 1
    cp_df["y_cart"] = ((cp_df["y_cart"] > 0) | (cp_df["addcart_quantity"] > 0)).astype(int)
    cp_df["y_purchase"] = ((cp_df["y_purchase"] > 0) | (cp_df["purchase_quantity"] > 0)).astype(int)

    cp_df = cp_df[["user_id", "item_id", "timestamp", "y_cart", "y_purchase"]]
    cp_df = cp_df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])

    logger.info(f"Cart/Purchase records: {len(cp_df)}")

    # ===============================
    # Step 3: 合并表并修正“漏斗包含逻辑” (核心修改)
    # ===============================
    logger.info("Step 3: Executing Outer Join and fixing Click logic...")
    
    # 使用 outer join 确保不论是点击还是加购购买，所有行为都被捕获
    y_df = pd.merge(
        click_df,
        cp_df,
        on=["user_id", "item_id", "timestamp"],
        how="outer"
    )

    # 逻辑修正：如果发生了加购 (y_cart=1) 或购买 (y_purchase=1)，
    # 则该行为在逻辑上必然包含了“点击曝光”，强制设置 y_click = 1
    y_df["y_cart"] = y_df["y_cart"].fillna(0).astype(int)
    y_df["y_purchase"] = y_df["y_purchase"].fillna(0).astype(int)
    
    # 如果 y_click 是空，且转化标签是 1，则将其补为 1；否则保留原值，缺失补 0
    mask_exposed = (y_df['y_cart'] == 1) | (y_df['y_purchase'] == 1)
    y_df.loc[mask_exposed, 'y_click'] = 1
    y_df['y_click'] = y_df['y_click'].fillna(0).astype(int)

    # 填充位置信息的缺失
    y_df["position"] = y_df["position"].fillna("trans_direct") # 标记为直接转化（非点击流引导）

    # ===============================
    # Step 4: 清洗与排序
    # ===============================
    logger.info("Step 4: Final cleaning...")
    # 统一 item_id 格式（去前导零）
    y_df["item_id"] = y_df["item_id"].astype(str).str.lstrip("0")
    # 按用户和时间排序，方便后续序列分析（如有需要）
    y_df = y_df.sort_values(by=["user_id", "timestamp"])

    # ===============================
    # Step 5: 保存文件（覆盖原有旧表）
    # ===============================
    logger.info("Step 5: Overwriting existing data files...")
    parquet_path = os.path.join(OUTPUT_DIR, "y_behavior.parquet")
    csv_path = os.path.join(OUTPUT_DIR, "y_behavior.csv")

    y_df.to_parquet(parquet_path, index=False)
    y_df.to_csv(csv_path, index=False)

    logger.info(f"Successfully saved to {parquet_path}")
    
    # ===============================
    # 数据统计摘要
    # ===============================
    logger.info("--- Data Summary ---")
    logger.info(f"Total merged rows: {len(y_df)}")
    logger.info(f"Final y_click=1: {y_df['y_click'].sum()}")
    logger.info(f"Final y_cart=1: {y_df['y_cart'].sum()}")
    logger.info(f"Final y_purchase=1: {y_df['y_purchase'].sum()}")
    
    end_time = time.time()
    logger.info(f"Total time elapsed: {round(end_time - start_time, 2)} seconds")

except Exception as e:
    logger.exception("Processing failed!")


# python /home/xzhe162/wh_workspace/casual/data_preprocessing/suning_preprocessing/build_y_funnel.py