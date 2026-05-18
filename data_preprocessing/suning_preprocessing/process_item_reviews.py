import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 0. 路径与环境配置
# ==========================================
RAW_REVIEWS_PATH = "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/6_reviews&rating.xlsx"
SAVE_DIR_RATING = "/home/xzhe162/wh_workspace/casual/processed_data/suning/rating"
SAVE_DIR_REVIEWS = "/home/xzhe162/wh_workspace/casual/processed_data/suning/pure_reviews"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/raw_data/log"

os.makedirs(SAVE_DIR_RATING, exist_ok=True)
os.makedirs(SAVE_DIR_REVIEWS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
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
        logger.info(">>> 正在从 Excel 载入用户评论与评分数据...")
        # 1. 载入数据
        df = pd.read_excel(RAW_REVIEWS_PATH)
        
        # 统一去除表头空格
        df.columns = df.columns.str.strip()
        
        # 2. 根据您提供的中文字段样式进行重命名映射
        # 字段顺序：会员编码, 商品编码, 评价星级, 评价内容, 是否纯文本, 是否视频, 是否带图, 评论创建时间
        mapping = {
            '会员编码': 'user_id',
            '商品编码': 'item_id',
            '评价星级': 'rating',
            '评价内容': 'review_text',
            '是否纯文本评论': 'is_text_only',
            '是否视频评论': 'is_video_review',
            '是否带图评论': 'is_image_review',
            '评论创建时间': 'timestamp'
        }
        df = df.rename(columns=mapping)
        
        # 定义进度条
        steps = ["清洗主键与时间戳", "文本深度清洗(针对System 2)", "构建评分数据集(System 1)", "构建评论数据集(System 2)", "数据导出"]
        pbar = tqdm(total=len(steps), desc="[Reviews Data 预处理]")

        # ------------------------------------------
        # STEP 1: 清洗主键与时间戳对齐
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[0]}")
        # item_id 去前导零
        df['item_id'] = df['item_id'].astype(str).str.strip().str.lstrip('0')
        df['user_id'] = df['user_id'].astype(str).str.strip()
        
        # 时间戳转换为 YYYYMMDD 字符串格式
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d')
        pbar.update(1)

        # ------------------------------------------
        # STEP 2: 文本深度清洗 (针对理性路径设计)
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[1]}")
        garbage_texts = ["此用户没有填写评价内容", "该用户没有填写评价内容", "默认好评", "无评价内容", "nan", "None"]
        
        df['review_text'] = df['review_text'].astype(str).str.strip()
        
        # 初始化有效标记
        df['is_valid_review'] = True
        
        # 过滤掉无意义系统文本
        df.loc[df['review_text'].isin(garbage_texts), 'is_valid_review'] = False
        
        # 修正报错点：使用 .str.len() 获取长度
        df.loc[df['review_text'].str.len() < 2, 'is_valid_review'] = False
        pbar.update(1)

        # ------------------------------------------
        # STEP 3: 构建评分数据集 (用于 Path 2: Heuristics)
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[2]}")
        df_rating_final = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()
        df_rating_final['rating'] = pd.to_numeric(df_rating_final['rating'], errors='coerce')
        pbar.update(1)

        # ------------------------------------------
        # STEP 4: 构建评论数据集 (用于 Path 1: Analytical)
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[3]}")
        # 筛选有效评论
        df_pure_reviews = df[df['is_valid_review'] == True].copy()
        
        # 只保留实验需要的列
        keep_cols = ['user_id', 'item_id', 'review_text', 'timestamp']
        # 动态检查是否存在这些辅助列，存在则保留
        for col in ['is_text_only', 'is_video_review', 'is_image_review']:
            if col in df_pure_reviews.columns:
                keep_cols.append(col)
                
        df_pure_reviews = df_pure_reviews[keep_cols]
        pbar.update(1)

        # ------------------------------------------
        # STEP 5: 数据存储与导出
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[4]}")
        
        # 保存评分数据
        df_rating_final.to_parquet(os.path.join(SAVE_DIR_RATING, "user_item_rating.parquet"), index=False)
        df_rating_final.to_csv(os.path.join(SAVE_DIR_RATING, "user_item_rating.csv"), index=False)
        
        # 保存纯评论文本数据
        df_pure_reviews.to_parquet(os.path.join(SAVE_DIR_REVIEWS, "user_item_pure_reviews.parquet"), index=False)
        df_pure_reviews.to_csv(os.path.join(SAVE_DIR_REVIEWS, "user_item_pure_reviews.csv"), index=False, encoding='utf-8-sig')
        
        pbar.update(1)
        pbar.close()

        logger.info("=========================================")
        logger.info(f"🎉 评论与评分预处理完成！")
        logger.info(f"   - 原始总记录数: {len(df)}")
        logger.info(f"   - 有效评分记录: {len(df_rating_final)}")
        logger.info(f"   - 有效分析文本记录: {len(df_pure_reviews)}")
        logger.info(f"   - 结果保存至: {SAVE_DIR_RATING} 和 {SAVE_DIR_REVIEWS}")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"❌ 处理评论数据时出错: {e}")
        raise

if __name__ == "__main__":
    main()
    
    #. python /home/xzhe162/wh_workspace/casual/data_preprocessing/suning_preprocessing/process_item_reviews.py