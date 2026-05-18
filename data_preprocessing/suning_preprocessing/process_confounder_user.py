import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 路径与环境配置
# ==========================================
RAW_DATA_PATH = "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/3_user_confounder.xlsx"
SAVE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/Confounder_user"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/raw_data/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
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
        logger.info(">>> 正在从 Excel 载入用户原始数据...")
        # 1. 载入数据
        df = pd.read_excel(RAW_DATA_PATH)
        
        # 统一去除表头可能存在的空格
        df.columns = df.columns.str.strip()
        
        # 根据截图样式的 7 列进行强制命名
        # 字段依次为：user_id, tier, reg_year, gende, age, city_tie, spending
        expected_cols = ['user_id', 'tier', 'reg_year', 'gende', 'age', 'city_tie', 'spending']
        
        if len(df.columns) == len(expected_cols):
            df.columns = expected_cols
        else:
            logger.warning(f"检测到列数 ({len(df.columns)}) 与预期 ({len(expected_cols)}) 不符，将尝试使用原表头。")

        # 定义进度条
        steps = ["清洗 ID 与缺失值", "特征转换(Tenure)", "One-Hot 编码类别特征", "构建最终特征矩阵", "数据存储导出"]
        pbar = tqdm(total=len(steps), desc="[User Confounder 预处理]")

        # ------------------------------------------
        # STEP 1: 清洗 ID 与缺失值
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[0]}")
        df['user_id'] = df['user_id'].astype(str).str.strip()
        
        # 针对分类字段填充缺失值
        fill_unknown_cols = ['tier', 'gende', 'age', 'city_tie']
        for col in fill_unknown_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        
        # 针对支出字段，将空白转为“未消费”
        df['spending'] = df['spending'].fillna('0元_未消费').astype(str).str.strip()
        pbar.update(1)

        # ------------------------------------------
        # STEP 2: 特征转换 (网龄 Tenure)
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[1]}")
        current_year = 2025
        df['reg_year'] = pd.to_numeric(df['reg_year'], errors='coerce')
        # 计算网龄
        df['user_tenure'] = current_year - df['reg_year']
        # 填充缺失网龄为中位数
        df['user_tenure'] = df['user_tenure'].fillna(df['user_tenure'].median())
        
        # Z-Score 标准化
        scaler = StandardScaler()
        df['user_tenure_scaled'] = scaler.fit_transform(df[['user_tenure']])
        pbar.update(1)

        # ------------------------------------------
        # STEP 3: One-Hot 编码
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[2]}")
        # 定义需要进行 One-Hot 的离散变量
        # 注意：即便 spending 只有几个区间，将其 One-hot 也是控制消费偏误的最佳实践
        cat_features = ['tier', 'gende', 'age', 'city_tie', 'spending']
        
        df_encoded = pd.get_dummies(df[cat_features], prefix=cat_features)
        
        # 将 True/False 转换为 0/1
        df_encoded = df_encoded.astype(int)
        pbar.update(1)

        # ------------------------------------------
        # STEP 4: 构建最终特征矩阵
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[3]}")
        # 拼接 user_id (主键), user_tenure_scaled (连续特征) 和 df_encoded (离散特征)
        x_user_final = pd.concat([
            df[['user_id', 'user_tenure_scaled']], 
            df_encoded
        ], axis=1)
        pbar.update(1)

        # ------------------------------------------
        # STEP 5: 数据存储导出
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[4]}")
        parquet_path = os.path.join(SAVE_DIR, "confounder_user_matrix.parquet")
        csv_path = os.path.join(SAVE_DIR, "confounder_user_matrix.csv")
        
        # 导出
        x_user_final.to_parquet(parquet_path, index=False)
        x_user_final.to_csv(csv_path, index=False)
        
        pbar.update(1)
        pbar.close()

        # 打印日志摘要
        logger.info("=========================================")
        logger.info(f"🎉 处理完成！最终用户特征矩阵详情：")
        logger.info(f"   - 样本行数: {len(x_user_final)}")
        logger.info(f"   - 原始字段: {list(df.columns)}")
        logger.info(f"   - 最终特征维度: {x_user_final.shape[1] - 1} 维")
        logger.info(f"   - 保存路径: {parquet_path}")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"❌ 预处理失败，错误信息: {e}")
        raise

if __name__ == "__main__":
    main()
    
# python /home/xzhe162/wh_workspace/casual/data_preprocessing/suning_preprocessing/process_confounder_user.py