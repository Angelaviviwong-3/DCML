import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 0. 路径与环境配置
# ==========================================
RAW_PARAMS_PATH = "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/4_item_features.xlsx"
RAW_PRICE_PATH = "/home/xzhe162/wh_workspace/casual/raw_data/suning_raw/5_item_price.xlsx"
SAVE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/item_feature&Confounder_price"
LOG_DIR = "/home/xzhe162/wh_workspace/casual/raw_data/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志
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
        # 定义进度条
        steps = [
            "载入商品参数与价格表", 
            "清洗商品ID(去前导零)", 
            "为LMM构建自然语言文本(W)", 
            "处理价格混淆变量(x_pri)", 
            "合并矩阵并导出"
        ]
        pbar = tqdm(total=len(steps), desc="[Item Text & Price 预处理]")

        # ------------------------------------------
        # STEP 1: 载入数据
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[0]}")
        df_params = pd.read_excel(RAW_PARAMS_PATH)
        df_price = pd.read_excel(RAW_PRICE_PATH)
        
        # 去除表头可能存在的空格
        df_params.columns = df_params.columns.str.strip()
        df_price.columns = df_price.columns.str.strip()
        pbar.update(1)

        # ------------------------------------------
        # STEP 2: 清洗商品 ID
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[1]}")
        # 根据最新列名 item_id 进行清洗：统一转字符串，去两端空格，去掉开头的 '0'
        df_params['item_id'] = df_params['item_id'].astype(str).str.strip().str.lstrip('0')
        df_price['item_id'] = df_price['item_id'].astype(str).str.strip().str.lstrip('0')
        pbar.update(1)

        # ------------------------------------------
        # STEP 3: 为 LMM 构建自然语言文本 (W)
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[2]}")
        
        # 填充 NaN 以防字符串拼接报错
        fill_cols = ['brand_name', 'cat1_name', 'cat2_name', 'cat3_name', 'item_name']
        for col in fill_cols:
            if col in df_params.columns:
                df_params[col] = df_params[col].fillna("未知")
        
        # 将宽表的基础信息拼接为一段大模型友好的描述语句
        # 格式示例："品牌：海尔(Haier) | 类目：冰箱洗衣机及相关配件-洗(烘)衣机-洗衣机 | 商品名称：海尔波轮洗衣机EB100B20Mate1"
        df_params['text_W'] = (
            "品牌：" + df_params['brand_name'].astype(str) + " | " +
            "类目：" + df_params['cat1_name'].astype(str) + "-" + 
                       df_params['cat2_name'].astype(str) + "-" + 
                       df_params['cat3_name'].astype(str) + " | " +
            "商品名称：" + df_params['item_name'].astype(str)
        )
        
        # 因为已经是宽表（每行代表一个商品），去重即可，不需要复杂的 groupby 聚合
        df_text_agg = df_params.drop_duplicates(subset=['item_id'])[['item_id', 'text_W', 'cat3_name']]
        pbar.update(1)

        # ------------------------------------------
        # STEP 4: 处理价格混淆变量 (x_pri)
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[3]}")
        # 价格表去重：防止同一商品存在多行价格
        df_price_clean = df_price.drop_duplicates(subset=['item_id'])[['item_id', 'price']]
        df_price_clean['price'] = pd.to_numeric(df_price_clean['price'], errors='coerce')
        
        # 将价格特征合并到文本特征表上
        df_final = pd.merge(df_text_agg, df_price_clean, on='item_id', how='left')

        # 处理价格缺失值 (Imputation)
        # 策略：优先用同类目(cat3_name)的中位数填充，若还是空，用全局中位数填充
        cat_median = df_final.groupby('cat3_name')['price'].transform('median')
        df_final['price'] = df_final['price'].fillna(cat_median)
        df_final['price'] = df_final['price'].fillna(df_final['price'].median())

        # 对数变换：log(x + 1)，因果推断处理具有长尾分布的货币变量的标准操作
        df_final['x_pri_log'] = np.log1p(df_final['price'])
        pbar.update(1)

        # ------------------------------------------
        # STEP 5: 保存最终矩阵
        # ------------------------------------------
        pbar.set_description(f"正在执行: {steps[4]}")
        # 提取论文模型需要的干净字段
        final_cols = ['item_id', 'text_W', 'x_pri_log']
        df_output = df_final[final_cols]
        
        parquet_path = os.path.join(SAVE_DIR, "item_text_price_matrix.parquet")
        csv_path = os.path.join(SAVE_DIR, "item_text_price_matrix.csv")
        
        # 保存 (CSV 加上 utf-8-sig 防止中文乱码)
        df_output.to_parquet(parquet_path, index=False)
        df_output.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        pbar.update(1)
        pbar.close()

        # 打印日志
        logger.info("=========================================")
        logger.info(f"🎉 商品侧多模态特征文本与价格混淆变量处理完成！")
        logger.info(f"   - 独立商品数: {len(df_output)}")
        logger.info(f"   - 构造的多模态文本 (W) 示例: \n     {df_output['text_W'].iloc[0]}")
        logger.info(f"   - 最终输出维度: {list(df_output.columns)}")
        logger.info(f"   - 结果保存至: {parquet_path}")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"❌ 处理过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
    
#. python /home/xzhe162/wh_workspace/casual/data_preprocessing/suning_preprocessing/process_item_text\&confounder.py