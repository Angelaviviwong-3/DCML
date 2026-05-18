#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================
# 修复 OpenBLAS 多核并行溢出的段错误
# ⚠️ 必须放在所有其他 import 之前！
# ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import pandas as pd
import numpy as np
import json
import re
import glob
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from scipy.stats import rankdata

# ==========================================
# 0. 参数解析与配置 (Test 1 模式)
# ==========================================
parser = argparse.ArgumentParser(description="Amazon Multimodal Feature Extraction (Test1 Mode)")
parser.add_argument("-d", "--dataset", default="amazon_appliances", choices=["amazon_appliances", "amazon_beauty"], help="选择数据集")
args = parser.parse_args()

DATASET_NAME = args.dataset

# ✅ 开启测试模式
TEST_MODE = True  
TEST_SIZE = 1000   

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

BASE_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}/"
ITEM_BASE_PATH = os.path.join(BASE_DIR, "item_feature&Confounder_price/item_text_price_matrix.parquet")
ITEM_REVIEWS_PATH = os.path.join(BASE_DIR, "pure_reviews/user_item_pure_reviews.parquet")
# ⚠️ 注意亚马逊的图片文件夹名为 pictures
IMAGE_DIR = os.path.join(BASE_DIR, "pictures") 

# ✅ 输出路径 (全部使用 _test1 后缀)
SAVE_DIR = os.path.join(BASE_DIR, "item_multimodal_scalars")
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, f"{DATASET_NAME}_item_multimodal_scalars_test1.parquet")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, f"checkpoint_{DATASET_NAME}_scalars_test1.csv")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"lmm_{DATASET_NAME}_test1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 加载模型
# ==========================================
logger.info(f"========== 启动亚马逊多模态提取 (数据集: {DATASET_NAME} | 模式: TEST1) ==========")
logger.info(f">>> 正在加载大模型: {MODEL_ID} ...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
logger.info(">>> 模型加载完毕！")

# ==========================================
# 2. 核心提取函数（抽取 3大宏观 + 6大微观原子）
# ==========================================
def extract_causal_scalars_hierarchical(item_id, aggregated_text, img_path):

    # ⚠️ 这里已经全部替换为严格的英文 Prompt，以对齐英文数据集并减少大模型幻觉
    prompt = f"""
    You are an extremely rigorous and critical e-commerce visual review expert. Your task is to objectively extract atomic features from product image-text combinations and evaluate their visual attention weights.

    [Core Rule: Anti-Sympathy Bias]
    We require a true continuous distribution. Mindless high-scoring is strictly prohibited! You must rigorously follow this "Pyramid Scoring Rule":
    - [0]: The element is completely absent in the visual/textual space.
    - [1-3]: Barely mentioned, extremely small font, or placed at the edge (The vast majority of items fall here).
    - [4-6]: Conventionally displayed, occupying a moderate portion.
    - [7-8]: Highly prominent, with large typography in the core area.
    - [9-10]: Extremely rare! Unless the element consumes the entire visual canvas, scores above 9 are absolutely forbidden!

    [Special Suppression for Functional/Factual Features (atomic_functional)]:
    Do NOT give high scores just because "specifications" exist! High scores (7+) are ONLY allowed if the image presents hard-core "internal structure perspectives", "macro magnifications of core materials", or "dozens of parameters in a comparison table". For ordinary white-background product images, decisively assign a score of 1-2!

    [Task Requirements]
    1. Evaluate the intensity score of each atomic component (0-10). If the element does not exist, you must decisively assign a score of 0!
    2. Assign an [Attention Weight / Saliency Ratio] (0.0-1.0) for components within the same category. The sum of weights within a category MUST equal 1.0.
    3. 'a_text_fact' represents the independent factual density in text; it only needs a score (0-10), NO weight allocation is required.

    [Product Textual Context]:
    {aggregated_text}

    You MUST strictly output the following JSON format (Do NOT change the keys, they must perfectly align with our mathematical formulas):
    {{
        "reasoning": "According to the Pyramid Scoring Rule, critically analyze the visual and textual hierarchy...",
        "atomic_marketing": {{
            "a_pri": {{"score": [0-10], "weight": [0.0-1.0]}},
            "a_gft": {{"score": [0-10], "weight": [0.0-1.0]}},
            "a_sub": {{"score": [0-10], "weight": [0.0-1.0]}},
            "a_urg": {{"score": [0-10], "weight": [0.0-1.0]}}
        }},
        "atomic_functional": {{
            "a_text_fact": {{"score": [0-10]}},
            "a_spec": {{"score": [0-10], "weight": [0.0-1.0]}},
            "a_str": {{"score": [0-10], "weight": [0.0-1.0]}}
        }}
    }}
    """

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": prompt}
        ]
    }]

    text_ptr = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_ptr],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    def safe_extract(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            m = re.search(r'\d*\.?\d+', val)
            if m:
                return float(m.group())
        return 0.0

    try:
        # ✅ 三次蒙特卡洛逼近（论文 C.2）
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            num_return_sequences=3
        )

        generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
        result_strs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        scores = {
            'T_con_mkt': [], 'T_int_vis': [], 'T_int_fac': [],
            'a_pri': [], 'a_gft': [], 'a_sub': [], 'a_urg': [],
            'a_spec': [], 'a_str': []
        }

        for res_str in result_strs:
            res_str = res_str.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', res_str, re.DOTALL)
            if not match:
                continue

            try:
                res_json = json.loads(match.group(0))
                mkt = res_json.get('atomic_marketing', {})
                func = res_json.get('atomic_functional', {})
                
                def get_sw(d, key):
                    s = safe_extract(d.get(key, {}).get('score', 0)) / 10.0
                    w = safe_extract(d.get(key, {}).get('weight', 0.0))
                    return s, w

                # --- 1. 获取所有原子的 Score 和 Weight ---
                s_pri, w_pri = get_sw(mkt, 'a_pri')
                s_gft, w_gft = get_sw(mkt, 'a_gft')
                s_sub, w_sub = get_sw(mkt, 'a_sub')
                s_urg, w_urg = get_sw(mkt, 'a_urg')

                s_spec, w_spec = get_sw(func, 'a_spec')
                s_str, w_str = get_sw(func, 'a_str')
                
                s_fact = safe_extract(func.get('a_text_fact', {}).get('score', 0)) / 10.0

                # --- 2. 计算宏观得分 (Macro Aggregation) ---
                sum_w_mkt = sum([w_pri, w_gft, w_sub, w_urg])
                sum_w_mkt = sum_w_mkt if sum_w_mkt > 0 else 1.0
                macro_mkt = (s_pri*w_pri + s_gft*w_gft + s_sub*w_sub + s_urg*w_urg) / sum_w_mkt

                sum_w_vis = sum([w_spec, w_str])
                sum_w_vis = sum_w_vis if sum_w_vis > 0 else 1.0
                macro_func_vis = (s_spec*w_spec + s_str*w_str) / sum_w_vis

                # --- 3. 填入字典 ---
                scores['T_con_mkt'].append(macro_mkt)
                scores['T_int_vis'].append(macro_func_vis)
                scores['T_int_fac'].append(s_fact) 

                scores['a_pri'].append(s_pri)
                scores['a_gft'].append(s_gft)
                scores['a_sub'].append(s_sub)
                scores['a_urg'].append(s_urg)
                scores['a_spec'].append(s_spec)
                scores['a_str'].append(s_str)

            except:
                continue

        # 如果3次解析全失败，视为无效
        if not scores['T_con_mkt']:
            return None

        return {
            "item_id": item_id,
            "T_con_mkt": round(np.mean(scores['T_con_mkt']), 4),
            "T_int_vis": round(np.mean(scores['T_int_vis']), 4),
            "T_int_fac": round(np.mean(scores['T_int_fac']), 4),
            
            "atomic_a_pri": round(np.mean(scores['a_pri']), 4),
            "atomic_a_gft": round(np.mean(scores['a_gft']), 4),
            "atomic_a_sub": round(np.mean(scores['a_sub']), 4),
            "atomic_a_urg": round(np.mean(scores['a_urg']), 4),
            "atomic_a_spec": round(np.mean(scores['a_spec']), 4),
            "atomic_a_str": round(np.mean(scores['a_str']), 4)
        }

    except Exception as e:
        logger.error(f"[{item_id}] LMM 报错: {e}")
        return None

# ==========================================
# 3. 完美平滑零值保留秩归一化
# ==========================================
def zero_preserved_rank_norm(x, seed=2025):
    """
    1. 严格保留 0 值作为对照组。
    2. 加入 1e-6 级别极小随机扰动打破同分。
    3. ordinal 强制连续唯一排名。
    """
    x_arr = np.array(x, dtype=float)
    x_calibrated = np.zeros_like(x_arr)
    
    mask_pos = (x_arr > 0)
    num_pos = mask_pos.sum()
    
    if num_pos > 0:
        x_pos = x_arr[mask_pos]
        np.random.seed(seed)
        jitter = np.random.uniform(0, 1e-6, size=num_pos)
        x_pos_jittered = x_pos + jitter
        
        ranks = rankdata(x_pos_jittered, method='ordinal')
        x_calibrated[mask_pos] = ranks / float(num_pos)
        
    return x_calibrated

# ==========================================
# 4. 主流程
# ==========================================
def main():
    logger.info(">>> 读取特征与评论数据...")
    df_base = pd.read_parquet(ITEM_BASE_PATH)
    df_reviews = pd.read_parquet(ITEM_REVIEWS_PATH)

    # 聚合每个商品的评论前5条，防止上下文超长
    df_reviews_agg = df_reviews.groupby('item_id')['review_text'].apply(
        lambda x: ' | '.join(x.dropna().astype(str).unique()[:5])[:300]
    ).reset_index().rename(columns={'review_text': 'all_reviews'})

    df_input = pd.merge(
        df_base[['item_id', 'text_W']],
        df_reviews_agg,
        on='item_id',
        how='left'
    )

    # ⚠️ 文本构建也全部替换为了英文头部，避免模型中英混淆
    df_input['aggregated_text'] = (
        "[Core Specifications]\n" + df_input['text_W'] +
        "\n\n[In-depth User Feedback]\n" + df_input['all_reviews'].fillna("No in-depth feedback available.")
    )
    
    # ✅ 测试模式截断
    if TEST_MODE:
        df_input = df_input.head(TEST_SIZE)
        logger.info(f"⚠️ 开启 TEST_MODE_1，仅处理前 {TEST_SIZE} 条数据用于验证。")

    # 断点续传逻辑
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f">>> 加载已有 Checkpoint: {CHECKPOINT_PATH}")
        df_ckpt = pd.read_csv(CHECKPOINT_PATH)
        processed_ids = set(df_ckpt['item_id'].astype(str))
        results = df_ckpt.to_dict('records')
    else:
        processed_ids = set()
        results = []

    logger.info(f">>> 当前任务进度: {len(processed_ids)} / {len(df_input)}")

    pbar = tqdm(total=len(df_input), desc=f"[{DATASET_NAME} Test1] 提取中")
    pbar.update(len(processed_ids))

    for _, row in df_input.iterrows():
        item_id = str(row['item_id'])

        if item_id in processed_ids:
            pbar.update(1)
            continue

        # ⚠️ 亚马逊适配：通过 glob 获取图片，不强绑后缀名
        img_pattern = os.path.join(IMAGE_DIR, f"{item_id}_MAIN.*")
        matched_imgs = glob.glob(img_pattern)
        
        if not matched_imgs:
            # 如果这件商品没爬下来图片，跳过
            pbar.update(1)
            continue
            
        img_path = matched_imgs[0]

        # 调用核心推理
        scores = extract_causal_scalars_hierarchical(
            item_id,
            row['aggregated_text'],
            img_path
        )

        if scores:
            results.append(scores)
            processed_ids.add(item_id)

            pbar.set_postfix({
                "mkt(System1)": f"{scores['T_con_mkt']:.2f}",
                "vis(System2)": f"{scores['T_int_vis']:.2f}"
            })

        # 备份机制
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            
        pbar.update(1)
            
    pbar.close()

    if len(results) == 0:
        logger.error("❌ 没有抽取到任何有效数据 (可能是图片全没找到或 LMM 全盘报错)，程序退出。")
        return

    df_res = pd.DataFrame(results)

    # ==========================================
    # ✅ 对所有宏观和视觉原子成分执行绝对平滑的分布校准
    # ==========================================
    logger.info(">>> 正在执行计量经济学要求的分布校准 (Zero-Preserved Rank Norm)...")
    
    cols_to_calibrate = [
        'T_con_mkt', 'T_int_vis', 'T_int_fac',
        'atomic_a_pri', 'atomic_a_gft', 'atomic_a_sub', 'atomic_a_urg', 
        'atomic_a_spec', 'atomic_a_str'
    ]
    
    for col in cols_to_calibrate:
        if col in df_res.columns:
            df_res[f'{col}_calibrated'] = zero_preserved_rank_norm(df_res[col])

    # 保存最终 Parquet
    df_res.to_parquet(FINAL_SAVE_PATH, index=False)
    # 为了方便你直观肉眼查看，同时也存一份 CSV
    csv_save_path = FINAL_SAVE_PATH.replace(".parquet", ".csv")
    df_res.to_csv(csv_save_path, index=False)
    
    logger.info(f"🎉 TEST1 数据提取与平滑校准完成！")
    logger.info(f"💾 纯数据已保存在: {FINAL_SAVE_PATH}")
    logger.info(f"💾 CSV明文已保存在: {csv_save_path}")
    logger.info(f"🧐 最新一条商品抽样的样例输出核对:\n{df_res.iloc[-1].to_dict()}")

if __name__ == "__main__":
    main()
    
    
# 测试 Appliances
# python /home/xzhe162/wh_workspace/casual/model/M_amazon/MCRE/extract_lmm_multimodal_feature_attention_amazon_test1.py -d amazon_appliances

# 或者测试 Beauty
# python /home/xzhe162/wh_workspace/casual/model/M_amazon/MCRE/extract_lmm_multimodal_feature_attention_amazon_test1.py -d amazon_beauty