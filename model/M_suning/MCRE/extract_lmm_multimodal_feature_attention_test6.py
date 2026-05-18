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
import logging
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from scipy.stats import rankdata

# ==========================================
# 0. 配置（小样本测试模式 test6）
# ==========================================
TEST_MODE = True  # ✅ 开启测试模式
TEST_SIZE = 1000  # ✅ 仅抽取 1000 条

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/"
ITEM_BASE_PATH = os.path.join(BASE_DIR, "item_feature&Confounder_price/item_text_price_matrix.parquet")
ITEM_REVIEWS_PATH = os.path.join(BASE_DIR, "pure_reviews/user_item_pure_reviews.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "picture")

# ✅ 输出路径 (全部使用 _test6 后缀)
SAVE_DIR = os.path.join(BASE_DIR, "item_multimodal_scalars")
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, "item_multimodal_scalars_test6.parquet")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_scalars_test6.csv")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"lmm_test6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 加载模型
# ==========================================
logger.info(f">>> Loading model: {MODEL_ID}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
logger.info(">>> Model loaded")

# ==========================================
# 2. 核心提取函数（抽取 3大宏观 + 6大微观原子）
# ==========================================
def extract_causal_scalars_hierarchical(item_id, aggregated_text, img_path):

    prompt = f"""
    你是一名极其严苛且吹毛求疵的电商视觉与认知心理学专家。你的任务是客观提取商品图文中的原子特征，并评估它们的视觉注意力权重。

    【核心规则：严禁高分通胀 (Sympathy Bias)】
    我们要求的是真实的连续分布，严禁无脑打高分！你必须遵循以下“金字塔打分法则”：
    - 【0分】：画面/文本中完全不存在该元素。
    - 【1-3分】：仅仅提了一嘴，或者字号极小、在画面边缘（绝大部分商品应属于此列）。
    - 【4-6分】：常规展示，占据画面1/4左右，属于正常介绍。
    - 【7-8分】：非常显著，大字号标注，占据画面核心位置。
    - 【9-10分】：极其罕见！除非整个画面全被该元素占满，否则绝不允许打9分以上！

    【针对功能与事实特征 (atomic_functional) 的特殊压制】：
    不要因为有“规格”就给高分！只有当图片呈现了极其硬核的“内部结构透视图”、“核心材质微距放大”或“几十项参数对比表”时，才允许打高分。普通的白底商品图，请果断打 1-2 分！

    【任务要求】
    1. 评估各个原子组件的强度（0-10）。如果没有该元素，必须果断打 0 分！
    2. 为同属于视觉/营销类别的组件分配【注意力权重/显著性占比】（0.0-1.0）。权重之和必须为 1.0。
    3. a_text_fact 为独立的文本事实密度，只需给出 0-10 的 score，无需分配 weight。

    【商品文本背景】：
    {aggregated_text}

    必须严格输出以下JSON格式（严禁改变键名，必须与论文公式符号完美对齐）：
    {{
        "reasoning": "根据金字塔打分法则，严格批判画面和文本的主次...",
        "atomic_marketing": {{
            "a_pri": {{"score": 0-10, "weight": 0.0-1.0}},
            "a_gft": {{"score": 0-10, "weight": 0.0-1.0}},
            "a_sub": {{"score": 0-10, "weight": 0.0-1.0}},
            "a_urg": {{"score": 0-10, "weight": 0.0-1.0}}
        }},
        "atomic_functional": {{
            "a_text_fact": {{"score": 0-10}},
            "a_spec": {{"score": 0-10, "weight": 0.0-1.0}},
            "a_str": {{"score": 0-10, "weight": 0.0-1.0}}
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
        logger.error(f"{item_id} error: {e}")
        return None

# ==========================================
# 3. [升级版] 完美平滑零值保留秩归一化
# ==========================================
def zero_preserved_rank_norm(x, seed=2025):
    """
    1. 严格保留 0 值作为对照组 (Control Group)。
    2. 加入 1e-6 级别的极小随机扰动(Jitter)打破同分(Ties)。
    3. 使用 ordinal 强制赋予连续且唯一的排名，彻底消除 ECDF 图的阶梯状。
    """
    x_arr = np.array(x, dtype=float)
    x_calibrated = np.zeros_like(x_arr)
    
    mask_pos = (x_arr > 0)
    num_pos = mask_pos.sum()
    
    if num_pos > 0:
        x_pos = x_arr[mask_pos]
        
        # 加入极微小的扰动打破同分结，严格固定种子保证结果100%可复现
        np.random.seed(seed)
        jitter = np.random.uniform(0, 1e-6, size=num_pos)
        x_pos_jittered = x_pos + jitter
        
        # ordinal 会强行给相同的值排出一个顺序，结合 Jitter 做到绝对平滑
        ranks = rankdata(x_pos_jittered, method='ordinal')
        x_calibrated[mask_pos] = ranks / float(num_pos)
        
    return x_calibrated

# ==========================================
# 4. 主流程
# ==========================================
def main():
    logger.info(">>> Loading data...")
    df_base = pd.read_parquet(ITEM_BASE_PATH)
    df_reviews = pd.read_parquet(ITEM_REVIEWS_PATH)

    df_reviews_agg = df_reviews.groupby('item_id')['review_text'].apply(
        lambda x: ' | '.join(x.dropna().astype(str).unique()[:5])[:300]
    ).reset_index().rename(columns={'review_text': 'all_reviews'})

    df_input = pd.merge(
        df_base[['item_id', 'text_W']],
        df_reviews_agg,
        on='item_id',
        how='left'
    )

    df_input['aggregated_text'] = (
        "【核心规格】\n" + df_input['text_W'] +
        "\n\n【用户反馈】\n" + df_input['all_reviews'].fillna("暂无")
    )
    
    # ✅ 测试模式截断
    if TEST_MODE:
        df_input = df_input.head(TEST_SIZE)
        logger.info(f"⚠️ 开启 TEST_MODE，仅处理前 {TEST_SIZE} 条数据")

    if os.path.exists(CHECKPOINT_PATH):
        logger.info(">>> Loading test6 checkpoint...")
        df_ckpt = pd.read_csv(CHECKPOINT_PATH)
        processed_ids = set(df_ckpt['item_id'].astype(str))
        results = df_ckpt.to_dict('records')
    else:
        processed_ids = set()
        results = []

    logger.info(f">>> 已完成: {len(processed_ids)} / {len(df_input)}")

    pbar = tqdm(total=len(df_input), desc="Processing Test6 Dataset")
    pbar.update(len(processed_ids))

    for _, row in df_input.iterrows():
        item_id = str(row['item_id'])

        if item_id in processed_ids:
            pbar.update(1)
            continue

        img_path = os.path.join(IMAGE_DIR, f"{item_id}_1.jpg")
        if not os.path.exists(img_path):
            pbar.update(1)
            continue

        scores = extract_causal_scalars_hierarchical(
            item_id,
            row['aggregated_text'],
            img_path
        )

        if scores:
            results.append(scores)
            processed_ids.add(item_id)

            pbar.set_postfix({
                "mkt": scores['T_con_mkt'],
                "vis": scores['T_int_vis']
            })

        # ✅ 测试模式每 50 条存一次
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            logger.info(f"Test6 Checkpoint saved: {len(results)}")
        
        pbar.update(1)
            
    pbar.close()

    if len(results) == 0:
        logger.info("没有抽取到任何有效数据，程序退出。")
        return

    df_res = pd.DataFrame(results)

    # ==========================================
    # ✅ 对所有宏观和视觉原子成分执行绝对平滑的分布校准
    # ==========================================
    logger.info(">>> 执行因果变量的绝对平滑分布校准 (Jittered Rank Norm)...")
    
    cols_to_calibrate = [
        'T_con_mkt', 'T_int_vis', 'T_int_fac',
        'atomic_a_pri', 'atomic_a_gft', 'atomic_a_sub', 'atomic_a_urg', 
        'atomic_a_spec', 'atomic_a_str'
    ]
    
    for col in cols_to_calibrate:
        if col in df_res.columns:
            df_res[f'{col}_calibrated'] = zero_preserved_rank_norm(df_res[col])

    df_res.to_parquet(FINAL_SAVE_PATH, index=False)
    
    logger.info(f"🎉 TEST6 数据提取与平滑校准完成！纯数据已保存在: {FINAL_SAVE_PATH}")
    logger.info(f"🧐 样例输出核对:\n{df_res.iloc[-1].to_dict()}")

if __name__ == "__main__":
    main()
    
    
    #. python /home/xzhe162/wh_workspace/casual/model/MCRE/extract_lmm_multimodal_feature_attention_test6.py