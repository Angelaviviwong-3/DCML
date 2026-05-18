import pandas as pd
import numpy as np
import os
import json
import re
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

# ✅ 引入 vLLM 核心库
from vllm import LLM, SamplingParams

# ==========================================
# 0. 配置（小样本测试模式 test5）
# ==========================================
TEST_MODE = True  # ✅ 开启测试模式
TEST_SIZE = 1000  # ✅ 仅抽取 1000 条

# 根据需要可更换为 "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct" 

BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/"
ITEM_BASE_PATH = os.path.join(BASE_DIR, "item_feature&Confounder_price/item_text_price_matrix.parquet")
ITEM_REVIEWS_PATH = os.path.join(BASE_DIR, "pure_reviews/user_item_pure_reviews.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "picture")

SAVE_DIR = os.path.join(BASE_DIR, "item_multimodal_scalars")
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, "item_multimodal_scalars_test5.parquet")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_scalars_test5.csv")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"lmm_test5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# vLLM 批处理大小（可根据显存调整，如 16, 32, 50）
BATCH_SIZE = 50 

# ==========================================
# 1. 加载 vLLM 模型
# ==========================================
logger.info(f">>> Loading model via vLLM: {MODEL_ID}")
# vLLM 初始化：支持多模态，自动管理显存
llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},  # 限制每条 prompt 一张图
    max_model_len=4096,                # 控制最大上下文，防止 OOM
    gpu_memory_utilization=0.9,        # 显存利用率
)

# 替代原来的 model.generate 参数，n=3 代表 num_return_sequences=3
sampling_params = SamplingParams(
    n=3,
    temperature=0.6,
    top_p=0.9,
    max_tokens=1024
)
logger.info(">>> vLLM Model loaded successfully")


# ==========================================
# 2. 辅助函数：安全提取与 JSON 解析
# ==========================================
def safe_extract(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r'\d*\.?\d+', val)
        if m:
            return float(m.group())
    return 0.0

def parse_vllm_outputs(outputs, item_id):
    """解析 vLLM 返回的 n=3 条生成结果并进行平均"""
    scores = {
        'T_con_mkt': [], 'T_int_vis': [], 'T_int_fac':[],
        'a_pri': [], 'a_gft': [], 'a_sub':[], 'a_urg': [],
        'a_spec': [], 'a_str':[]
    }
    
    # 遍历该条数据的 3 个采样结果
    for output in outputs.outputs:
        res_str = output.text.replace("```json", "").replace("```", "").strip()
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

            s_pri, w_pri = get_sw(mkt, 'a_pri')
            s_gft, w_gft = get_sw(mkt, 'a_gft')
            s_sub, w_sub = get_sw(mkt, 'a_sub')
            s_urg, w_urg = get_sw(mkt, 'a_urg')
            s_spec, w_spec = get_sw(func, 'a_spec')
            s_str, w_str = get_sw(func, 'a_str')
            
            s_fact = safe_extract(func.get('a_text_fact', {}).get('score', 0)) / 10.0

            sum_w_mkt = sum([w_pri, w_gft, w_sub, w_urg])
            sum_w_mkt = sum_w_mkt if sum_w_mkt > 0 else 1.0
            macro_mkt = (s_pri*w_pri + s_gft*w_gft + s_sub*w_sub + s_urg*w_urg) / sum_w_mkt

            sum_w_vis = sum([w_spec, w_str])
            sum_w_vis = sum_w_vis if sum_w_vis > 0 else 1.0
            macro_func_vis = (s_spec*w_spec + s_str*w_str) / sum_w_vis

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

# ==========================================
# 3. 零值保留秩归一化 (Calibration)
# ==========================================
def zero_preserved_rank_norm(x):
    x_arr = np.array(x)
    x_calibrated = np.zeros_like(x_arr, dtype=float)
    mask_pos = (x_arr > 0)
    if mask_pos.sum() > 0:
        ranks = rankdata(x_arr[mask_pos], method='average')
        x_calibrated[mask_pos] = ranks / len(ranks)
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
    
    if TEST_MODE:
        df_input = df_input.head(TEST_SIZE)
        logger.info(f"⚠️ 开启 TEST_MODE，仅处理前 {TEST_SIZE} 条数据")

    if os.path.exists(CHECKPOINT_PATH):
        logger.info(">>> Loading test5 checkpoint...")
        df_ckpt = pd.read_csv(CHECKPOINT_PATH)
        processed_ids = set(df_ckpt['item_id'].astype(str))
        results = df_ckpt.to_dict('records')
    else:
        processed_ids = set()
        results =[]

    logger.info(f">>> 已完成: {len(processed_ids)} / {len(df_input)}")

    # 1. 过滤未处理的合法数据并组装 vLLM messages
    valid_tasks =[]
    
    for _, row in df_input.iterrows():
        item_id = str(row['item_id'])
        if item_id in processed_ids:
            continue

        img_path = os.path.join(IMAGE_DIR, f"{item_id}_1.jpg")
        if not os.path.exists(img_path):
            continue

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
        {row['aggregated_text']}

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
        
        try:
            # vLLM 支持直接传入 PIL.Image 对象
            image = Image.open(img_path).convert("RGB")
            messages = [
                {"role": "user", "content":[
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            valid_tasks.append({"item_id": item_id, "messages": messages})
        except Exception as e:
            logger.error(f"Failed to load image for item {item_id}: {e}")
            continue

    # 2. 批量推理与处理 (Batch Processing)
    pbar = tqdm(total=len(valid_tasks), desc="vLLM Processing")
    
    for i in range(0, len(valid_tasks), BATCH_SIZE):
        batch = valid_tasks[i : i + BATCH_SIZE]
        batch_messages = [task["messages"] for task in batch]
        batch_ids = [task["item_id"] for task in batch]

        # ✅ 调用 vLLM 的 chat 接口进行并行批量推理
        outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)

        # 解析每一条返回
        for output, item_id in zip(outputs, batch_ids):
            scores = parse_vllm_outputs(output, item_id)
            if scores:
                results.append(scores)
                processed_ids.add(item_id)
        
        # 批量保存 Checkpoint
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        pbar.update(len(batch))
        pbar.set_postfix({"Saved": len(results)})

    pbar.close()

    if len(results) == 0:
        logger.info("没有抽取到任何有效数据，程序退出。")
        return

    df_res = pd.DataFrame(results)

    # ==========================================
    # ✅ 对所有宏观和视觉原子成分执行分布校准
    # ==========================================
    logger.info(">>> 执行因果变量的分布校准...")
    cols_to_calibrate =[
        'T_con_mkt', 'T_int_vis', 'T_int_fac',
        'atomic_a_pri', 'atomic_a_gft', 'atomic_a_sub', 'atomic_a_urg', 
        'atomic_a_spec', 'atomic_a_str'
    ]
    
    for col in cols_to_calibrate:
        if col in df_res.columns:
            df_res[f'{col}_calibrated'] = zero_preserved_rank_norm(df_res[col])

    df_res.to_parquet(FINAL_SAVE_PATH, index=False)

    # ==========================================
    # ✅ 分布图 (绘制宏观变量代表)
    # ==========================================
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df_res['T_int_fac'], bins=30, kde=True, color='gray')
    plt.title("Raw: T_int_fac (Factual Text)")

    plt.subplot(1, 3, 2)
    sns.histplot(df_res['T_int_vis'], bins=30, kde=True, color='gray')
    plt.title("Raw: T_int_vis (Functional Image)")

    plt.subplot(1, 3, 3)
    sns.histplot(df_res['T_con_mkt'], bins=30, kde=True, color='gray')
    plt.title("Raw: T_con_mkt (Marketing Visual)")

    plt.tight_layout()
    raw_plot_path = os.path.join(SAVE_DIR, "distribution_raw_test5.png")
    plt.savefig(raw_plot_path)
    plt.close()

    logger.info(f"🎉 TEST5 vLLM 特征提取完成！数据保存在: {FINAL_SAVE_PATH}")
    logger.info(f"🧐 样例输出核对:\n{df_res.iloc[-1].to_dict()}")

if __name__ == "__main__":
    main()