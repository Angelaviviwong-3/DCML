# ==========================================
# Limit OpenBLAS threads to avoid multithreading-related crashes
# ⚠️ Set these environment variables before all other imports!
# ==========================================

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[3]))
import project_paths as _dcml_paths

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
# 0. Configuration (full extraction mode: full_3)
# ==========================================
TEST_MODE = False  # ✅ Disable test mode for full extraction

MODEL_ID = os.environ.get("DCML_MLLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

BASE_DIR = str(_dcml_paths.project_path('processed_data/suning/')) + "/"
ITEM_BASE_PATH = os.path.join(BASE_DIR, "item_feature&Confounder_price/item_text_price_matrix.parquet")
ITEM_REVIEWS_PATH = os.path.join(BASE_DIR, "pure_reviews/user_item_pure_reviews.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "picture")

# ✅ Output paths (all use the _full_3 suffix)
SAVE_DIR = os.path.join(BASE_DIR, "item_multimodal_scalars")
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, "item_multimodal_scalars_full_3.parquet")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_scalars_full_3.csv")
LOG_DIR = str(_dcml_paths.project_path('log'))

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"lmm_full_3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
# 1. Load the model
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
# 2. Core extraction function (three macro measures and six visual atoms)
# ==========================================
def extract_causal_scalars_hierarchical(item_id, aggregated_text, img_path):

    prompt = f"""
    You are a rigorous and highly critical expert in e-commerce visuals and cognitive psychology. Objectively extract atomic features from product images and text, and assess their visual attention weights.

    [Core rule: no inflated scores (sympathy bias)]
    Use a realistic continuous score distribution without indiscriminately assigning high scores. Follow this pyramid scoring rubric:
    - [0]: The element is entirely absent from the image or text.
    - [1-3]: The element is only mentioned briefly, appears in tiny type, or sits at the image edge. Most products should fall in this range.
    - [4-6]: Ordinary presentation occupying roughly one quarter of the image.
    - [7-8]: Highly prominent, with large text in the central image area.
    - [9-10]: Extremely rare. Do not assign a score above 9 unless the element fills the entire image.

    [Additional restraint for functional and factual features (atomic_functional)]
    The mere presence of specifications does not justify a high score. Reserve high scores for detailed internal-structure cutaways, close-up views of core materials, or comparisons of dozens of parameters. Assign ordinary product photos on a white background a score of 1-2.

    [Task requirements]
    1. Score the intensity of each atomic component from 0 to 10. An absent element must receive 0.
    2. Assign attention weights or salience shares (0.0-1.0) to components within each visual or marketing category. The weights within each category must sum to 1.0.
    3. a_text_fact measures independent factual-text density. Assign only a score from 0 to 10, without a weight.

    [Product text context]:
    {aggregated_text}

    Return exactly the following JSON structure. Do not change keys; they must match the notation in the manuscript:
    {{
        "reasoning": "Critically assess the relative prominence of image and text elements using the pyramid scoring rubric...",
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

                # --- 1. Read scores and weights for all atomic components ---
                s_pri, w_pri = get_sw(mkt, 'a_pri')
                s_gft, w_gft = get_sw(mkt, 'a_gft')
                s_sub, w_sub = get_sw(mkt, 'a_sub')
                s_urg, w_urg = get_sw(mkt, 'a_urg')

                s_spec, w_spec = get_sw(func, 'a_spec')
                s_str, w_str = get_sw(func, 'a_str')

                s_fact = safe_extract(func.get('a_text_fact', {}).get('score', 0)) / 10.0

                # --- 2. Compute macro scores (macro aggregation) ---
                sum_w_mkt = sum([w_pri, w_gft, w_sub, w_urg])
                sum_w_mkt = sum_w_mkt if sum_w_mkt > 0 else 1.0
                macro_mkt = (s_pri*w_pri + s_gft*w_gft + s_sub*w_sub + s_urg*w_urg) / sum_w_mkt

                sum_w_vis = sum([w_spec, w_str])
                sum_w_vis = sum_w_vis if sum_w_vis > 0 else 1.0
                macro_func_vis = (s_spec*w_spec + s_str*w_str) / sum_w_vis

                # --- 3. Populate the result dictionary ---
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
# 3. Zero-preserving rank normalization with jitter
# ==========================================
def zero_preserved_rank_norm(x, seed=2025):
    """
    1. Preserve zero values for the control group.
    2. Add small random jitter of order 1e-6 to break ties.
    3. Use ordinal ranks to assign a unique consecutive rank to each positive value.
    """
    x_arr = np.array(x, dtype=float)
    x_calibrated = np.zeros_like(x_arr)

    mask_pos = (x_arr > 0)
    num_pos = mask_pos.sum()

    if num_pos > 0:
        x_pos = x_arr[mask_pos]

        # Break ties with small jitter and a fixed random seed
        np.random.seed(seed)
        jitter = np.random.uniform(0, 1e-6, size=num_pos)
        x_pos_jittered = x_pos + jitter

        # Combine ordinal ranking with jitter to resolve tied positive values
        ranks = rankdata(x_pos_jittered, method='ordinal')
        x_calibrated[mask_pos] = ranks / float(num_pos)

    return x_calibrated

# ==========================================
# 4. Main workflow
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
        "[Core specifications]\n" + df_input['text_W'] +
        "\n\n[User feedback]\n" + df_input['all_reviews'].fillna("None available")
    )

    # ✅ Truncate input in test mode
    if TEST_MODE:
        TEST_SIZE = 1000
        df_input = df_input.head(TEST_SIZE)
        logger.info(f"⚠️ TEST_MODE enabled: processing only the first {TEST_SIZE} records")

    if os.path.exists(CHECKPOINT_PATH):
        logger.info(">>> Loading full_3 checkpoint...")
        df_ckpt = pd.read_csv(CHECKPOINT_PATH)
        processed_ids = set(df_ckpt['item_id'].astype(str))
        results = df_ckpt.to_dict('records')
    else:
        processed_ids = set()
        results = []

    logger.info(f">>> Completed: {len(processed_ids)} / {len(df_input)}")

    pbar = tqdm(total=len(df_input), desc="Processing Full_3 Dataset")
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

        # ✅ Full extraction: checkpoint every 500 records
        if len(results) % 500 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            logger.info(f"Full_3 Checkpoint saved: {len(results)}")

        pbar.update(1)

    pbar.close()

    if len(results) == 0:
        logger.info("No valid data were extracted. Exiting.")
        return

    df_res = pd.DataFrame(results)

    # ==========================================
    # ✅ Apply distribution calibration to all macro and visual atomic components
    # ==========================================
    logger.info(">>> Calibrating treatment distributions (Jittered Rank Norm)...")

    cols_to_calibrate = [
        'T_con_mkt', 'T_int_vis', 'T_int_fac',
        'atomic_a_pri', 'atomic_a_gft', 'atomic_a_sub', 'atomic_a_urg', 
        'atomic_a_spec', 'atomic_a_str'
    ]

    for col in cols_to_calibrate:
        if col in df_res.columns:
            df_res[f'{col}_calibrated'] = zero_preserved_rank_norm(df_res[col])

    df_res.to_parquet(FINAL_SAVE_PATH, index=False)

    logger.info(f"🎉 FULL_3 extraction and calibration complete! Data saved to: {FINAL_SAVE_PATH}")
    logger.info(f"🧐 Example output:\n{df_res.iloc[-1].to_dict()}")

if __name__ == "__main__":
    main()


    #. python causal/model/MCRE/extract_lmm_multimodal_feature_attention.py
