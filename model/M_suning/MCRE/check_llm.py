import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 路径配置（对齐 _full_3 版本的输出）
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/"
SAVE_DIR = os.path.join(BASE_DIR, "item_multimodal_scalars")
LOG_DIR = "/home/xzhe162/wh_workspace/casual/log"

os.makedirs(LOG_DIR, exist_ok=True)

# ✅ 输入文件: 自动指向跑完的全量 _full_3.parquet
INPUT_FILE = os.path.join(SAVE_DIR, "item_multimodal_scalars_full_3.parquet")

# ✅ 输出图像路径: 尾缀全部改为 _full_3
RAW_MACRO_PLOT = os.path.join(SAVE_DIR, "dist_01_macro_raw_full_3.png")
CALIB_MACRO_PLOT = os.path.join(SAVE_DIR, "dist_02_macro_calib_full_3.png")
CALIB_MKT_ATOMIC_PLOT = os.path.join(SAVE_DIR, "dist_03_atomic_mkt_calib_full_3.png")
CALIB_VIS_ATOMIC_PLOT = os.path.join(SAVE_DIR, "dist_04_atomic_vis_calib_full_3.png")

# 日志配置
log_filename = f"ecdf_plotting_full_3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 核心绘图函数
# ==========================================
def generate_ecdf_comparison(df, columns, titles, colors, save_path, xlabel_text):
    """
    生成并排的 ECDF 图，支持动态列数适应
    """
    # 设定学术绘图风格
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "black", 
        "grid.linestyle": "--", 
        "grid.alpha": 0.5,
        "font.family": "serif"
    })
    
    n_cols = len(columns)
    plt.figure(figsize=(6 * n_cols, 5))
    
    for i, (col, title, color) in enumerate(zip(columns, titles, colors), 1):
        plt.subplot(1, n_cols, i)
        
        if col in df.columns:
            # 绘制 ECDF 曲线
            sns.ecdfplot(data=df, x=col, color=color, linewidth=2.5)
            
            # 绘制 0 点辅助线，用于视觉确认对照组 (Control Group) 比例
            plt.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
            
            plt.title(title, fontsize=14, pad=15, fontweight='bold')
            plt.xlabel(xlabel_text, fontsize=11)
            
            if i == 1:
                plt.ylabel("Empirical CDF (Cumulative Prob.)", fontsize=11)
            else:
                plt.ylabel("")
                
            plt.xlim(-0.05, 1.05)
            plt.ylim(0, 1.05)
        else:
            logger.warning(f"⚠️ 列 {col} 不存在于数据中！跳过绘制。")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"✅ 图像绘制成功并存至: {save_path}")

# ==========================================
# 2. 主流程
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"❌ 找不到全量数据文件: {INPUT_FILE}。请先运行 full_3 数据抽取脚本！")
        return

    logger.info(f">>> 正在读取全量数据: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    logger.info(f"   - 样本量: {len(df)}")
    logger.info(f"   - 数据字段: {list(df.columns)}")

    # ---- 1. 宏观变量：原始数据 (Raw) ----
    logger.info(">>> 1/4 绘制宏观变量原始 ECDF...")
    cols_raw = ['T_int_fac', 'T_int_vis', 'T_con_mkt']
    titles_raw = ['Raw: Factual Text Density', 'Raw: Functional Visual', 'Raw: Marketing Saliency']
    colors_raw = ['dimgray', 'dimgray', 'dimgray']
    generate_ecdf_comparison(df, cols_raw, titles_raw, colors_raw, RAW_MACRO_PLOT, "Raw LMM Intensity Score")

    # ---- 2. 宏观变量：校准后数据 (Calibrated) ----
    logger.info(">>> 2/4 绘制宏观变量校准后 ECDF...")
    cols_calib = ['T_int_fac_calibrated', 'T_int_vis_calibrated', 'T_con_mkt_calibrated']
    titles_calib = ['Calib: Factual Density', 'Calib: Functional Visual', 'Calib: Marketing Saliency']
    colors_calib = ['#1f77b4', '#2ca02c', '#d62728'] # 经典蓝绿红
    generate_ecdf_comparison(df, cols_calib, titles_calib, colors_calib, CALIB_MACRO_PLOT, "Calibrated Rank Norm Score")

    # ---- 3. 营销原子变量：校准后数据 (Path 2 Heuristics) ----
    logger.info(">>> 3/4 绘制 [营销] 原子变量校准后 ECDF (4 个组件)...")
    cols_mkt_atomic = [
        'atomic_a_pri_calibrated', 'atomic_a_gft_calibrated', 
        'atomic_a_sub_calibrated', 'atomic_a_urg_calibrated'
    ]
    titles_mkt_atomic = [
        'Calib: Price Shock (a_pri)', 'Calib: Gift Appeal (a_gft)', 
        'Calib: Subsidy Auth (a_sub)', 'Calib: Urgency (a_urg)'
    ]
    colors_mkt_atomic = ['#e377c2', '#ff7f0e', '#9467bd', '#8c564b'] # 暖色调为主
    generate_ecdf_comparison(df, cols_mkt_atomic, titles_mkt_atomic, colors_mkt_atomic, CALIB_MKT_ATOMIC_PLOT, "Calibrated Atomic Score")

    # ---- 4. 视觉原子变量：校准后数据 (Path 1 Analytical) ----
    logger.info(">>> 4/4 绘制 [视觉] 原子变量校准后 ECDF (2 个组件)...")
    cols_vis_atomic = ['atomic_a_spec_calibrated', 'atomic_a_str_calibrated']
    titles_vis_atomic = ['Calib: Spec Overlay (a_spec)', 'Calib: Internal Structure (a_str)']
    colors_vis_atomic = ['#17becf', '#bcbd22'] # 冷色/科技感色调
    generate_ecdf_comparison(df, cols_vis_atomic, titles_vis_atomic, colors_vis_atomic, CALIB_VIS_ATOMIC_PLOT, "Calibrated Atomic Score")

    logger.info(f"🎉 绘图任务全部完成！四组图像已保存至 {SAVE_DIR} 目录。")

if __name__ == "__main__":
    main()
    
    
     #. python /home/xzhe162/wh_workspace/casual/model/M_suning/MCRE/check_llm.py