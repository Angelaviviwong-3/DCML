#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

# ==========================================
# 0. 路径配置 (对接 dataset_builder_amazon_app.py v3 的输出)
# ==========================================
DATASET_NAME = "amazon_appliances"
DATA_DIR = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}/build_dataset"

# 指向带 _2 后缀的全量分片
SET_B_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_DCML_Set_B_3.parquet")
SET_C_PATH = os.path.join(DATA_DIR, f"{DATASET_NAME}_DCML_Set_C_3.parquet")

def inspect_data(df, name):
    print(f"\n{'='*40} [ 检查 {name} ] {'='*40}")
    print(f"🔹 样本行数: {len(df):,}")
    print(f"🔹 特征总列数: {df.shape[1]}")
    
    # 1. 检查是否存在缺失值
    null_counts = df.isnull().sum()
    if null_counts.max() > 0:
        print("🚨 警告：发现缺失值 (NaN)！请核对合并逻辑。")
        print(null_counts[null_counts > 0])
    else:
        print("✅ 缺失值检查: 完美，全表无任何 NaN！")

    # 2. 检查漏斗结果分布 (Outcomes)
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    print(f"\n📊 转化漏斗分布 (Outcomes):")
    for col in Y_cols:
        if col in df.columns:
            rate = df[col].mean() * 100
            print(f"   - {col} 发生率 (转化率): {rate:.2f}%")

    # 3. 检查 12 个处理变量 (Treatments)
    print("\n🔍 12个核心因果处理变量 (Treatments) 统计:")
    
    # A. 9个 LMM 校准变量 (Macro + Atomic)
    calib_t = [c for c in df.columns if c.endswith('_calibrated')]
    # B. 3个 统计/偏好变量
    stat_t = ['T_int_sem', 'T_con_soc', 'T_con_rat']
    
    all_t = calib_t + stat_t
    
    valid_t = [t for t in all_t if t in df.columns]
    
    if len(valid_t) < 12:
        print(f"⚠️ 注意：当前只找到了 {len(valid_t)} 个 Treatment 变量，预期 12 个。")
        missing = [t for t in all_t if t not in df.columns]
        print(f"   缺失列表: {missing}")
        
    if valid_t:
        # 按照名称排序打印，方便一眼看到 max=1.0
        print(df[sorted(valid_t)].describe().loc[['mean', 'std', 'min', 'max']].T)

    # 4. 检查中介变量 (Mediator)
    if 'M_norm' in df.columns:
        print("\n📢 心理中介变量 (Mediator: Subjective Norms) 统计:")
        print(df[['M_norm']].describe().loc[['mean', 'std', 'min', 'max']].T)

    # 5. 检查控制变量与调节变量
    print("\n🧬 控制变量 (Confounders) & 调节变量 (Moderator):")
    c_cols = ['x_pri_log', 'x_pos_freq', 'H_level', 'user_review_count_scaled', 'user_avg_rating_scaled']
    valid_c = [c for c in c_cols if c in df.columns]
    if valid_c:
        print(df[valid_c].describe().loc[['mean', 'std', 'min', 'max']].T)

if __name__ == "__main__":
    print(f">>> 正在加载亚马逊 ({DATASET_NAME}) DCML v2 实验数据集...")
    
    if os.path.exists(SET_B_PATH) and os.path.exists(SET_C_PATH):
        df_b = pd.read_parquet(SET_B_PATH)
        df_c = pd.read_parquet(SET_C_PATH)
        
        inspect_data(df_b, "Set B (训练集)")
        inspect_data(df_c, "Set C (推断集)")
        
        print("\n" + "="*100)
        print("🎉 数据集检查完成！")
        print("请确认：")
        print("1. 所有以 _calibrated 结尾的变量 Max 是否均为 1.0？")
        print("2. T_int_sem 的 Max 是否为 1.0？")
        print("3. 全表是否存在 0 个 NaN？")
        print("\n如果以上三点全部通过，请发送指令，我们将正式开启 Stage II：多任务残差网络训练！")
        print("="*100)
    else:
        print(f"❌ 找不到数据文件。请检查路径或后缀是否为 _2.parquet: \n{SET_B_PATH}")
        
        #.  python /home/xzhe162/wh_workspace/casual/model/M_amazon/MCRE/inspect_dataset_amazon_app.py
        