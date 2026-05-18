#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

# ==========================================
# 0. 路径配置 (对接刚才运行出的 Final 残差文件)
# ==========================================
DATASET_NAME = "amazon_appliances"
RES_PATH = f"/home/xzhe162/wh_workspace/casual/processed_data/{DATASET_NAME}/DML_Results/{DATASET_NAME}_DML_Residuals_Final_2.parquet"

def main():
    print("="*100)
    print(f">>> [DML Residuals Quality Check] 亚马逊 {DATASET_NAME} 残差矩阵体检报告")
    print("="*100)

    if not os.path.exists(RES_PATH):
        print(f"❌ 找不到文件: {RES_PATH}。请检查 Step 1 运行路径。")
        return

    print(f"正在加载数据 (样本量: 21,152,537)... 请稍候...")
    df_res = pd.read_parquet(RES_PATH)

    # 1. 检查缺失值
    null_counts = df_res.isnull().sum()
    if null_counts.max() > 0:
        print("\n🚨 警告：残差中存在 NaN 值！这可能会导致回归矩阵奇异。")
        print(null_counts[null_counts > 0])
    else:
        print("\n✅ [1/4] 缺失值检查: 完美，全表 2100万行无任何 NaN！")

    # 2. 检查 Treatment 残差统计量 (12个核心处理变量)
    t_res_cols = [col for col in df_res.columns if col.startswith('res_T_') or col.startswith('res_atomic_')]
    print("\n📊 [2/4] 处理变量残差 (Treatment Residuals) 统计分布:")
    print("   * 理论期望: 均值 (mean) 应极度趋近于 0 (例如 e-16 级别)，标准差 (std) 反映了扣除干扰后的净干预方差。")
    print("-" * 90)
    t_stats = df_res[t_res_cols].describe().loc[['mean', 'std', 'min', 'max']].T
    # 增加一列检查均值是否合格 (绝对值是否小于 1e-2)
    t_stats['Mean_Check'] = t_stats['mean'].apply(lambda x: "✅ OK" if abs(x) < 0.01 else "🚨 High Bias")
    print(t_stats)

    # 3. 检查 Outcome 残差统计量 (漏斗三阶段)
    y_res_cols = [col for col in df_res.columns if col.startswith('res_y_')]
    print("\n📊 [3/4] 结果变量残差 (Outcome Residuals) 统计分布:")
    print("   * 理论期望: 均值 (mean) 趋近于 0。")
    print("-" * 90)
    y_stats = df_res[y_res_cols].describe().loc[['mean', 'std', 'min', 'max']].T
    y_stats['Mean_Check'] = y_stats['mean'].apply(lambda x: "✅ OK" if abs(x) < 0.01 else "🚨 High Bias")
    print(y_stats)

    # 4. 核心：跨认知系统残差相关性 (证明 Gram-Schmidt 的必要性)
    print("\n🔗 [4/4] 跨认知系统残差相关性矩阵 (Justify Gram-Schmidt):")
    print("   * 理论动机: 即使去除了混淆变量 X，'系统1(res_T_con)' 与 '系统2(res_T_int)' 往往依然由于数据生成过程高度相关。")
    print("   * 如果相关系数 > 0.1，则必须执行论文 Eq. 17 的多元正交化，否则 System 1 的效应会污染 System 2。")
    print("-" * 90)
    
    # 选取典型的宏观与原子变量进行对角展示
    cross_check_cols = [
        'res_T_con_mkt_calibrated',   # 系统1: 营销视觉
        'res_T_int_vis_calibrated',   # 系统2: 功能视觉
        'res_T_int_fac_calibrated',   # 系统2: 文本事实
        'res_T_con_soc',               # 系统1: 从众信号
        'res_atomic_a_spec_calibrated' # 原子: 参数规格
    ]
    
    # 过滤掉不存在的列名
    valid_cross = [c for c in cross_check_cols if c in df_res.columns]
    if valid_cross:
        corr_matrix = df_res[valid_cross].corr().round(4)
        print(corr_matrix)
    else:
        print("⚠️ 未找到指定的相关性校验列，请核对列名。")

    print("\n" + "="*100)
    print("🎉 残差诊断完成！如果 [Mean_Check] 全部为 OK，且相关矩阵不全为 0，请开启 Stage III：因果估计。")
    print("="*100)

if __name__ == "__main__":
    main()
    
    #. python /home/xzhe162/wh_workspace/casual/model/M_amazon/DML/inspect_residuals_amazon_app.py