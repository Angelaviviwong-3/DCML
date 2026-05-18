import pandas as pd
import os

# ==========================================
# 0. 路径配置 (对接 dataset_builder_v4.py 的 _4 尾缀)
# ==========================================
DATA_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/build_dataset"
SET_B_PATH = os.path.join(DATA_DIR, "DCML_B_5.parquet")
SET_C_PATH = os.path.join(DATA_DIR, "DCML_C_5.parquet")

def inspect_data(df, name):
    print(f"\n{'='*40} [ 检查 {name} ] {'='*40}")
    print(f"🔹 样本行数: {len(df)}")
    print(f"🔹 特征总列数: {df.shape[1]}")
    
    # 1. 检查是否存在缺失值
    null_counts = df.isnull().sum()
    if null_counts.max() > 0:
        print("🚨 警告：发现缺失值 (NaN)！请检查合并逻辑。")
        print(null_counts[null_counts > 0])
    else:
        print("✅ 缺失值检查: 完美，全表无任何 NaN！")

    # 2. 检查漏斗结果分布 (Outcomes)
    Y_cols = ['y_click', 'y_cart', 'y_purchase']
    print(f"\n📊 转化漏斗分布 (Outcomes):")
    for col in Y_cols:
        rate = df[col].mean() * 100
        print(f"   - {col} 转化率: {rate:.2f}%")

    # 3. 检查处理变量 (Treatments - 6个宏观 + 原子)
    print("\n🔍 处理变量 (Treatments) 统计概览:")
    # A. 校准过的 LMM 变量 (Macro + Atomic)
    calib_t = [c for c in df.columns if c.endswith('_calib')]
    # B. 原始统计变量 (sem, soc, rat)
    stat_t = ['T_int_sem', 'T_con_soc', 'T_con_rat']
    
    all_t = calib_t + stat_t
    print(df[all_t].describe().loc[['mean', 'std', 'min', 'max']].T)

    # 4. 检查中介变量 (Mediator)
    if 'M_norm' in df.columns:
        print("\n📢 中介变量 (Mediator) 统计概览:")
        print(df[['M_norm']].describe().loc[['mean', 'std', 'min', 'max']].T)
    else:
        print("\n❌ 错误：未找到中介变量 M_norm！")

    # 5. 检查控制变量与调节变量
    print("\n🧬 控制变量 (Confounders) & 调节变量 (Moderator):")
    c_cols = ['x_pri_log', 'x_pos_freq', 'H_level']
    print(df[c_cols].describe().loc[['mean', 'std', 'min', 'max']].T)

if __name__ == "__main__":
    print(">>> 正在从数据目录加载 DCML v4 实验数据集...")
    
    if os.path.exists(SET_B_PATH) and os.path.exists(SET_C_PATH):
        df_b = pd.read_parquet(SET_B_PATH)
        df_c = pd.read_parquet(SET_C_PATH)
        
        inspect_data(df_b, "Set B (Nuisance Models Training)")
        inspect_data(df_c, "Set C (Causal Inference Set)")
        
        print("\n" + "="*100)
        print("🎉 v4 数据体检完成！请重点核对 Treatment 列表是否包含你论文承诺的所有原子和宏观变量。")
        print("如果一切正常，请准备运行下一步的 DML 多任务残差计算。")
        print("="*100)
    else:
        print(f"❌ 找不到 v4 数据文件。请确认路径: \n{SET_B_PATH}\n{SET_C_PATH}")
    
    
    #. python /home/xzhe162/wh_workspace/casual/model/M_suning/MCRE/inspect_dataset.py