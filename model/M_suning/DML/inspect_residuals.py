import pandas as pd
import os

# ==========================================
# 0. 路径配置 (对接 dcml_step1_nuisance_models 的输出)
# ==========================================
RES_PATH = "/home/xzhe162/wh_workspace/casual/processed_data/suning/DML_Results/DCML_Residuals_3.parquet"

def main():
    print("="*80)
    print(">>> [DML Residuals Inspection v2] 因果残差质量体检")
    print("="*80)

    if not os.path.exists(RES_PATH):
        print(f"❌ 找不到文件: {RES_PATH}。请确保 Step 1 已经运行完毕！")
        return

    df_res = pd.read_parquet(RES_PATH)

    # 1. 检查缺失值
    null_counts = df_res.isnull().sum()
    if null_counts.max() > 0:
        print("🚨 警告：残差中存在 NaN！")
        print(null_counts[null_counts > 0])
    else:
        print("✅ [1/4] 缺失值检查: 完美，没有任何 NaN！")

    # 2. 检查 Treatment 残差统计量 (包含宏观和原子级)
    # 获取所有以 'res_T' 或是 'res_atomic' 开头的列
    t_cols = [col for col in df_res.columns if col.startswith('res_T') or col.startswith('res_atomic')]
    print("\n📊 [2/4] 治疗变量残差 (Treatment Residuals) 统计特征:")
    print("   * 理论期望: 均值 (mean) 应极度趋近于 0")
    print("-" * 60)
    print(df_res[t_cols].describe().loc[['mean', 'std', 'min', 'max']].T)

    # 3. 检查 Outcome 残差统计量
    y_cols = [col for col in df_res.columns if col.startswith('res_y')]
    print("\n📊 [3/4] 转化结果残差 (Outcome Residuals) 统计特征:")
    print("   * 理论期望: 均值 (mean) 应极度趋近于 0")
    print("-" * 60)
    print(df_res[y_cols].describe().loc[['mean', 'std', 'min', 'max']].T)

    # 4. 检查残差相关性矩阵 (为何需要 Gram-Schmidt)
    print("\n🔗 [4/4] 治疗变量残差间的皮尔逊相关系数 (Correlation Matrix):")
    print("   * 理论期望: 即使剥离了 X，系统1 (营销/从众) 和系统2 (事实/功能) 之间依然存在非 0 相关性。")
    print("   * 这就是论文必须执行 Multivariate Gram-Schmidt Orthogonalization (Eq. 17) 的根本原因！")
    print("-" * 60)
    
    # 为了展示清楚，我们选几个核心宏观变量和原子变量来展示矩阵
    core_cols = [
        'res_T_con_mkt_calib', 'res_T_int_vis_calib', 'res_T_int_fac_calib', 
        'res_atomic_a_pri_calib', 'res_atomic_a_spec_calib'
    ]
    # 确保这些列确实存在于数据中
    core_cols_exist = [col for col in core_cols if col in df_res.columns]
    
    print(df_res[core_cols_exist].corr().round(4))

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
    
    
     #. python /home/xzhe162/wh_workspace/casual/model/M_suning/DML/inspect_residuals.py