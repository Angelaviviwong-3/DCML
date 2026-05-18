import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
CATE_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Final_CATE_Results_2.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output/Figures")
os.makedirs(SAVE_DIR, exist_ok=True)

def add_error_bars(ax, data, feature_order, hue_order):
    """
    精准映射 Seaborn barplot 的 patches 与实际数据的 Std_Error 并绘制误差线
    """
    n_features = len(feature_order)
    for idx, p in enumerate(ax.patches):
        # 过滤可能因为全0而被忽略的空柱子
        if p.get_height() == 0 and p.get_y() == 0 and p.get_width() == 0: 
            continue
            
        # Seaborn 绘制顺序：先画完第一个 hue 的所有 feature，再画第二个 hue 的所有 feature
        hue_idx = idx // n_features
        feat_idx = idx % n_features
        
        if hue_idx >= len(hue_order): 
            break
        
        current_hue = hue_order[hue_idx]
        current_feat = feature_order[feat_idx]
        
        # 查找数据中对应的 Std_Error
        row = data[(data['Feature'] == current_feat) & (data['Involvement'] == current_hue)]
        if not row.empty:
            # 1.96 * Std_Error 对应 95% 置信区间
            err = row['Std_Error'].values[0] * 1.96
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.errorbar(x, y, yerr=err, fmt='none', c='black', capsize=4, elinewidth=1.2)

def main():
    df = pd.read_csv(CATE_PATH)
    
    # 1. 过滤 involvement 组别 (Low vs High)
    df_plot = df[df['H_level'].isin(['0', '3', 0, 3])].copy()
    df_plot['H_level'] = df_plot['H_level'].astype(int)
    df_plot['Involvement'] = df_plot['H_level'].map({0: 'Low Involvement', 3: 'High Involvement'})
    
    # 2. 定义全部 6+6 个变量的精准映射
    macro_map = {
        'T_con_mkt': 'Mkt Saliency', 'T_con_soc': 'Social Signal', 'T_con_rat': 'Heuristic Rat',
        'T_int_fac': 'Fact Density', 'T_int_vis': 'Vis Utility', 'T_int_sem': 'Sem Align'
    }
    atomic_map = {
        'a_pri': 'Price Shock', 'a_gft': 'Gift Appeal', 'a_sub': 'Subsidy Auth',
        'a_urg': 'Urgency/Disc', 'a_spec': 'Spec Overlay', 'a_str': 'Internal Str'
    }
    
    # 漏斗阶段定义
    stages = ['click', 'cart', 'purchase']
    stage_titles = ['(a) Click Stage', '(b) Add-to-Cart Stage', '(c) Purchase Stage']
    
    # 3. 初始化画布：3 行 (阶段) x 2 列 (Macro/Atomic)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), sharex=False)
    
    palette = {'Low Involvement': '#bdc3c7', 'High Involvement': '#2c3e50'} # 浅灰 vs 深黑
    hue_order = ['Low Involvement', 'High Involvement']
    
    for i, stage in enumerate(stages):
        df_stage = df_plot[df_plot['Stage'] == stage]
        
        # ---------------------------------------------------------
        # 第一列：绘制 6 个 Macro 变量
        # ---------------------------------------------------------
        ax_macro = axes[i, 0]
        d_macro = df_stage[df_stage['Component'].isin(macro_map.keys())].copy()
        d_macro['Feature'] = d_macro['Component'].map(macro_map)
        feature_order_macro = list(macro_map.values())
        
        sns.barplot(x='Feature', y='Coefficient', hue='Involvement', data=d_macro, 
                    palette=palette, order=feature_order_macro, hue_order=hue_order, 
                    edgecolor='black', ax=ax_macro)
        
        add_error_bars(ax_macro, d_macro, feature_order_macro, hue_order)
        
        ax_macro.set_title(f"{stage_titles[i]} - Macro Features (Sys 1 & 2)", fontsize=15, weight='bold', pad=10)
        ax_macro.set_ylabel("CATE", fontsize=13, weight='bold')
        ax_macro.set_xlabel("")
        ax_macro.axhline(0, color='black', linewidth=1)
        ax_macro.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax_macro.get_legend().remove()
        
        # ---------------------------------------------------------
        # 第二列：绘制 6 个 Atomic 变量
        # ---------------------------------------------------------
        ax_atomic = axes[i, 1]
        d_atomic = df_stage[df_stage['Component'].isin(atomic_map.keys())].copy()
        d_atomic['Feature'] = d_atomic['Component'].map(atomic_map)
        feature_order_atomic = list(atomic_map.values())
        
        sns.barplot(x='Feature', y='Coefficient', hue='Involvement', data=d_atomic, 
                    palette=palette, order=feature_order_atomic, hue_order=hue_order, 
                    edgecolor='black', ax=ax_atomic)
        
        add_error_bars(ax_atomic, d_atomic, feature_order_atomic, hue_order)
        
        ax_atomic.set_title(f"{stage_titles[i]} - Atomic Features", fontsize=15, weight='bold', pad=10)
        ax_atomic.set_ylabel("")
        ax_atomic.set_xlabel("")
        ax_atomic.axhline(0, color='black', linewidth=1)
        ax_atomic.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax_atomic.get_legend().remove()
        
        # X轴标签倾斜防重叠，提升可读性
        for ax in [ax_macro, ax_atomic]:
            ax.tick_params(axis='x', rotation=15, labelsize=12)

    # 4. 统一全局图例，放置在图像最下方
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=15, bbox_to_anchor=(0.5, -0.02))
    
    # 调整布局与保存
    plt.tight_layout(pad=2.0)
    pdf_path = os.path.join(SAVE_DIR, "Fig_RQ3_CATE_Moderation_0510.pdf")
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"✅ 全阶段、全变量 (6+6) CATE 卷入度异质性图已成功保存至: {pdf_path}")

if __name__ == "__main__":
    main()
    
    #. python /home/xzhe162/wh_workspace/casual/model/M_suning/DML/plot_2_cate_moderation.py