import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
ATE_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Final_ATE_Results_2.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output/Figures")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    df = pd.read_csv(ATE_PATH)
    
    # 漏斗映射
    stage_map = {'click': 0, 'cart': 1, 'purchase': 2}
    df['Stage_Order'] = df['Stage'].map(stage_map)
    df['Stage_Label'] = df['Stage'].str.capitalize()
    
    # --- 变量与颜色配置 ---
    # 暖色系: System 1 (Conformity)
    sys1_macro = {'T_con_mkt': ('#e74c3c', 'Mkt Saliency'), 'T_con_soc': ('#e67e22', 'Social Signal'), 'T_con_rat': ('#d35400', 'Heuristic Rat')}
    sys1_atomic = {'a_pri': ('#e74c3c', 'Price Shock'), 'a_gft': ('#e67e22', 'Gift Appeal'), 'a_sub': ('#f39c12', 'Subsidy'), 'a_urg': ('#d35400', 'Urgency')}
    
    # 冷色系: System 2 (Interest)
    sys2_macro = {'T_int_fac': ('#2980b9', 'Fact Density'), 'T_int_vis': ('#27ae60', 'Vis Utility'), 'T_int_sem': ('#8e44ad', 'Sem Align')}
    sys2_atomic = {'a_spec': ('#2980b9', 'Spec Overlay'), 'a_str': ('#27ae60', 'Structure View'), 'T_int_fac': ('#8e44ad', 'Fact Density')}

    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    x_pos = [0, 1, 2]
    x_labels = ['Click', 'Cart', 'Purchase']

    def plot_track(ax, res_level, sys1_dict, sys2_dict, title):
        subset = df[df['Resolution'] == res_level]
        
        # 绘制 System 1
        for comp, (color, label) in sys1_dict.items():
            dat = subset[subset['Component'] == comp].sort_values('Stage_Order')
            if len(dat) == 3:
                ax.plot(x_pos, dat['Coefficient'], marker='o', color=color, linewidth=2.5, markersize=8, label=f"Sys 1: {label}")
        
        # 绘制 System 2
        for comp, (color, label) in sys2_dict.items():
            dat = subset[subset['Component'] == comp].sort_values('Stage_Order')
            if len(dat) == 3:
                ax.plot(x_pos, dat['Coefficient'], marker='s', color=color, linewidth=2.5, markersize=8, linestyle='--', label=f"Sys 2: {label}")

        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=13)
        ax.set_title(title, fontsize=15, weight='bold', pad=15)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel("Marginal Causal Effect (ATE)", fontsize=13, weight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.8)

    # 图 A: Macro
    plot_track(axes[0], 'Macro', sys1_macro, sys2_macro, "(a) Macro-Level Multimodal Features")
    # 图 B: Atomic
    plot_track(axes[1], 'Atomic', sys1_atomic, sys2_atomic, "(b) Atomic-Level Information Cues")

    plt.tight_layout()
    pdf_path = os.path.join(SAVE_DIR, "Fig_RQ2_Dual_Track_Reversal.pdf")
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"✅ 全特征因果反转图已保存: {pdf_path}")

if __name__ == "__main__":
    main()
    
  #. python /home/xzhe162/wh_workspace/casual/model/DML/plot_1_dual_track_reversal.py