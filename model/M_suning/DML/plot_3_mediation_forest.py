import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning"
MED_PATH = os.path.join(BASE_DIR, "Final_Causal_Output/Final_Mediation_Results.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Final_Causal_Output/Figures")
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    df = pd.read_csv(MED_PATH)
    
    # 名字映射
    name_map = {'T_con_mkt': 'Mkt Saliency', 'T_con_soc': 'Social Signal', 'T_con_rat': 'Heuristic Rat'}
    df['Cue_Name'] = df['Heuristic_Cue'].map(name_map)
    df['Stage_Label'] = df['Funnel_Stage'].str.capitalize()
    
    # 构建 Y 轴 Label (例如: Mkt Saliency -> Click)
    df['Y_Label'] = df['Cue_Name'] + " \u2192 " + df['Stage_Label']
    
    # 为了画图好看，按特征和阶段排序
    stage_map = {'click': 2, 'cart': 1, 'purchase': 0} # 纵向：purchase 在最上
    df['Stage_Order'] = df['Funnel_Stage'].map(stage_map)
    df = df.sort_values(['Heuristic_Cue', 'Stage_Order'])

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(df))
    
    # 计算误差范围 (左/右差值)
    xerr_lower = df['Indirect_Effect'] - df['95%_CI_Lower']
    xerr_upper = df['95%_CI_Upper'] - df['Indirect_Effect']
    
    # 画 Error bars
    ax.errorbar(
        x=df['Indirect_Effect'], y=y_pos, 
        xerr=[xerr_lower, xerr_upper], 
        fmt='o', color='#c0392b', ecolor='#34495e',
        elinewidth=2, capsize=5, markersize=8
    )

    # 美化
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Y_Label'], fontsize=12)
    ax.set_xlabel("Estimated Indirect Effect via Subjective Norm ($M_{norm}$)", fontsize=13, weight='bold')
    ax.set_title("Mediation Analysis (Bootstrap 95% CI)", fontsize=15, weight='bold', pad=15)
    
    # 根据 Bonferroni 显著性标星号
    for idx, row in df.iterrows():
        sig = "***" if row['P_Value_Bonferroni'] < 0.01 else ("*" if row['Sig_Bonferroni'] == 'Yes' else "ns")
        ax.text(row['95%_CI_Upper'] + 0.0005, y_pos[list(df.index).index(idx)], sig, 
                va='center', color='red' if sig != 'ns' else 'grey', weight='bold')

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    pdf_path = os.path.join(SAVE_DIR, "Fig_RQ3_Mediation_Forest.pdf")
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"✅ 中介效应森林图已保存: {pdf_path}")

if __name__ == "__main__":
    main()
    
     #. python /home/xzhe162/wh_workspace/casual/model/DML/plot_3_mediation_forest.py