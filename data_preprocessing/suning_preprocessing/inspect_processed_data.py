import pandas as pd
import os

# ==========================================
# 0. 路径配置
# ==========================================
BASE_DIR = "/home/xzhe162/wh_workspace/casual/processed_data/suning/"

files = {
    "Y_Behavior": "Y/y_behavior.parquet",
    "User_Confounder": "Confounder_user/confounder_user_matrix.parquet",
    "Item_Base": "item_feature&Confounder_price/item_text_price_matrix.parquet",
    "Item_Reviews": "pure_reviews/user_item_pure_reviews.parquet",
    "Item_Rating": "rating/user_item_rating.parquet"
}

def inspect():
    data_frames = {}
    
    print("="*80)
    print(f"{'表名':<20} | {'行数':<12} | {'列数':<8} | {'关键字段'}")
    print("-"*80)

    for name, rel_path in files.items():
        path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            continue
            
        df = pd.read_parquet(path)
        data_frames[name] = df
        
        # 打印元数据
        cols = ", ".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else "")
        print(f"{name:<20} | {len(df):<12,} | {len(df.columns):<8} | {cols}")

    # ==========================================
    # 1. 随机抽样展示
    # ==========================================
    for name, df in data_frames.items():
        print("\n" + "="*30 + f" [ {name} 随机样本 ] " + "="*30)
        # 随机抽取2条，如果总数不够则取全部
        sample_n = min(len(df), 2)
        display_df = df.sample(sample_n)
        
        # 针对长文本列进行截断显示，防止刷屏
        if 'text_W' in display_df.columns:
            display_df['text_W'] = display_df['text_W'].str[:60] + "..."
        if 'review_text' in display_df.columns:
            display_df['review_text'] = display_df['review_text'].str[:60] + "..."
            
        print(display_df.to_string(index=False))

    # ==========================================
    # 2. 逻辑对齐校验 (核心)
    # ==========================================
    print("\n" + "="*30 + " [ 数据对齐逻辑校验 ] " + "="*30)
    
    y_df = data_frames.get("Y_Behavior")
    user_df = data_frames.get("User_Confounder")
    item_df = data_frames.get("Item_Base")

    if y_df is not None:
        # 校验 User ID 匹配率
        if user_df is not None:
            missing_users = y_df[~y_df['user_id'].isin(user_df['user_id'])]['user_id'].nunique()
            match_rate = (1 - missing_users / y_df['user_id'].nunique()) * 100
            print(f"▶ 用户匹配: 行为表中有 {missing_users} 个用户在特征表中找不到 (匹配率: {match_rate:.2f}%)")
            if missing_users > 0:
                print(f"   样例 ID: {y_df[~y_df['user_id'].isin(user_df['user_id'])]['user_id'].iloc[0]}")

        # 校验 Item ID 匹配率
        if item_df is not None:
            missing_items = y_df[~y_df['item_id'].isin(item_df['item_id'])]['item_id'].nunique()
            match_rate = (1 - missing_items / y_df['item_id'].nunique()) * 100
            print(f"▶ 商品匹配: 行为表中有 {missing_items} 个商品在特征表中找不到 (匹配率: {match_rate:.2f}%)")
            if missing_items > 0:
                print(f"   样例 ID: {y_df[~y_df['item_id'].isin(item_df['item_id'])]['item_id'].iloc[0]}")

    print("="*80)

if __name__ == "__main__":
    inspect()
    
    # python /home/xzhe162/wh_workspace/casual/data_preprocessing/suning_preprocessing/inspect_processed_data.py