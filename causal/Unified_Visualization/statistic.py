#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate descriptive statistics tables for the research manuscript.

Reported statistics:

1. Number of users
2. Number of products
3. Number of interactions
4. Density
5. Number of clicks
6. Number of cart events
7. Number of purchases
8. Number of images
9. Number of reviews
10. Mean review length
11. Mean rating

Outputs:
1. dataset_descriptive_statistics.csv
2. dataset_descriptive_statistics.txt
3. dataset_descriptive_statistics_latex.txt

Output directory:
causal/Unified_Visualization/output
"""

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm


# =========================================================
# Output directory
# =========================================================

OUTPUT_DIR = (
    str(_dcml_paths.project_path('Unified_Visualization/output'))
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# Data paths
# =========================================================

DATASETS = {

    "Amazon Appliances": {

        "behavior_path":
            str(_dcml_paths.project_path('processed_data/amazon_appliances/Y/y_behavior.csv')),

        "user_path":
            str(_dcml_paths.project_path('processed_data/amazon_appliances/Confounder_user/confounder_user_matrix.csv')),

        "item_path":
            str(_dcml_paths.project_path('processed_data/amazon_appliances/item_feature&Confounder_price/item_text_price_matrix.csv')),

        "image_dir":
            str(_dcml_paths.project_path('processed_data/amazon_appliances/pictures')),

        "review_dir":
            str(_dcml_paths.project_path('processed_data/amazon_appliances/pure_reviews')),

        "rating_dir":
            str(_dcml_paths.project_path('processed_data/amazon_appliances/rating')),

        "suning_mode": False
    },

    "Amazon Beauty": {

        "behavior_path":
            str(_dcml_paths.project_path('processed_data/amazon_beauty/Y/y_behavior.csv')),

        "user_path":
            str(_dcml_paths.project_path('processed_data/amazon_beauty/Confounder_user/confounder_user_matrix.csv')),

        "item_path":
            str(_dcml_paths.project_path('processed_data/amazon_beauty/item_feature&Confounder_price/item_text_price_matrix.csv')),

        "image_dir":
            str(_dcml_paths.project_path('processed_data/amazon_beauty/pictures')),

        "review_dir":
            str(_dcml_paths.project_path('processed_data/amazon_beauty/pure_reviews')),

        "rating_dir":
            str(_dcml_paths.project_path('processed_data/amazon_beauty/rating')),

        "suning_mode": False
    },

    "Suning": {

        "behavior_path":
            str(_dcml_paths.project_path('processed_data/suning/Y/y_behavior.csv')),

        "user_path":
            str(_dcml_paths.project_path('processed_data/suning/Confounder_user/confounder_user_matrix.csv')),

        "item_path":
            str(_dcml_paths.project_path('processed_data/suning/item_feature&Confounder_price/item_text_price_matrix.csv')),

        "image_dir":
            str(_dcml_paths.project_path('processed_data/suning/picture')),

        "review_dir":
            str(_dcml_paths.project_path('processed_data/suning/pure_reviews')),

        "rating_dir":
            str(_dcml_paths.project_path('processed_data/suning/rating')),

        "suning_mode": True
    }
}


# =========================================================
# Format numeric values
# =========================================================

def format_k(num):

    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"

    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"

    else:
        return str(num)


# =========================================================
# Count users and products
# =========================================================

def analyze_user_item(user_path, item_path):

    user_df = pd.read_csv(user_path)

    item_df = pd.read_csv(item_path)

    user_num = user_df["user_id"].nunique()

    item_num = item_df["item_id"].nunique()

    return user_num, item_num


# =========================================================
# Behavior statistics
# =========================================================

def analyze_behaviors(behavior_path):

    df = pd.read_csv(behavior_path)

    interaction_num = len(df)

    click_num = int(df["y_click"].sum())

    cart_num = int(df["y_cart"].sum())

    purchase_num = int(df["y_purchase"].sum())

    return (
        interaction_num,
        click_num,
        cart_num,
        purchase_num
    )


# =========================================================
# Density
# =========================================================

def compute_density(
        interaction_num,
        user_num,
        item_num
):

    density = (
        interaction_num /
        (user_num * item_num)
    ) * 100

    return density


# =========================================================
# Image statistics
# =========================================================

def count_images(image_dir, suning_mode=False):

    if suning_mode:

        image_paths = glob.glob(
            os.path.join(image_dir, "*_1.jpg")
        )

    else:

        image_paths = glob.glob(
            os.path.join(image_dir, "*.jpg")
        )

    return len(image_paths)


# =========================================================
# Review statistics
# =========================================================

def analyze_reviews(review_dir):

    review_files = glob.glob(
        os.path.join(review_dir, "*")
    )

    review_count = 0

    review_lengths = []

    for file_path in tqdm(
            review_files,
            desc=f"Reviews"
    ):

        try:

            with open(
                    file_path,
                    "r",
                    encoding="utf-8"
            ) as f:

                lines = f.readlines()

            for line in lines:

                text = line.strip()

                if len(text) == 0:
                    continue

                words = text.split()

                review_lengths.append(len(words))

                review_count += 1

        except Exception:
            continue

    avg_review_len = (
        np.mean(review_lengths)
        if len(review_lengths) > 0 else 0
    )

    return (
        review_count,
        round(avg_review_len, 2)
    )


# =========================================================
# Rating statistics
# =========================================================

def analyze_ratings(rating_dir):

    rating_files = glob.glob(
        os.path.join(rating_dir, "*")
    )

    rating_values = []

    rating_count = 0

    for file_path in tqdm(
            rating_files,
            desc=f"Ratings"
    ):

        try:

            if file_path.endswith(".csv"):

                df = pd.read_csv(file_path)

            elif file_path.endswith(".txt"):

                df = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python"
                )

            elif file_path.endswith(".json"):

                df = pd.read_json(
                    file_path,
                    lines=True
                )

            else:
                continue

            rating_count += len(df)

            if "rating" in df.columns:

                vals = pd.to_numeric(
                    df["rating"],
                    errors="coerce"
                ).dropna()

                rating_values.extend(vals.tolist())

        except Exception:
            continue

    avg_rating = (
        np.mean(rating_values)
        if len(rating_values) > 0 else 0
    )

    return (
        rating_count,
        round(avg_rating, 2)
    )


# =========================================================
# Main program
# =========================================================

results = []

for dataset_name, cfg in DATASETS.items():

    print("\n")
    print("=" * 100)
    print(f"Processing: {dataset_name}")

    # =====================================================
    # Users and products
    # =====================================================

    user_num, item_num = analyze_user_item(
        cfg["user_path"],
        cfg["item_path"]
    )

    # =====================================================
    # Behavior statistics
    # =====================================================

    (
        interaction_num,
        click_num,
        cart_num,
        purchase_num
    ) = analyze_behaviors(
        cfg["behavior_path"]
    )

    # =====================================================
    # density
    # =====================================================

    density = compute_density(
        interaction_num,
        user_num,
        item_num
    )

    # =====================================================
    # Images
    # =====================================================

    image_num = count_images(
        cfg["image_dir"],
        cfg["suning_mode"]
    )

    # =====================================================
    # Reviews
    # =====================================================

    review_num, avg_review_len = analyze_reviews(
        cfg["review_dir"]
    )

    # =====================================================
    # rating
    # =====================================================

    rating_num, avg_rating = analyze_ratings(
        cfg["rating_dir"]
    )

    # =====================================================
    # Save
    # =====================================================

    results.append({

        "Dataset": dataset_name,

        "Users": user_num,

        "Items": item_num,

        "Interactions": interaction_num,

        "Density (%)": round(density, 4),

        "Clicks": click_num,

        "Carts": cart_num,

        "Purchases": purchase_num,

        "Images": image_num,

        "Reviews": review_num,

        "Avg_Review_Length": avg_review_len,

        "Avg_Rating": avg_rating
    })


# =========================================================
# DataFrame
# =========================================================

df = pd.DataFrame(results)

# =========================================================
# Save CSV
# =========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "dataset_descriptive_statistics.csv"
)

df.to_csv(csv_path, index=False)

# =========================================================
# Save TXT
# =========================================================

txt_path = os.path.join(
    OUTPUT_DIR,
    "dataset_descriptive_statistics.txt"
)

with open(txt_path, "w", encoding="utf-8") as f:

    f.write("=" * 100 + "\n")
    f.write("DATASET DESCRIPTIVE STATISTICS\n")
    f.write("=" * 100 + "\n\n")

    for _, row in df.iterrows():

        for col in df.columns:

            f.write(f"{col}: {row[col]}\n")

        f.write("\n" + "-" * 60 + "\n\n")

# =========================================================
# Generate the LaTeX table
# =========================================================

latex_path = os.path.join(
    OUTPUT_DIR,
    "dataset_descriptive_statistics_latex.txt"
)

with open(latex_path, "w", encoding="utf-8") as f:

    f.write("\\begin{table*}[htbp]\n")
    f.write("\\centering\n")
    f.write(
        "\\caption{Comprehensive Statistics of the "
        "Multimodal Recommendation Datasets}\n"
    )

    f.write("\\label{tab:dataset_summary}\n")

    f.write("\\resizebox{\\textwidth}{!}{%\n")

    f.write("\\begin{tabular}{lcccccccccc}\n")

    f.write("\\toprule\n")

    f.write(
        "\\multirow{2}{*}{\\textbf{Dataset}} "
        "& \\multicolumn{4}{c}{\\textbf{Interaction Graph}} "
        "& \\multicolumn{3}{c}{\\textbf{Funnel Behaviors}} "
        "& \\multicolumn{3}{c}{\\textbf{Content Statistics}} \\\\\n"
    )

    f.write(
        "\\cmidrule(lr){2-5}"
        "\\cmidrule(lr){6-8}"
        "\\cmidrule(lr){9-11}\n"
    )

    f.write(
        "& \\textbf{Users}"
        "& \\textbf{Items}"
        "& \\textbf{Interactions}"
        "& \\textbf{Density}"

        "& \\textbf{Click}"
        "& \\textbf{Cart}"
        "& \\textbf{Purchase}"

        "& \\textbf{Images}"
        "& \\textbf{Reviews}"
        "& \\textbf{Avg. Review Length} \\\\\n"
    )

    f.write("\\midrule\n")

    for _, row in df.iterrows():

        f.write(

            f"{row['Dataset']} "

            f"& {format_k(row['Users'])} "

            f"& {format_k(row['Items'])} "

            f"& {format_k(row['Interactions'])} "

            f"& {row['Density (%)']:.4f}\\% "

            f"& {format_k(row['Clicks'])} "

            f"& {format_k(row['Carts'])} "

            f"& {format_k(row['Purchases'])} "

            f"& {format_k(row['Images'])} "

            f"& {format_k(row['Reviews'])} "

            f"& {row['Avg_Review_Length']:.2f} \\\\\n"
        )

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}%\n")
    f.write("}\n\n")

    f.write("\\vspace{2mm}\n\n")

    f.write("\\raggedright\n")
    f.write("\\footnotesize\n")

    f.write(
        "\\textit{Note.} "
        "Density is computed as "
        "$\\frac{|\\mathcal{E}|}"
        "{|\\mathcal{U}|\\times|\\mathcal{I}|}"
        "\\times100\\%$, "
        "where "
        "$|\\mathcal{E}|$ "
        "denotes the number of observed interactions. "
        "Avg. Review Length denotes "
        "the average number of words per review.\n"
    )

    f.write("\\end{table*}\n")


# =========================================================
# Output
# =========================================================

print("\n")
print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(df)

print("\nSaved Files:")

print(csv_path)
print(txt_path)
print(latex_path)

print("\nDone.")


# =========================================================
# Usage
# =========================================================

# python causal/Unified_Visualization/statistic.py
