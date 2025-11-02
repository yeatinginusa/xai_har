import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from matplotlib import rcParams
plt.rcParams['font.family'] = 'Times New Roman'


parser = argparse.ArgumentParser(description="LIME + SHAP + IG + SG Combined Plot Visualizer")
parser.add_argument("--lime_csv", type=str, required=True)
parser.add_argument("--shap_csv", type=str, required=True)
parser.add_argument("--ig_csv", type=str, required=True)
parser.add_argument("--sg_csv", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--framework", type=str, required=True)
parser.add_argument("--backbone", type=str, required=True)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_feature", type=int, default=6)
parser.add_argument("--show", action="store_true", help="Show plots interactively")
args = parser.parse_args()


def clean_importance_df(df):
    """Keep only one numeric importance column and a clean Feature column."""
    df.columns = [c.strip() for c in df.columns]
    # Get the first numeric column dynamically
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # if the CSV has two cols: Feature, Importance (string numeric)
        numeric_cols = [df.columns[-1]]
        df[numeric_cols[0]] = df[numeric_cols[0]].astype(float)
    importance_col = numeric_cols[0]
    cleaned = pd.DataFrame({
        "Feature": df.iloc[:, 0].astype(str).str.replace('"', '').str.strip(),
        "Importance": df[importance_col].astype(float)
    })
    return cleaned


def visualize_all_methods(lime_df, shap_df, ig_df, sg_df, args, save_path="./plot/xai_all", show_plot=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Clean and standardize dataframes
    lime_df = clean_importance_df(lime_df)
    shap_df = clean_importance_df(shap_df)
    ig_df = clean_importance_df(ig_df)
    sg_df = clean_importance_df(sg_df)

    # --- Sort each by its importance
    for df in [lime_df, shap_df, ig_df, sg_df]:
        df.sort_values(by="Importance", ascending=False, inplace=True, ignore_index=True)

    top_k = min(len(lime_df), len(shap_df), len(ig_df), len(sg_df), 5)
    lime_top, shap_top, ig_top, sg_top = (
        lime_df.head(top_k), shap_df.head(top_k), ig_df.head(top_k), sg_df.head(top_k)
    )

    x = np.arange(top_k)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#084E87', '#ef8a00', '#267226', '#BF3F3F']
    hatches = ['//', 'xx', '\\\\', '--']
    labels = ['LIME', 'SHAP', 'IG', 'SG']

    ax.bar(x - 1.5*width, lime_top["Importance"], width, label=labels[0], 
           facecolor='none', hatch=hatches[0], edgecolor=colors[0])
    ax.bar(x - 0.5*width, shap_top["Importance"], width, label=labels[1], 
           facecolor='none', hatch=hatches[1], edgecolor=colors[1])
    ax.bar(x + 0.5*width, ig_top["Importance"], width, label=labels[2], 
           facecolor='none', hatch=hatches[2], edgecolor=colors[2])
    ax.bar(x + 1.5*width, sg_top["Importance"], width, label=labels[3], 
           facecolor='none', hatch=hatches[3], edgecolor=colors[3])

    # --- Formatting
    ax.set_xticks([])
    ax.set_xlabel("Top Features", fontsize=20, labelpad=8)
    ax.set_ylabel("Normalized Importance (%)", fontsize=20, labelpad=8)
    
    # make both x and y tick labels larger
    ax.tick_params(axis='x', labelsize=16, pad=6)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add a bit of headroom
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    # Legend above
    ax.legend(
    ncol=4, fontsize=20,
    frameon=False
)
    ax.set_xlim(-1, top_k)  # ensure equal bar width spacing
    ax.set_ylim(0, 2.2)     # fix y-axis to same range for all models
    fig.set_size_inches(11, 4)  # unify exact figure height and width


    plt.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.20)
    plt.tight_layout()

    plt.savefig(f"{save_path}_bar.png", dpi=400)
    if show_plot:
        plt.show()
    plt.close()
    print(f"[INFO] Saved clean combined plot → {save_path}_bar.png")


if __name__ == "__main__":
    lime_df = pd.read_csv(args.lime_csv)
    shap_df = pd.read_csv(args.shap_csv)
    ig_df = pd.read_csv(args.ig_csv)
    sg_df = pd.read_csv(args.sg_csv)

    save_path = f"./results/lime_shap_ig_sg_plots/{args.dataset}_{args.framework}_{args.backbone}"
    visualize_all_methods(lime_df, shap_df, ig_df, sg_df, args, save_path=save_path, show_plot=args.show)
