import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

plt.rcParams['font.family'] = 'Times New Roman'

# ---------------------- Argument setup ----------------------
parser = argparse.ArgumentParser(description="Visualize interpretability results (LIME, SHAP, IG, SG) for LIMU-BERT or FLM")
parser.add_argument("--lime_csv", type=str, required=True)
parser.add_argument("--shap_csv", type=str, required=True)
parser.add_argument("--ig_csv", type=str, required=True)
parser.add_argument("--sg_csv", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True, choices=["limubert", "flm"], help="Choose which model to visualize")
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_feature", type=int, default=6)
parser.add_argument("--show", action="store_true", help="Show figure interactively")
args = parser.parse_args()


def load_and_clean_csv(path):
    """Load CSV and clean column names and importance values."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)

    colnames = [c.lower() for c in df.columns]

    # Handle presence or absence of headers
    if any("importance" in c for c in colnames):
        imp_col = [c for c in df.columns if "importance" in c.lower()][0]
        feat_col = [c for c in df.columns if "feature" in c.lower()][0]
        df = df[[feat_col, imp_col]]
        df.columns = ["Feature", "Importance"]
    elif df.shape[1] == 2:
        df.columns = ["Feature", "Importance"]
    elif df.shape[1] >= 4:
        df = df.iloc[:, [0, -1]]
        df.columns = ["Feature", "Importance"]
    else:
        raise ValueError(f"Unexpected CSV format in {path}")

    df["Feature"] = df["Feature"].astype(str).str.replace('"', '').str.strip()
    df["Importance"] = (
        df["Importance"].astype(str)
        .str.replace("%", "")
        .str.replace("Importance", "")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.strip()
    )
    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce")
    df = df.dropna(subset=["Importance"])
    return df


def visualize_model(lime_path, shap_path, ig_path, sg_path, dataset, model_name, save_path):
    # Load and clean
    lime_df = load_and_clean_csv(lime_path)
    shap_df = load_and_clean_csv(shap_path)
    ig_df = load_and_clean_csv(ig_path)
    sg_df = load_and_clean_csv(sg_path)

    # Sort and take top features
    for df in [lime_df, shap_df, ig_df, sg_df]:
        df.sort_values(by="Importance", ascending=False, inplace=True, ignore_index=True)

    top_k = min(len(lime_df), len(shap_df), len(ig_df), len(sg_df), 5)
    lime_top, shap_top, ig_top, sg_top = (
        lime_df.head(top_k), shap_df.head(top_k), ig_df.head(top_k), sg_df.head(top_k)
    )

    # --- Plot setup ---
    x = np.arange(top_k)
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

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

    # --- Formatting ---
    ax.set_xticks([])
    ax.set_xlabel("Top Features", fontsize=20, labelpad=8)
    ax.set_ylabel("Normalized Importance (%)", fontsize=20, labelpad=8)
    ax.tick_params(axis='x', labelsize=16, pad=6)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(ncol=4, fontsize=20, frameon=False)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    plt.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.20)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    print(f"[INFO] Saved {model_name.upper()} {dataset.upper()} figure → {save_path}")

    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    save_dir = f"./results/{args.model}_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.dataset}_{args.model}_bar.png")

    visualize_model(
        args.lime_csv, args.shap_csv, args.ig_csv, args.sg_csv,
        args.dataset, args.model, save_path
    )
