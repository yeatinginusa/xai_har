import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import random
import shap

# ---------------- Reproducibility ----------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- Argparse ----------------
parser = argparse.ArgumentParser(description="SHAP Decoder for FLM (One-Fits-All LIMU-BERT)")
parser.add_argument("--dataset", type=str, default="uci", choices=["uci", "hhar", "motion", "shoaib"])
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_class", type=int, default=6)
parser.add_argument("--flm_ckpt", type=str, required=True, help="Path to FLM checkpoint (.pt/.pth)")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# ---------------- Load Dataset ----------------
def load_npy_dataset(dataset_name, expected_features, expected_len):
    data_dir = f"./data/{dataset_name}"
    X = np.load(os.path.join(data_dir, "data_20_120.npy"))
    y = np.load(os.path.join(data_dir, "label_20_120.npy"))
    if y.ndim > 1:
        y = y.squeeze()
    y = y.astype(np.int64)

    # Adjust shape
    if X.shape[-1] < expected_features:
        X = np.pad(X, ((0, 0), (0, 0), (0, expected_features - X.shape[-1])))
    elif X.shape[-1] > expected_features:
        X = X[:, :, :expected_features]

    if X.shape[1] < expected_len:
        X = np.pad(X, ((0, 0), (0, expected_len - X.shape[1]), (0, 0)))
    elif X.shape[1] > expected_len:
        X = X[:, :expected_len, :]

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long))
    total = len(dataset)
    val_size, test_size = int(0.1 * total), int(0.2 * total)
    train_size = total - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])


def build_loaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader


# ---------------- Axis Names ----------------
AXIS_NAMES_6 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
AXIS_NAMES_9 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                "body_acc_x", "body_acc_y", "body_acc_z"]

def decode_feature_index(flat_idx, T, C, axis_names):
    t = flat_idx // C
    ch = flat_idx % C
    axis = axis_names[ch] if ch < len(axis_names) else f"feat_{ch}"
    return t, ch, axis


# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading dataset: {args.dataset}")

    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # ---------------- Load FLM model ----------------
    from models import LIMUBertModel4Pretrain  # FLM uses the same class
    class Cfg:
        feature_num = args.n_feature
        hidden = 768  # ✅ match FLM hidden dimension
        seq_len = args.len_sw
        n_heads = 4
        n_layers = 2
        dropout = 0.1

    cfg = Cfg()
    print(f"[INFO] Loading FLM checkpoint: {args.flm_ckpt}")
    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    state = torch.load(args.flm_ckpt, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    # ---------------- Classifier head ----------------
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feat_dim = model(dummy).shape[-1]

    classifier = nn.Linear(feat_dim, args.n_class)
    if "classifier.weight" in state:
        try:
            classifier.load_state_dict({
                k.replace("classifier.", ""): v for k, v in state.items()
                if "classifier" in k
            }, strict=False)
        except Exception:
            print("[WARN] Skipping classifier weight loading.")
    classifier = classifier.to(DEVICE).eval()

    # ---------------- Prepare data ----------------
    def extract_X_y(dataset):
        data = [dataset[i] for i in range(len(dataset))]
        X = np.stack([d[0] for d in data])
        y = np.array([d[1] for d in data])
        return X, y

    X_train, _ = extract_X_y(train_ds)
    X_val, _ = extract_X_y(val_ds)
    X_test, _ = extract_X_y(test_ds)
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    print(f"[INFO] Combined dataset shape: {X_all.shape}")

    # ---------------- Normalize ----------------
    X_all_flat = X_all.reshape(X_all.shape[0], -1)
    mean = np.mean(X_all_flat, axis=0, keepdims=True)
    std = np.std(X_all_flat, axis=0, keepdims=True) + 1e-8
    X_all_flat = (X_all_flat - mean) / std
    X_all_flat = np.nan_to_num(X_all_flat)
    print(f"[INFO] Flattened input shape: {X_all_flat.shape}")

    # ---------------- Prediction function ----------------
    def predict_fn(x_numpy):
        x_numpy = np.nan_to_num(x_numpy)
        outputs = []
        bs = 16
        for i in range(0, len(x_numpy), bs):
            xb = torch.tensor(x_numpy[i:i+bs], dtype=torch.float32, device=DEVICE)
            xb = xb.view(-1, args.len_sw, args.n_feature)
            with torch.no_grad():
                feats = model(xb)
                logits = classifier(feats)
                if logits.ndim == 3:
                    logits = logits.mean(dim=1)
            preds = logits.cpu().numpy()
            outputs.append(preds)
        return np.concatenate(outputs, axis=0)

    # ---------------- SHAP Setup ----------------
    args.shap_bg = 20
    args.shap_nsamples = 300
    num_eval = 5
    print(f"[INFO] Running SHAP (bg={args.shap_bg}, nsamples={args.shap_nsamples}, num_eval={num_eval})")

    background = shap.kmeans(X_all_flat, args.shap_bg).data
    explainer = shap.KernelExplainer(predict_fn, background, link="identity")

    selected_idx = np.random.choice(len(X_all_flat), size=num_eval, replace=False)
    X_sample = X_all_flat[selected_idx]

    shap_values = explainer.shap_values(X_sample, nsamples=args.shap_nsamples)
    shap_values = np.nan_to_num(np.array(shap_values))

    # ---------------- Aggregate ----------------
    shap_abs = np.abs(shap_values).mean(axis=(0, 1))
    shap_norm = (shap_abs / shap_abs.sum()) * 100
    sorted_idx = np.argsort(shap_norm)[::-1]

    # ---------------- Decode ----------------
    axis_names = AXIS_NAMES_9 if args.n_feature == 9 else AXIS_NAMES_6
    print("\n[INFO] SHAP Global Feature Importance (FLM, Wide-Timestep)")
    for rank in range(5):
        idx = sorted_idx[rank]
        t, ch, axis = decode_feature_index(idx, args.len_sw, args.n_feature, axis_names)
        print(f"f{idx} (t={t}, ch={ch}, axis={axis}): {shap_norm[idx]:.2f}%")

    # ---------------- Save ----------------
    os.makedirs("./results/shap_outputs", exist_ok=True)

    csv_global = f"./results/shap_outputs/shap_wide_{args.dataset}_flm.csv"
    pd.DataFrame({
        "Feature": [f"f{i}" for i in range(len(shap_norm))],
        "Importance (%)": shap_norm
    }).to_csv(csv_global, index=False)
    print(f"[INFO] Saved SHAP global importance → {csv_global}")

    csv_top5 = f"./results/shap_outputs/shap_wide_top5_{args.dataset}_flm.csv"
    top_rows = []
    for rank in range(5):
        idx = sorted_idx[rank]
        t, ch, axis = decode_feature_index(idx, args.len_sw, args.n_feature, axis_names)
        top_rows.append([rank + 1, f"f{idx}", t, ch, axis, shap_norm[idx]])
    pd.DataFrame(top_rows, columns=["Rank", "Feature", "Timestep", "Channel", "Axis", "Importance (%)"]).to_csv(csv_top5, index=False)
    print(f"[INFO] Saved Top-5 summary → {csv_top5}")
