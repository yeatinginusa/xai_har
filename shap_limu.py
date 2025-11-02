import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from lime.lime_tabular import LimeTabularExplainer
import random
import pandas as pd

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="LIME Decoder for LIMU-BERT")
parser.add_argument("--dataset", type=str, default="uci",
                    choices=["uci", "hhar", "motion", "shoaib"])
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_class", type=int, default=6)
parser.add_argument("--pretrain_ckpt", type=str, required=True,
                    help="Path to LIMU-BERT pretrained model checkpoint (.pt/.pth)")
parser.add_argument("--lincls_ckpt", type=str, required=True,
                    help="Path to trained linear classifier checkpoint (.pt/.pth)")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

def load_npy_dataset(dataset_name, expected_features, expected_len):
    data_dir = f"./data/{dataset_name}"
    X = np.load(os.path.join(data_dir, "data_20_120.npy"))
    y = np.load(os.path.join(data_dir, "label_20_120.npy"))

    if y.ndim > 1:
        y = y.squeeze()
    y = y.astype(np.int64)

    # Pad or truncate features
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
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader

AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "body_acc_x", "body_acc_y", "body_acc_z"]


def decode_feature_index(flat_idx, input_shape):
    T, C = input_shape[-2], input_shape[-1]
    time_idx = flat_idx // C
    channel_idx = flat_idx % C
    axis_name = AXIS_NAMES[channel_idx] if channel_idx < len(AXIS_NAMES) else f"feat_{channel_idx}"
    return time_idx, channel_idx, axis_name

if __name__ == "__main__":
    import shap

    # ---------------- Parse arguments ----------------
    args = parser.parse_args()

    # ---------------- SHAP configuration ----------------
    args.shap_bg = 20           # fewer background samples
    args.shap_nsamples = 300    # lightweight explanation
    num_eval = 5                # explain 5 samples max

    # ---------------- Setup ----------------
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")

    # ---------------- Load dataset ----------------
    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # ---------------- Load LIMU-BERT encoder ----------------
    from models import LIMUBertModel4Pretrain

    class Cfg:
        feature_num = args.n_feature
        hidden = 72
        seq_len = args.len_sw
        n_heads = 4
        n_layers = 2
        dropout = 0.1

    cfg = Cfg()
    print(f"[INFO] Loading LIMU-BERT pretrained model from: {args.pretrain_ckpt}")
    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    state = torch.load(args.pretrain_ckpt, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    # ---------------- Load classifier ----------------
    print(f"[INFO] Loading classifier checkpoint from: {args.lincls_ckpt}")
    ckpt = torch.load(args.lincls_ckpt, map_location=DEVICE)
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feat_dim = model(dummy).shape[-1]

    classifier = nn.Linear(feat_dim, args.n_class)
    classifier.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    classifier = classifier.to(DEVICE).eval()

    # ---------------- Prepare dataset ----------------
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
    X_all_flat = np.nan_to_num(X_all_flat, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[INFO] Flattened input shape: {X_all_flat.shape}")

    # ---------------- Safe prediction ----------------
    def predict_logits_safe(x_numpy):
        x_numpy = np.nan_to_num(x_numpy, nan=0.0, posinf=0.0, neginf=0.0)
        x_numpy = np.clip(x_numpy, -1e4, 1e4)
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
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            preds = np.clip(preds, -1e4, 1e4)
            outputs.append(preds)
            torch.cuda.empty_cache()

        final_out = np.concatenate(outputs, axis=0)
        final_out = np.nan_to_num(final_out, nan=0.0, posinf=0.0, neginf=0.0)
        final_out = np.clip(final_out, -1e4, 1e4)
        return final_out.astype(np.float64)

    # ---------------- Pre-check for invalid values ----------------
    print("[INFO] Validating model predictions before SHAP...")
    test_preds = predict_logits_safe(X_all_flat[:10])
    if np.any(np.isnan(test_preds)) or np.any(np.isinf(test_preds)):
        print("[WARN] Detected NaN/Inf in model outputs, replaced with 0.0")
        test_preds = np.nan_to_num(test_preds, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- SHAP setup ----------------
    bg_size = min(args.shap_bg, len(X_all_flat))
    nsamples = args.shap_nsamples
    background = shap.kmeans(X_all_flat, bg_size).data

    print(f"[INFO] Running SHAP (bg={bg_size}, nsamples={nsamples}, num_eval={num_eval})...")
    print(f"[INFO] GPU pre-SHAP: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # Using 'identity' link avoids log-domain NaNs
    explainer = shap.KernelExplainer(predict_logits_safe, background, link="identity")

    selected_idx = np.random.choice(len(X_all_flat), size=num_eval, replace=False)
    X_sample = np.nan_to_num(X_all_flat[selected_idx], nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[INFO] Explaining {len(X_sample)} samples...")

    try:
        shap_values = explainer.shap_values(X_sample, nsamples=nsamples)
    except Exception as e:
        print(f"[ERROR] SHAP failed with: {e}")
        print("[INFO] Retrying with smaller nsamples=150, num_eval=3, safe-fallback mode...")
        nsamples = 150
        X_sample = X_sample[:3]
        shap_values = explainer.shap_values(X_sample, nsamples=nsamples)

    shap_values = np.nan_to_num(np.array(shap_values), nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- Aggregate importance ----------------
    shap_abs = np.abs(shap_values).mean(axis=(0, 1))
    shap_norm = (shap_abs / shap_abs.sum()) * 100
    sorted_idx = np.argsort(shap_norm)[::-1]

    # ---------------- Decode ----------------
    AXIS_NAMES_6 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    AXIS_NAMES_9 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                    "body_acc_x", "body_acc_y", "body_acc_z"]
    AXIS_NAMES = AXIS_NAMES_9 if args.n_feature == 9 else AXIS_NAMES_6

    def decode_feature_index(flat_idx, input_shape):
        T, C = input_shape[-2], input_shape[-1]
        t = flat_idx // C
        ch = flat_idx % C
        axis = AXIS_NAMES[ch] if ch < len(AXIS_NAMES) else f"feat_{ch}"
        return t, ch, axis

    # ---------------- Print Top-5 ----------------
    print("\n[INFO] SHAP Global Feature Importance Summary (LIMU-BERT, Wide-Timestep)")
    print(f"Dataset: {args.dataset} | Model: LIMU-BERT")
    for rank in range(5):
        idx = sorted_idx[rank]
        t, ch, axis = decode_feature_index(idx, (args.len_sw, args.n_feature))
        print(f"f{idx} (t={t}, ch={ch}, axis={axis}): {shap_norm[idx]:.2f}%")

    # ---------------- Save CSV ----------------
    os.makedirs("./results/shap_outputs", exist_ok=True)

    csv_global = f"./results/shap_outputs/shap_wide_{args.dataset}_limubert.csv"
    pd.DataFrame({
        "Feature": [f"f{i}" for i in range(len(shap_norm))],
        "Importance (%)": shap_norm
    }).to_csv(csv_global, index=False)
    print(f"[INFO] Saved global SHAP importance → {csv_global}")

    csv_top5 = f"./results/shap_outputs/shap_wide_top5_{args.dataset}_limubert.csv"
    top_rows = []
    for rank in range(5):
        idx = sorted_idx[rank]
        t, ch, axis = decode_feature_index(idx, (args.len_sw, args.n_feature))
        top_rows.append([rank + 1, f"f{idx}", t, ch, axis, shap_norm[idx]])
    pd.DataFrame(top_rows, columns=["Rank", "Feature", "Timestep", "Channel", "Axis", "Importance (%)"]).to_csv(csv_top5, index=False)
    print(f"[INFO] Saved Top-5 SHAP summary → {csv_top5}")
