import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from captum.attr import IntegratedGradients

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --------------------------------------------------------------
# 1. Parser and device setup
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Integrated Gradients for FLM (One-Fits-All LIMU-BERT)")
parser.add_argument("--dataset", type=str, default="uci", choices=["uci", "hhar", "motion", "shoaib"])
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_class", type=int, default=6)
parser.add_argument("--flm_ckpt", type=str, required=True, help="Path to FLM one-fits-all checkpoint (.pth)")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------
# 2. Axis and decode helper
# --------------------------------------------------------------
AXIS_NAMES_6 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
AXIS_NAMES_9 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                "body_acc_x", "body_acc_y", "body_acc_z"]

def decode_feature_index(flat_idx, T, C, axis_names):
    t = flat_idx // C
    ch = flat_idx % C
    axis = axis_names[ch] if ch < len(axis_names) else f"feat_{ch}"
    return t, ch, axis

def axis_lookup(n_feature):
    return AXIS_NAMES_9 if n_feature == 9 else AXIS_NAMES_6


# --------------------------------------------------------------
# 3. Data loading
# --------------------------------------------------------------
def load_npy_dataset(dataset_name, expected_features, expected_len, test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    X = np.load(os.path.join(data_dir, "data_20_120.npy"))
    y = np.load(os.path.join(data_dir, "label_20_120.npy"))

    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)

    # Adjust features
    C_now = X.shape[-1]
    if C_now < expected_features:
        X = np.pad(X, ((0, 0), (0, 0), (0, expected_features - C_now)))
    elif C_now > expected_features:
        X = X[:, :, :expected_features]

    # Adjust timesteps
    T_now = X.shape[1]
    if T_now < expected_len:
        X = np.pad(X, ((0, 0), (0, expected_len - T_now), (0, 0)))
    elif T_now > expected_len:
        X = X[:, :expected_len, :]

    d = np.zeros(len(y), dtype=np.int64)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long),
                            torch.tensor(d, dtype=torch.long))
    total = len(dataset)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


def build_loaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader


# --------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # ---------------- Load FLM model ----------------
    print(f"[INFO] Loading FLM checkpoint: {args.flm_ckpt}")
    from models import LIMUBertModel4Pretrain  # assuming LIMU-BERT backbone used
    class Cfg:
        feature_num = args.n_feature
        hidden = 768
        seq_len = args.len_sw
        n_heads = 4
        n_layers = 2
        dropout = 0.1
    cfg = Cfg()

    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    state = torch.load(args.flm_ckpt, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    # ---------------- Prepare classifier ----------------
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feat_dim = model(dummy).shape[-1]
    classifier = nn.Linear(feat_dim, args.n_class).to(DEVICE).eval()
    if "classifier.weight" in state:
        classifier.load_state_dict({k.replace("classifier.", ""): v
                                    for k, v in state.items() if "classifier" in k}, strict=False)

    # ---------------- IG setup ----------------
    def forward_fn(x):
        feats = model(x)
        if feats.ndim == 3:
            feats = feats.mean(dim=1)
        logits = classifier(feats)
        return logits

    ig = IntegratedGradients(forward_fn)

    MICRO = max(1, args.batch_size // 8)
    all_attr = []

    print("[INFO] Running Integrated Gradients attribution ...")
    for xb, yb, _ in test_loader:
        for i in range(0, xb.size(0), MICRO):
            batch = xb[i:i+MICRO].to(DEVICE)
            labels = yb[i:i+MICRO].to(DEVICE)
            target = labels.argmax(dim=1) if labels.ndim > 1 else labels

            stds = batch.std(dim=(0, 1), keepdim=True) + 1e-8
            batch = (batch / stds).requires_grad_(True)
            baseline = torch.zeros_like(batch)

            attributions = ig.attribute(inputs=batch, baselines=baseline, target=target)
            all_attr.append(attributions.detach().cpu().numpy())

            del batch, labels, baseline, attributions
            torch.cuda.empty_cache()

    # ---------------- Aggregate & Normalize ----------------
    all_attr = np.concatenate(all_attr, axis=0)  # (N, T, C)
    attr_abs = np.abs(all_attr).mean(axis=0)     # (T, C)
    per_channel_sum = attr_abs.sum(axis=0, keepdims=True) + 1e-12
    attr_norm = 100.0 * (attr_abs / per_channel_sum)
    flat_importance = attr_norm.flatten(order="C")

    # ---------------- Save results ----------------
    os.makedirs("./results/ig_outputs", exist_ok=True)
    csv_path = f"./results/ig_outputs/ig_global_{args.dataset}_flm.csv"
    rows = [(f"f{i}", i // args.n_feature, i % args.n_feature, float(v)) for i, v in enumerate(flat_importance)]
    pd.DataFrame(rows, columns=["Feature", "Timestep", "Channel", "Importance (%)"]).to_csv(csv_path, index=False)
    print(f"[INFO] Saved IG global importance → {csv_path}")

    # ---------------- Print Top-5 ----------------
    axis_names = axis_lookup(args.n_feature)
    top_idx = np.argsort(flat_importance)[::-1][:5]
    print(f"\n[INFO] Top 5 most influential features (IG, FLM):")
    for idx in top_idx:
        t, ch, axis = decode_feature_index(idx, args.len_sw, args.n_feature, axis_names)
        print(f"  f{idx} (t={t}, ch={ch}, axis={axis}): {flat_importance[idx]:.2f}%")
