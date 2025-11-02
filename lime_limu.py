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
    # ---------------- Setup ----------------
    args = parser.parse_args()
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # ---------------- Load dataset ----------------
    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    train_loaders, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # ---------------- Load LIMU-BERT encoder ----------------
    from models import LIMUBertModel4Pretrain

    class Cfg:
        feature_num = args.n_feature
        hidden = 72          # adjust to match pretrained checkpoint config
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
    backbone = model.transformer

    # ---------------- Load linear classifier ----------------
    print(f"[INFO] Loading classifier checkpoint from: {args.lincls_ckpt}")
    ckpt = torch.load(args.lincls_ckpt, map_location=DEVICE)
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feats = model(dummy)
        feat_dim = feats.shape[-1]

    classifier = nn.Linear(feat_dim, args.n_class)
    if "state_dict" in ckpt:
        classifier.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        classifier.load_state_dict(ckpt, strict=False)
    classifier = classifier.to(DEVICE).eval()

    # ---------------- Extract full dataset ----------------
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

    # ---------------- Flatten + normalize ----------------
    X_all_flat = X_all.reshape(X_all.shape[0], -1)
    mean = np.mean(X_all_flat, axis=0, keepdims=True)
    std = np.std(X_all_flat, axis=0, keepdims=True) + 1e-8
    X_all_flat = (X_all_flat - mean) / std

    # ---------------- Prediction function ----------------
    def predict_fn(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=DEVICE)
        x_tensor = x_tensor.view(-1, args.len_sw, args.n_feature)
        with torch.no_grad():
            feats = model(x_tensor)
            logits = classifier(feats)
            if logits.ndim == 3:  # if model outputs per timestep
                logits = logits.mean(dim=1)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ---------------- Run LIME ----------------
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_all_flat,
        mode="classification",
        feature_names=[f"f{i}" for i in range(X_all_flat.shape[1])],
        discretize_continuous=False,
        random_state=42,
        sample_around_instance=True
    )

    global_importance = {}
    num_samples = min(200, len(X_all_flat))
    selected_idx = np.random.choice(len(X_all_flat), size=num_samples, replace=False)
    print(f"[INFO] Running LIME on {num_samples} randomly selected samples...")

    for i, idx in enumerate(selected_idx):
        instance = X_all_flat[idx]
        exp = explainer.explain_instance(instance, predict_fn, num_features=10, num_samples=5000)
        for feat, weight in exp.as_list():
            global_importance[feat] = global_importance.get(feat, 0) + abs(weight)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")

    total = sum(global_importance.values())
    global_importance = {k: (v / total) * 100 for k, v in global_importance.items()}
    sorted_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)

    # ---------------- Save results ----------------
    save_dir = "./results/lime_outputs"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"lime_global_{args.dataset}_limubert.csv")

    lime_df = pd.DataFrame(sorted_global, columns=["Feature", "Normalized_Importance(%)"])
    lime_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved LIME global importance → {csv_path}")

    # ---------------- Decode Top Features ----------------
    AXIS_NAMES_6 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    AXIS_NAMES_9 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                    "body_acc_x", "body_acc_y", "body_acc_z"]
    AXIS_NAMES = AXIS_NAMES_9 if args.n_feature == 9 else AXIS_NAMES_6

    print("\n[INFO] Top 5 most influential features (decoded):")
    for feat, weight in sorted_global[:5]:
        idx = int(feat.replace("f", ""))
        t, ch, axis = decode_feature_index(idx, (args.len_sw, args.n_feature))
        print(f"{feat} (t={t}, ch={ch}, axis={axis}): {weight:.2f}%")
