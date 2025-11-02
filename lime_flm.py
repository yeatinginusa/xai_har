import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import random

# ---------------- Random seeds ----------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(description="LIME Explanation for FLM (One-Fits-All LIMU-BERT)")
parser.add_argument("--dataset", type=str, default="uci",
                    choices=["uci", "hhar", "motion", "shoaib"])
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_class", type=int, default=6)
parser.add_argument("--flm_ckpt", type=str, required=True,
                    help="Path to FLM one-fits-all checkpoint (.pth)")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# ---------------- Axis setup ----------------
AXIS_NAMES_6 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
AXIS_NAMES_9 = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                "body_acc_x", "body_acc_y", "body_acc_z"]

def axis_lookup(n_feature):
    return AXIS_NAMES_9 if n_feature == 9 else AXIS_NAMES_6

def decode_feature_index(flat_idx, T, C, axis_names):
    t = flat_idx // C
    ch = flat_idx % C
    axis = axis_names[ch] if ch < len(axis_names) else f"feat_{ch}"
    return t, ch, axis

# ---------------- Dataset loading ----------------
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

    # Adjust time
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


# ---------------- Main ----------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    train_loaders, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # ---------------- Load FLM model ----------------
    print(f"[INFO] Loading FLM checkpoint: {args.flm_ckpt}")
    from models import LIMUBertModel4Pretrain

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

    # ---------------- Build classifier ----------------
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feat_dim = model(dummy).shape[-1]
    classifier = nn.Linear(feat_dim, args.n_class).to(DEVICE).eval()
    if "classifier.weight" in state:
        classifier.load_state_dict({k.replace("classifier.", ""): v
                                    for k, v in state.items() if "classifier" in k}, strict=False)

    # ---------------- Extract dataset ----------------
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
            if logits.ndim == 3:
                logits = logits.mean(dim=1)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ---------------- Run LIME ----------------
    explainer = LimeTabularExplainer(
        X_all_flat,
        mode="classification",
        feature_names=[f"f{i}" for i in range(X_all_flat.shape[1])],
        discretize_continuous=False,
        random_state=42,
        sample_around_instance=True
    )

    global_importance = {}
    num_samples = min(100, len(X_all_flat))  # smaller for FLM speed
    selected_idx = np.random.choice(len(X_all_flat), size=num_samples, replace=False)
    print(f"[INFO] Running LIME on {num_samples} random samples...")

    for i, idx in enumerate(selected_idx):
        instance = X_all_flat[idx]
        exp = explainer.explain_instance(instance, predict_fn, num_features=10, num_samples=3000)
        for feat, weight in exp.as_list():
            global_importance[feat] = global_importance.get(feat, 0) + abs(weight)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")

    # ---------------- Normalize importance ----------------
    total = sum(global_importance.values())
    global_importance = {k: (v / total) * 100 for k, v in global_importance.items()}
    sorted_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)

    # ---------------- Save results ----------------
    save_dir = "./results/lime_outputs"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"lime_global_{args.dataset}_flm.csv")
    pd.DataFrame(sorted_global, columns=["Feature", "Importance(%)"]).to_csv(csv_path, index=False)
    print(f"[INFO] Saved LIME results → {csv_path}")

    # ---------------- Decode Top 5 ----------------
    axis_names = axis_lookup(args.n_feature)
    print("\n[INFO] Top 5 most influential features (LIME, FLM):")
    for feat, weight in sorted_global[:5]:
        idx = int(feat.replace("f", ""))
        t, ch, axis = decode_feature_index(idx, args.len_sw, args.n_feature, axis_names)
        print(f"{feat} (t={t}, ch={ch}, axis={axis}): {weight:.2f}%")
