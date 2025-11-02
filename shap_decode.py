# shap_decode.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =========================
# Arguments (mirrors lime)
# =========================
parser = argparse.ArgumentParser(description="SHAP Decoder for CL-HAR")

parser.add_argument("--framework", type=str, default="simclr",
                    choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc'])
parser.add_argument("--backbone", type=str, default="Transformer",
                    choices=['FCN', 'DCL', 'LSTM', 'ResNet18', 'Transformer'])
parser.add_argument("--dataset", type=str, default="uci",
                    choices=['uci', 'hhar', 'motion', 'shoaib'])
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_class", type=int, default=6)

parser.add_argument("--pretrain_ckpt", type=str, required=True,
                    help="path to pretrained backbone checkpoint (.pt/.pth)")
parser.add_argument("--lincls_ckpt", type=str, required=True,
                    help="path to trained linear classifier checkpoint")

parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)

# projector dims (kept for compatibility with your loaders)
parser.add_argument('--p', type=int, default=128)
parser.add_argument('--phid', type=int, default=128)

# SHAP options
parser.add_argument('--shap_bg', type=int, default=200,
                    help="number of background samples for KernelExplainer")
parser.add_argument('--shap_nsamples', type=int, default=2000,
                    help="number of samples for SHAP KernelExplainer (trade speed/accuracy)")

args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# =========================
# Data helpers
# =========================
def load_npy_dataset(dataset_name, expected_features, expected_len, test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    data_path = os.path.join(data_dir, "data_20_120.npy")
    label_path = os.path.join(data_dir, "label_20_120.npy")

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing .npy files for dataset: {dataset_name} at {data_dir}")

    X = np.load(data_path)
    y = np.load(label_path)
    print(f"[INFO] Loaded raw {dataset_name}: X={X.shape}, y={y.shape}")

    # fix labels
    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)

    # fix channels
    C_now = X.shape[-1]
    if C_now < expected_features:
        diff = expected_features - C_now
        print(f"[WARN] Feature mismatch: expected {expected_features}, got {C_now}. Padding zeros...")
        X = np.pad(X, ((0, 0), (0, 0), (0, diff)), mode='constant')
    elif C_now > expected_features:
        print(f"[WARN] Feature mismatch: expected {expected_features}, got {C_now}. Truncating channels...")
        X = X[:, :, :expected_features]

    # fix time length
    T_now = X.shape[1]
    if T_now < expected_len:
        pad_len = expected_len - T_now
        print(f"[WARN] Seq length mismatch: expected {expected_len}, got {T_now}. Padding zeros...")
        X = np.pad(X, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
    elif T_now > expected_len:
        print(f"[WARN] Seq length mismatch: expected {expected_len}, got {T_now}. Truncating time...")
        X = X[:, :expected_len, :]

    d = np.zeros(len(y), dtype=np.int64)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    d_tensor = torch.tensor(d, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor, d_tensor)

    total = len(dataset)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    return train_ds, val_ds, test_ds

def build_loaders(train_ds, val_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader

# =========================
# Feature index decoder
# =========================
AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "body_acc_x", "body_acc_y", "body_acc_z"]

def decode_feature_index(flat_idx, input_shape):
    T, C = input_shape[-2], input_shape[-1]
    time_idx = flat_idx // C
    channel_idx = flat_idx % C
    axis_name = AXIS_NAMES[channel_idx] if channel_idx < len(AXIS_NAMES) else f"feat_{channel_idx}"
    return time_idx, channel_idx, axis_name

# =========================
# Safe SimCLR wrapper (same as lime)
# =========================
class SafeSimCLR(torch.nn.Module):
    """Clean reimplementation of SimCLR avoiding tuple dim bugs."""
    def __init__(self, encoder, args):
        super().__init__()
        import torch.nn as nn
        import numpy as np
        self.encoder = encoder
        self.backbone = encoder
        self.bb_dim = getattr(encoder, "out_dim", 128)

        dim = getattr(args, "pdim", getattr(args, "p", 128))
        if isinstance(dim, (tuple, list, torch.Size)):
            dim = int(np.prod(dim))
        elif not isinstance(dim, int):
            dim = int(dim)

        hidden = getattr(args, "phid", 128)
        if isinstance(hidden, (tuple, list, torch.Size)):
            hidden = int(np.prod(hidden))
        elif not isinstance(hidden, int):
            hidden = int(hidden)

        self.projector = nn.Sequential(
            nn.Linear(self.bb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim > 2:
            feats = torch.mean(feats, dim=1)
        z = self.projector(feats)
        return z

def load_framework_model(args, device):
    import numpy as np
    import torch
    from models.backbones import Transformer, ResNet18
    from models.frameworks import NNCLR, TSTCC

    # ensure ints
    for attr in ["p", "phid", "pdim"]:
        val = getattr(args, attr, 128)
        if isinstance(val, (tuple, list, torch.Size)):
            val = int(np.prod(val))
        elif not isinstance(val, int):
            try:
                val = int(val)
            except Exception:
                val = 128
        setattr(args, attr, val)

    # backbone
    bb = args.backbone.lower()
    if bb == "transformer":
        backbone = Transformer(input_dim=args.n_feature, d_model=int(args.p), seq_len=args.len_sw)
        out_dim = getattr(backbone, "out_dim", int(args.p))
        if isinstance(out_dim, (tuple, list, torch.Size)):
            out_dim = int(np.prod(out_dim))
        backbone.out_dim = int(out_dim)
    elif bb == "resnet18":
        backbone = ResNet18(args.n_feature, args.n_class)
        backbone.out_dim = 512
    else:
        raise ValueError(f"Unsupported backbone '{args.backbone}'. Use [Transformer, ResNet18].")

    print(f"[DEBUG] Backbone output dimension (out_dim): {backbone.out_dim}")

    fw = args.framework.lower()
    if fw == "simclr":
        model = SafeSimCLR(backbone, args)
    elif fw == "nnclr":
        model = NNCLR(backbone, args)
    elif fw == "tstcc":
        model = TSTCC(backbone, args)
    else:
        raise ValueError(f"Unsupported framework '{args.framework}'. Use [simclr, nnclr, tstcc].")

    # load checkpoint
    print(f"[INFO] Loading {args.framework.upper()} pretrained model from: {args.pretrain_ckpt}")
    state = torch.load(args.pretrain_ckpt, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        elif "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            raise RuntimeError("Unsupported checkpoint format.")

    model = model.to(device).eval()
    print(f"[INFO] {args.framework.upper()} model loaded successfully.")
    return model

if __name__ == "__main__":

    # ---------------- Load dataset ----------------
    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # ---------------- Load pretrained model ----------------
    model = load_framework_model(args, DEVICE)
    backbone = model.backbone

    # ---------------- Load linear classifier ----------------
    ckpt = torch.load(args.lincls_ckpt, map_location=DEVICE)
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feats = backbone(dummy)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim > 2:
            feats = feats.mean(dim=1)
        feat_dim = feats.shape[-1]
    classifier = nn.Linear(feat_dim, args.n_class)
    classifier.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    classifier = classifier.to(DEVICE).eval()

    # ---------------- Combine datasets ----------------
    def extract_X_y(dataset):
        """Safely extract X, y from torch Dataset or Subset."""
        if hasattr(dataset, "dataset"):
            base = dataset.dataset
            idx = dataset.indices
            data = [base[i] for i in idx]
        else:
            data = [dataset[i] for i in range(len(dataset))]
        X = np.stack([d[0] for d in data])
        y = np.array([d[1] for d in data])
        return X, y

    X_train, _ = extract_X_y(train_ds)
    X_val, _ = extract_X_y(val_ds)
    X_test, _ = extract_X_y(test_ds)
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    print(f"[INFO] Combined dataset shape: {X_all.shape}")

    # ---------------- Flatten (wide timestep) ----------------
    X_all_flat = X_all.reshape(X_all.shape[0], -1)
    mean = np.mean(X_all_flat, axis=0, keepdims=True)
    std = np.std(X_all_flat, axis=0, keepdims=True) + 1e-8
    X_all_flat = (X_all_flat - mean) / std
    print(f"[INFO] Flattened input shape: {X_all_flat.shape}")

    # ---------------- Prediction function (logits for SHAP) ----------------
    def predict_logits(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=DEVICE)
        x_tensor = x_tensor.view(-1, args.len_sw, args.n_feature)
        with torch.no_grad():
            feats = backbone(x_tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if feats.ndim > 2:
                feats = feats.mean(dim=1)
            logits = classifier(feats)
        return logits.cpu().numpy()

    # ---------------- SHAP Explainer (improved wide-timestep) ----------------
    bg_size = min(args.shap_bg, len(X_all_flat))
    nsamples = max(1000, args.shap_nsamples)
    background = X_all_flat[np.random.choice(len(X_all_flat), size=bg_size, replace=False)]
    print(f"[INFO] Running SHAP (wide timestep, improved) with background={bg_size}, nsamples={nsamples}...")

    explainer = shap.KernelExplainer(predict_logits, background, link="logit", l1_reg="num_features(10)")

    num_eval = min(80, len(X_all_flat))
    selected_idx = np.random.choice(len(X_all_flat), size=num_eval, replace=False)
    X_sample = X_all_flat[selected_idx]
    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)

    # ---------------- Aggregate importance ----------------
    shap_abs = np.abs(np.array(shap_values)).mean(axis=(0, 1))
    shap_norm = (shap_abs / shap_abs.sum()) * 100
    sorted_idx = np.argsort(shap_norm)[::-1]

    # ---------------- Print Top-5 ----------------
    print("\n[INFO] SHAP Global Feature Importance Summary (Wide-Timestep, Improved)")
    print(f"Dataset: {args.dataset} | Backbone: {args.backbone} | Framework: {args.framework}")
    for rank in range(5):
        feat_idx = sorted_idx[rank]
        t, ch, axis = decode_feature_index(feat_idx, args.len_sw, args.n_feature)
        print(f"  f{feat_idx} (t={t}, ch={ch}, axis={axis}): {shap_norm[feat_idx]:.2f}%")

    # ---------------- Save CSV outputs ----------------
    os.makedirs("./results/shap_outputs", exist_ok=True)

    csv_global = f"./results/shap_outputs/shap_wide_{args.dataset}_{args.framework}_{args.backbone}.csv"
    pd.DataFrame({
        "Feature": [f"f{i}" for i in range(len(shap_norm))],
        "Importance (%)": shap_norm
    }).to_csv(csv_global, index=False)
    print(f"[INFO] Saved global SHAP-wide importance → {csv_global}")

    csv_detailed = f"./results/shap_outputs/shap_wide_detailed_{args.dataset}_{args.framework}_{args.backbone}.csv"
    rows = []
    for i in range(len(shap_norm)):
        t, ch, axis = decode_feature_index(i, args.len_sw, args.n_feature)
        rows.append([f"f{i}", t, ch, axis, shap_norm[i]])
    pd.DataFrame(rows, columns=["Feature", "Timestep", "Channel", "Axis", "Importance (%)"]).to_csv(csv_detailed, index=False)
    print(f"[INFO] Saved detailed SHAP-wide values → {csv_detailed}")

    top5_path = f"./results/shap_outputs/shap_wide_top5_{args.dataset}_{args.framework}_{args.backbone}.csv"
    top_rows = []
    for rank in range(5):
        feat_idx = sorted_idx[rank]
        t, ch, axis = decode_feature_index(feat_idx, args.len_sw, args.n_feature)
        top_rows.append([rank + 1, f"f{feat_idx}", t, ch, axis, shap_norm[feat_idx]])
    pd.DataFrame(top_rows, columns=["Rank", "Feature", "Timestep", "Channel", "Axis", "Importance (%)"]).to_csv(top5_path, index=False)
    print(f"[INFO] Saved Top-5 SHAP-wide summary → {top5_path}")
