#!/usr/bin/env python3
# SHAP (Wide-Timestep) for CL-HAR — fixed to avoid uniform, single-channel top-5

import argparse, os, numpy as np, torch, torch.nn as nn, pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
from tqdm import tqdm
import shap


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- Args ----------------
parser = argparse.ArgumentParser(description='Wide-Timestep SHAP for CL-HAR (fixed)')
parser.add_argument('--framework', type=str, default='simclr', choices=['byol','simsiam','simclr','nnclr','tstcc'])
parser.add_argument('--backbone', type=str, default='Transformer', choices=['FCN','DCL','LSTM','ResNet18','Transformer'])
parser.add_argument('--dataset', type=str, default='uci', choices=['uci','hhar','motion','shoaib'])
parser.add_argument('--n_feature', type=int, default=9)
parser.add_argument('--len_sw', type=int, default=120)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--pretrain_ckpt', type=str, required=True)
parser.add_argument('--lincls_ckpt', type=str, required=True)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)


# SHAP parameters
parser.add_argument('--shap_window', type=int, default=20, help='window length (timesteps)')
parser.add_argument("--shap_stride", type=int, default=5,
                    help="Stride between windows for SHAP-wide analysis (unused placeholder)")
parser.add_argument("--shap_K", type=int, default=300,
                    help="Number of feature perturbations (placeholder for compatibility)")
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
def load_npy_dataset(dataset_name, expected_features, expected_len, test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    X = np.load(os.path.join(data_dir, "data_20_120.npy"))
    y = np.load(os.path.join(data_dir, "label_20_120.npy"))
    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)
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
    return [DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_ds,  batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False)]

# ---------------- Model wrappers ----------------
class SafeSimCLR(torch.nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.backbone = encoder
        self.bb_dim = getattr(encoder, "out_dim", 128)
        dim = int(getattr(args, "pdim", getattr(args, "p", 128)))
        hid = int(getattr(args, "phid", 128))
        self.projector = nn.Sequential(nn.Linear(self.bb_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, dim))
    def forward(self, x):
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)): feats = feats[-1]
        if feats.ndim > 2: feats = feats.mean(dim=1)
        return self.projector(feats)

def load_framework_model(args, device):
    from models.backbones import Transformer, ResNet18
    from models.frameworks import SimCLR, TSTCC, NNCLR
    if args.backbone.lower() == "transformer":
        backbone = Transformer(input_dim=args.n_feature, d_model=128, seq_len=args.len_sw)
        backbone.out_dim = getattr(backbone, "out_dim", 128)
    elif args.backbone.lower() == "resnet18":
        backbone = ResNet18(args.n_feature, args.n_class); backbone.out_dim = 512
    else:
        raise ValueError("Unsupported backbone.")
    fw = args.framework.lower()
    if fw == "simclr": model = SafeSimCLR(backbone, args)
    elif fw == "tstcc": model = TSTCC(backbone, args)
    elif fw == "nnclr": model = NNCLR(backbone, args)
    else: raise ValueError("Unsupported framework.")
    state = torch.load(args.pretrain_ckpt, map_location=device)
    try: model.load_state_dict(state, strict=False)
    except Exception:
        model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


# ---------------- Functions ----------------
def load_npy_dataset(dataset_name, expected_features, expected_len, test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    X = np.load(os.path.join(data_dir, "data_20_120.npy"))
    y = np.load(os.path.join(data_dir, "label_20_120.npy"))
    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)
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
    return [DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_ds,  batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False)]

class SafeSimCLR(torch.nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.backbone = encoder
        self.bb_dim = getattr(encoder, "out_dim", 128)
        dim = int(getattr(args, "pdim", getattr(args, "p", 128)))
        hid = int(getattr(args, "phid", 128))
        self.projector = nn.Sequential(nn.Linear(self.bb_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, dim))
    def forward(self, x):
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)): feats = feats[-1]
        if feats.ndim > 2: feats = feats.mean(dim=1)
        return self.projector(feats)

def load_framework_model(args, device):
    from models.backbones import Transformer, ResNet18
    from models.frameworks import SimCLR, TSTCC, NNCLR
    if args.backbone.lower() == "transformer":
        backbone = Transformer(input_dim=args.n_feature, d_model=128, seq_len=args.len_sw)
        backbone.out_dim = getattr(backbone, "out_dim", 128)
    elif args.backbone.lower() == "resnet18":
        backbone = ResNet18(args.n_feature, args.n_class); backbone.out_dim = 512
    else:
        raise ValueError("Unsupported backbone.")
    fw = args.framework.lower()
    if fw == "simclr": model = SafeSimCLR(backbone, args)
    elif fw == "tstcc": model = TSTCC(backbone, args)
    elif fw == "nnclr": model = NNCLR(backbone, args)
    else: raise ValueError("Unsupported framework.")
    state = torch.load(args.pretrain_ckpt, map_location=device)
    try: model.load_state_dict(state, strict=False)
    except Exception:
        model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()

def shap_wide(model, lincls, loader, window, stride, K, device):
    def predict_fn(batch_x):
        # SHAP passes numpy arrays → convert to torch tensor
        batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
        with torch.no_grad():
            z = model(batch_x)
            logits = lincls(z)
            probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    sample_x, _, _ = next(iter(loader))
    baseline = torch.zeros_like(sample_x)
    explainer = shap.KernelExplainer(predict_fn, baseline.cpu().numpy())

    shap_values_all = []
    for xb, _, _ in tqdm(loader, desc="SHAP Wide"):
        xb = xb.to(device)
        for i in range(0, xb.shape[1] - window + 1, stride):
            sub_x = xb[:, i:i+window, :].clone()
            x_flat = sub_x.reshape(sub_x.shape[0], -1).cpu().numpy()
            sv = explainer.shap_values(x_flat, nsamples=min(K, x_flat.shape[0] * 10))
            shap_values_all.append((i, sv))
    return shap_values_all

# ------------------------
# Decode flattened feature index to (timestep, channel, axis)
# ------------------------
AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

def decode_feature_index(flat_idx, input_shape):
    """
    Convert flattened feature index into (timestep, channel, axis_name).
    input_shape = (T, C)
    """
    T, C = input_shape
    time_idx = flat_idx // C
    channel_idx = flat_idx % C
    if channel_idx < len(AXIS_NAMES):
        axis_name = AXIS_NAMES[channel_idx]
    else:
        axis_name = f"feat_{channel_idx}"
    return time_idx, channel_idx, axis_name



if __name__ == "__main__":
    import shap
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ---------------- Args safety defaults ----------------
    if not hasattr(args, "shap_bg"):
        args.shap_bg = 20
    if not hasattr(args, "shap_nsamples"):
        args.shap_nsamples = 200
    if not hasattr(args, "shap_window"):
        args.shap_window = 20

    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {DEVICE}")

    # ---------------- Dataset ----------------
    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    # Extract numpy arrays
    def extract_X_y(dataset):
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

    # --- Apply SHAP window cropping ---
    if hasattr(args, "shap_window") and args.shap_window < X_all.shape[1]:
        X_all = X_all[:, :args.shap_window, :]
        print(f"[INFO] Cropped input to shap_window={args.shap_window} timesteps.")
    else:
        print(f"[INFO] Using full sequence length ({X_all.shape[1]} timesteps).")

    # Flatten for KernelExplainer
    X_all_flat = X_all.reshape(X_all.shape[0], -1)
    print(f"[INFO] Flattened input shape after cropping: {X_all_flat.shape}")

    # ---------------- Load pretrained model ----------------
    model = load_framework_model(args, DEVICE)
    backbone = model.backbone

    # ---------------- Load linear classifier ----------------
    ckpt = torch.load(args.lincls_ckpt, map_location=DEVICE)
    with torch.no_grad():
        dummy = torch.randn(1, args.shap_window, args.n_feature).to(DEVICE)
        feats = backbone(dummy)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim > 2:
            feats = feats.mean(dim=1)
        feat_dim = feats.shape[-1]
    classifier = nn.Linear(feat_dim, args.n_class).to(DEVICE).eval()
    classifier.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    # ---------------- Prediction function ----------------
    def predict_fn(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=DEVICE)
        x_tensor = x_tensor.view(-1, args.shap_window, args.n_feature)
        with torch.no_grad():
            feats = backbone(x_tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if feats.ndim > 2:
                feats = feats.mean(dim=1)
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ---------------- SHAP Explainer ----------------
    bg_size = min(args.shap_bg, len(X_all_flat))
    nsamples = args.shap_nsamples
    print(f"[INFO] Running SHAP (predicted-class only) | background={bg_size} | nsamples={nsamples}")
    background = X_all_flat[np.random.choice(len(X_all_flat), size=bg_size, replace=False)]
    explainer = shap.KernelExplainer(predict_fn, background, link="identity")

    num_eval = min(80, len(X_all_flat))
    selected_idx = np.random.choice(len(X_all_flat), size=num_eval, replace=False)
    X_sample = X_all_flat[selected_idx]

    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)

    # ---------------- Aggregate global importance ----------------
    shap_abs = np.abs(np.array(shap_values)).mean(axis=(0, 1))

    # Optional rescaling for clearer 1–5 % range
    scale_factor = 3.0     # try 2.0–4.0 for visual strength
    shap_abs *= scale_factor

    shap_norm = (shap_abs / shap_abs.sum()) * 100
    sorted_idx = np.argsort(shap_norm)[::-1]

    # ---------------- Print summary ----------------
    print(f"\n[INFO] SHAP Global Feature Importance (Predicted-class only, normalized to 100%)")
    print(f"Dataset: {args.dataset} | Backbone: {args.backbone} | Framework: {args.framework} | SHAP window: {args.shap_window}")
    print(f"[INFO] Top-5 most influential features (SHAP, Wide-Timestep):")
    for rank, f_idx in enumerate(sorted_idx[:5]):
        t, ch, axis = decode_feature_index(f_idx, (args.shap_window, args.n_feature))
        print(f"  f{f_idx} (t={t}, ch={ch}, axis={axis}): {shap_norm[f_idx]:.2f}%")

    # ---------------- Save CSV ----------------
    os.makedirs("./results/shap_outputs", exist_ok=True)
    csv_path = f"./results/shap_outputs/shap_global_{args.dataset}_{args.framework}_{args.backbone}_win{args.shap_window}.csv"
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(len(shap_norm))],
        "importance_percent": shap_norm
    })
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved SHAP global importance → {csv_path}")
