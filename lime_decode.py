import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from lime.lime_tabular import LimeTabularExplainer

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="LIME Decoder for CL-HAR")
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

    
parser.add_argument('--p', type=int, default=128)
parser.add_argument('--phid', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_cls', type=float, default=1e-3)
parser.add_argument('--n_epoch', type=int, default=60)
parser.add_argument('--aug1', type=str, default='t_warp',
                        choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp',
                                 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc',
                                 'p_shift', 'ap_p', 'ap_f'])
parser.add_argument('--aug2', type=str, default='resample',
                        choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp',
                                 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc',
                                 'p_shift', 'ap_p', 'ap_f'])
parser.add_argument('--EMA', type=float, default=0.996)
parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent'])
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=1.0)
parser.add_argument('--temp_unit', type=str, default='tsfm',
                        choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'])
parser.add_argument('--logdir', type=str, default='log/')

args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_lime_explanation(decoded_norm, args, save_path=None):
    # Convert feature IDs and weights
    feature_info = []
    for feat, weight in decoded_norm:
        try:
            feat_idx = int("".join(ch for ch in feat if ch.isdigit()))
            t, ch, axis = decode_feature_index(feat_idx, (args.len_sw, args.n_feature))
            feature_info.append((t, ch, axis, weight))
        except Exception:
            continue

    feature_info = np.array(feature_info, dtype=object)
    t_vals = feature_info[:,0].astype(int)
    ch_vals = feature_info[:,1].astype(int)
    w_vals = feature_info[:,3].astype(float)

    # 1️⃣ Bar plot of top features
    top_feats = sorted(decoded_norm, key=lambda x: abs(x[1]), reverse=True)[:10]
    feats, weights = zip(*top_feats)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=np.arange(len(feats)), y=weights, palette="coolwarm")
    plt.xticks(np.arange(len(feats)), [f for f in feats], rotation=45)
    plt.ylabel("Weight (%)")
    plt.title(f"LIME: Top Feature Contributions ({args.dataset}, {args.framework}, {args.backbone})")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_bar.png", dpi=300)
    plt.show()

    # 2️⃣ Heatmap of temporal importance
    heatmap = np.zeros((args.n_feature, args.len_sw))
    for (t, ch, _, w) in feature_info:
        heatmap[ch, t] += w
    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap, cmap="coolwarm", center=0)
    plt.xlabel("Time step (t)")
    plt.ylabel("Channel (sensor axis index)")
    plt.title(f"LIME Temporal Importance Heatmap ({args.dataset})")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_heatmap.png", dpi=300)
    plt.show()

    # 3️⃣ Aggregated importance per channel
    ch_importance = np.abs(heatmap).sum(axis=1)
    plt.figure(figsize=(6,4))
    sns.barplot(x=np.arange(args.n_feature), y=ch_importance, palette="viridis")
    plt.xlabel("Channel Index")
    plt.ylabel("Total Importance")
    plt.title(f"LIME Aggregated Channel Importance ({args.dataset})")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_channel.png", dpi=300)
    plt.show()


def load_npy_dataset(dataset_name, expected_features, expected_len, test_ratio=0.2, val_ratio=0.1):
    """
    Loads ./data/<dataset_name>/data_20_120.npy and label_20_120.npy,
    pads/truncates to (T=expected_len, C=expected_features) as needed,
    and returns (train_ds, val_ds, test_ds) of (X, y, d) tuples.
    """
    data_dir = os.path.join("./data", dataset_name)
    data_path = os.path.join(data_dir, "data_20_120.npy")
    label_path = os.path.join(data_dir, "label_20_120.npy")

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing .npy files for dataset: {dataset_name} at {data_dir}")

    X = np.load(data_path)
    y = np.load(label_path)
    print(f"[INFO] Loaded raw {dataset_name}: X={X.shape}, y={y.shape}")

    # --- Fix label dims to 1D ints ---
    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)

    # --- Harmonize channel/features ---
    C_now = X.shape[-1]
    if C_now < expected_features:
        diff = expected_features - C_now
        print(f"[WARN] Feature mismatch: expected {expected_features}, got {C_now}. Padding zeros...")
        X = np.pad(X, ((0, 0), (0, 0), (0, diff)), mode='constant')
    elif C_now > expected_features:
        print(f"[WARN] Feature mismatch: expected {expected_features}, got {C_now}. Truncating channels...")
        X = X[:, :, :expected_features]

    # --- Harmonize temporal length ---
    T_now = X.shape[1]
    if T_now < expected_len:
        pad_len = expected_len - T_now
        print(f"[WARN] Seq length mismatch: expected {expected_len}, got {T_now}. Padding zeros...")
        X = np.pad(X, ((0, 0), (0, pad_len), (0, 0)), mode='constant')
    elif T_now > expected_len:
        print(f"[WARN] Seq length mismatch: expected {expected_len}, got {T_now}. Truncating time...")
        X = X[:, :expected_len, :]

    # domain placeholder
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


AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "body_acc_x", "body_acc_y", "body_acc_z"]

def decode_feature_index(flat_idx, input_shape):
    T, C = input_shape[-2], input_shape[-1]
    time_idx = flat_idx // C
    channel_idx = flat_idx % C
    axis_name = AXIS_NAMES[channel_idx] if channel_idx < len(AXIS_NAMES) else f"feat_{channel_idx}"
    return time_idx, channel_idx, axis_name


class FullModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if feats.ndim > 2:
                feats = feats.mean(dim=1)
        logits = self.classifier(feats)
        return torch.softmax(logits, dim=1)


class SafeSimCLR(torch.nn.Module):
    """Clean reimplementation of SimCLR avoiding tuple dim bugs."""
    def __init__(self, encoder, args):
        super().__init__()
        import torch.nn as nn
        import numpy as np

        # Backbone / encoder
        self.encoder = encoder
        self.backbone = encoder   # ✅ alias for compatibility with later code
        self.bb_dim = getattr(encoder, "out_dim", 128)

        # ---- Fix all dims to be pure integers ----
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

        # ---- Simple SimCLR MLP projection head ----
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
    """Load SimCLR, NNCLR, or TSTCC framework with Transformer or ResNet18 backbone."""
    import numpy as np
    import torch
    from models.backbones import Transformer, ResNet18
    from models.frameworks import SimCLR, NNCLR, TSTCC

    # --- Ensure projector dimensions are plain ints ---
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
    print(f"[DEBUG] Projector dims → p: {args.p}, phid: {args.phid}")

    # ----- Load backbone -----
    backbone_name = args.backbone.lower()
    if backbone_name == "transformer":
        backbone = Transformer(
            input_dim=args.n_feature,
            d_model=int(args.p),
            seq_len=args.len_sw
        )
        out_dim = getattr(backbone, "out_dim", int(args.p))
        if isinstance(out_dim, (tuple, list, torch.Size)):
            out_dim = int(np.prod(out_dim))
        backbone.out_dim = int(out_dim)
    elif backbone_name == "resnet18":
        backbone = ResNet18(args.n_feature, args.n_class)
        backbone.out_dim = 512
    else:
        raise ValueError(f"Unsupported backbone '{args.backbone}'. Use one of [Transformer, ResNet18].")
    print(f"[DEBUG] Backbone output dimension (out_dim): {backbone.out_dim}")

    # ✅ Fix tuple dim BEFORE creating the SimCLR/NNCLR/TSTCC object
    # This intercepts the tuple directly and patches it into an int value
    if hasattr(args, "pdim"):
        if isinstance(args.pdim, (tuple, list, torch.Size)):
            args.pdim = int(np.prod(args.pdim))
    if hasattr(args, "p"):
        if isinstance(args.p, (tuple, list, torch.Size)):
            args.p = int(np.prod(args.p))
    if hasattr(args, "phid"):
        if isinstance(args.phid, (tuple, list, torch.Size)):
            args.phid = int(np.prod(args.phid))

    # And — just in case — create a clean integer fallback dim
    clean_dim = int(getattr(args, "pdim", getattr(args, "p", 128)))
    clean_hid = int(getattr(args, "phid", 128))
    args.pdim = clean_dim
    args.p = clean_dim
    args.phid = clean_hid
    print(f"[DEBUG] Cleaned dim for framework init: dim={clean_dim}, hidden={clean_hid}")

    # ----- Load framework -----
    fw = args.framework.lower()
    if fw == "simclr":
        model = SafeSimCLR(backbone, args)
    elif fw == "nnclr":
        model = NNCLR(backbone, args)
    elif fw == "tstcc":
        model = TSTCC(backbone, args)
    else:
        raise ValueError(f"Unsupported framework '{args.framework}'. Use one of [simclr, nnclr, tstcc].")

    # ----- Load pretrained checkpoint -----
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
    # ---------------- Setup ----------------
    args = parser.parse_args()
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # ---------------- Load dataset ----------------
    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    train_loaders, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

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
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    classifier = classifier.to(DEVICE).eval()

    # ---------------- Safe extraction of numpy arrays ----------------
    def extract_X_y(dataset):
        """Handle Subset and base Dataset types."""
        if hasattr(dataset, "dataset"):  # torch.utils.data.Subset
            base = dataset.dataset
            indices = dataset.indices
            if hasattr(base, "X") and hasattr(base, "y"):
                return base.X[indices], base.y[indices]
            else:
                data = [base[i] for i in indices]
                X = np.stack([d[0] for d in data])
                y = np.array([d[1] for d in data])
                return X, y
        elif hasattr(dataset, "X") and hasattr(dataset, "y"):
            return dataset.X, dataset.y
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
            feats = backbone(x_tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if feats.ndim > 2:
                feats = feats.mean(dim=1)
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ---------------- Initialize LIME explainer ----------------
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_all_flat,
        mode="classification",
        feature_names=[f"f{i}" for i in range(X_all_flat.shape[1])],
        discretize_continuous=False,
        random_state=42,
        sample_around_instance=True
    )

    # ---------------- Global LIME importance ----------------
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

    # Normalize and sort
    total = sum(global_importance.values())
    global_importance = {k: (v / total) * 100 for k, v in global_importance.items()}
    sorted_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)

    # ---------------- Save results ----------------
    import os, pandas as pd
    save_dir = "./results/lime_outputs"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"lime_global_{args.dataset}_{args.framework}_{args.backbone}.csv")
    

    lime_df = pd.DataFrame(sorted_global, columns=["Feature", "Normalized_Importance(%)"])
    lime_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved LIME global importance results → {csv_path}")
    
    print(f"Dataset: {args.dataset} | Backbone: {args.backbone} | Framework: {args.framework}")
    print("\n[INFO] Top 5 most influential features:")
    for feat, weight in sorted_global[:5]:
        try:
            feat_idx = int(''.join(ch for ch in feat if ch.isdigit()))
            t, ch, axis = decode_feature_index(feat_idx, (args.len_sw, args.n_feature))
            print(f"  {feat} → (t={t}, ch={ch}, axis={axis}): {weight:.2f}%")
        except Exception:
            print(f"  {feat}: {weight:.2f}%")
        
