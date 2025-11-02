import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from captum.attr import IntegratedGradients, NoiseTunnel

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='SmoothGrad explanation for CL-HAR (wide timestep version)')
parser.add_argument('--framework', type=str, default='simclr',
                    choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc'])
parser.add_argument('--backbone', type=str, default='Transformer',
                    choices=['FCN', 'DCL', 'LSTM', 'ResNet18', 'Transformer'])
parser.add_argument('--dataset', type=str, default='uci',
                    choices=['uci', 'hhar', 'motion', 'shoaib'])
parser.add_argument('--n_feature', type=int, default=9)
parser.add_argument('--len_sw', type=int, default=120)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--pretrain_ckpt', type=str, required=True)
parser.add_argument('--lincls_ckpt', type=str, required=True)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sg_window', type=int, default=20, help='wide timestep window length')
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def load_npy_dataset(dataset_name, expected_features, expected_len, test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    data_path = os.path.join(data_dir, "data_20_120.npy")
    label_path = os.path.join(data_dir, "label_20_120.npy")
    X = np.load(data_path)
    y = np.load(label_path)
    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)
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


def decode_feature_index(flat_idx, input_shape):
    T, C = input_shape[-2], input_shape[-1]
    time_idx = flat_idx // C
    channel_idx = flat_idx % C
    axis_name = AXIS_NAMES[channel_idx] if channel_idx < len(AXIS_NAMES) else f"feat_{channel_idx}"
    return time_idx, channel_idx, axis_name


def load_framework_model(args, device):
    import numpy as np
    from models.backbones import Transformer, ResNet18
    from models.frameworks import SimCLR, TSTCC, NNCLR

    # ---- ensure projector dims are pure ints ----
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
    print(f"[DEBUG] Projector dims → p: {args.p if hasattr(args, 'p') else 'N/A'}")

    # ---- load backbone ----
    if args.backbone.lower() == "transformer":
        from models.backbones import Transformer
        backbone = Transformer(input_dim=args.n_feature, d_model=128, seq_len=args.len_sw)
        backbone.out_dim = getattr(backbone, "out_dim", 128)
    elif args.backbone.lower() == "resnet18":
        from models.backbones import ResNet18
        backbone = ResNet18(args.n_feature, args.n_class)
        backbone.out_dim = 512
    else:
        raise ValueError("Unsupported backbone.")

    # ---- clean int dims for framework ----
    clean_dim = int(getattr(args, "pdim", getattr(args, "p", 128)))
    clean_hid = int(getattr(args, "phid", 128))
    args.pdim = clean_dim
    args.p = clean_dim
    args.phid = clean_hid
    print(f"[DEBUG] Cleaned dim for framework init: dim={clean_dim}, hidden={clean_hid}")

    # ---- choose framework ----
    fw = args.framework.lower()
    if fw == "simclr":
        model = SafeSimCLR(backbone, args)
    elif fw == "tstcc":
        model = TSTCC(backbone, args)
    elif fw == "nnclr":
        model = NNCLR(backbone, args)
    else:
        raise ValueError("Unsupported framework.")

    # ---- load pretrained checkpoint ----
    print(f"[INFO] Loading {args.framework.upper()} pretrained model from: {args.pretrain_ckpt}")
    state = torch.load(args.pretrain_ckpt, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
    model = model.to(device).eval()
    return model


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[INFO] SG window length = {args.sg_window} / Total sequence = {args.len_sw}")

    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)
    model = load_framework_model(args, DEVICE)
    ckpt = torch.load(args.lincls_ckpt, map_location=DEVICE)

    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(DEVICE)
        feats = model.backbone(dummy)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if feats.ndim > 2:
            feats = feats.mean(dim=1)
        feat_dim = feats.shape[-1]

    classifier = nn.Linear(feat_dim, args.n_class).to(DEVICE).eval()
    classifier.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    def forward_fn(x):
        z = model.backbone(x)
        if isinstance(z, (tuple, list)):
            z = z[-1]
        if z.ndim > 2:
            z = z.mean(dim=1)
        return classifier(z)

    ig = IntegratedGradients(forward_fn)
    sg = NoiseTunnel(ig)

    SG_MICROBATCH = max(1, args.batch_size // 8)
    NT_SAMPLES = 10
    N_STEPS = 20
    NOISE_STD = 0.05
    print(f"[INFO] SG micro-batch={SG_MICROBATCH}, nt_samples={NT_SAMPLES}, n_steps={N_STEPS}")

    all_attr = []
    for xb, yb, _ in test_loader:
        for i in range(0, xb.size(0), SG_MICROBATCH):
            batch = xb[i:i+SG_MICROBATCH].to(DEVICE)
            labels = yb[i:i+SG_MICROBATCH].to(DEVICE)
            target = labels.argmax(dim=1) if labels.ndim > 1 else labels

            if args.sg_window < batch.shape[1]:
                batch = batch[:, :args.sg_window, :]

            mean = batch.mean(dim=(0, 1), keepdim=True)
            std = batch.std(dim=(0, 1), keepdim=True) + 1e-8
            batch = ((batch - mean) / std).requires_grad_(True)
            baseline = torch.zeros_like(batch)

            with torch.cuda.amp.autocast(enabled=True):
                attr = sg.attribute(
                    batch,
                    baselines=baseline,
                    target=target,
                    nt_type='smoothgrad',
                    nt_samples=NT_SAMPLES,
                    stdevs=NOISE_STD,
                    n_steps=N_STEPS
                )

            all_attr.append(attr.detach().cpu().numpy())
            del batch, labels, baseline, attr
            torch.cuda.empty_cache()

    # Aggregate & normalize
    all_attr = np.concatenate(all_attr, axis=0)
    attr_abs = np.abs(all_attr).mean(axis=0)
    per_channel_sum = attr_abs.sum(axis=0, keepdims=True) + 1e-12
    attr_rel = attr_abs / per_channel_sum
    channel_weights = per_channel_sum / per_channel_sum.sum()
    attr_norm = 100.0 * (attr_rel * channel_weights)
    flat_importance = attr_norm.flatten(order="C")

    # Save results
    os.makedirs('./results/sg_outputs', exist_ok=True)
    out_csv = f'./results/sg_outputs/sg_wide_dynamic_{args.dataset}_{args.framework}_{args.backbone}_win{args.sg_window}.csv'

    rows = []
    for i, v in enumerate(flat_importance):
        t, ch, axis_name = decode_feature_index(i, (args.sg_window, args.n_feature))
        rows.append([f"f{i}", t, ch, axis_name, float(v)])
    pd.DataFrame(rows, columns=["Feature", "Timestep", "Channel", "Axis", "Importance (%)"]).to_csv(out_csv, index=False)
    print(f"[INFO] Saved SG results → {out_csv}")

    top_idx = np.argsort(flat_importance)[::-1][:5]
    print(f"\nDataset: {args.dataset} | Backbone: {args.backbone} | Framework: {args.framework}")
    print(f"[INFO] Top-5 most influential features (SmoothGrad, dynamically scaled):")
    for idx in top_idx:
        t, ch, axis_name = decode_feature_index(idx, (args.sg_window, args.n_feature))
        print(f"  f{idx} (t={t}, ch={ch}, axis={axis_name}): {flat_importance[idx]:.2f}%")