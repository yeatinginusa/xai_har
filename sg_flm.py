import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

# ---------------- Reproducibility ----------------
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="SmoothGrad (SG) for FLM (One-Fits-All LIMU-BERT)")
parser.add_argument("--dataset", type=str, default="uci", choices=["uci", "hhar", "motion", "shoaib"])
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=120)
parser.add_argument("--n_class", type=int, default=6)
parser.add_argument("--flm_ckpt", type=str, required=True, help="Path to FLM checkpoint (.pt/.pth)")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)

# SG params
parser.add_argument("--sg_std", type=float, default=0.10, help="Noise std (in normalized units)")
parser.add_argument("--sg_samples", type=int, default=25, help="# noisy samples per input for SG")
parser.add_argument("--topk", type=int, default=5, help="How many top features to print")

args = parser.parse_args()
DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# ---------------- Axis Names ----------------
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

# ---------------- Data ----------------
def load_npy_dataset(dataset_name, expected_features, expected_len,
                     test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    X = np.load(os.path.join(data_dir, "data_20_120.npy"))
    y = np.load(os.path.join(data_dir, "label_20_120.npy"))

    # fix labels to 1D int
    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]
    y = y.astype(np.int64)

    # channel adjust
    C_now = X.shape[-1]
    if C_now < expected_features:
        X = np.pad(X, ((0,0),(0,0),(0, expected_features - C_now)))
    elif C_now > expected_features:
        X = X[:, :, :expected_features]

    # time adjust
    T_now = X.shape[1]
    if T_now < expected_len:
        X = np.pad(X, ((0,0),(0, expected_len - T_now),(0,0)))
    elif T_now > expected_len:
        X = X[:, :expected_len, :]

    # dataset
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

# ---------------- Model (FLM uses LIMUBertModel4Pretrain with hidden=768) ----------------
def load_flm(args, device):
    from models import LIMUBertModel4Pretrain
    class Cfg:
        feature_num = args.n_feature
        hidden = 768     # FLM large hidden
        seq_len = args.len_sw
        n_heads = 4
        n_layers = 2
        dropout = 0.1
    cfg = Cfg()
    print(f"[INFO] Loading FLM checkpoint: {args.flm_ckpt}")
    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    state = torch.load(args.flm_ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # build classifier head with correct feat dim
    with torch.no_grad():
        dummy = torch.randn(1, args.len_sw, args.n_feature).to(device)
        feat_dim = model(dummy).shape[-1]
    classifier = nn.Linear(feat_dim, args.n_class)

    # try to load classifier weights from checkpoint if present
    loaded = False
    try:
        # common naming: "classifier.weight"/"classifier.bias"
        cls_state = {k.replace("classifier.", ""): v
                     for k, v in state.items() if "classifier." in k}
        if cls_state:
            classifier.load_state_dict(cls_state, strict=False)
            loaded = True
    except Exception as e:
        print(f"[WARN] Skipping classifier load due to mismatch: {e}")

    classifier = classifier.to(device).eval()
    if loaded:
        print("[INFO] Classifier head loaded from checkpoint (best-effort).")
    else:
        print("[INFO] Classifier head initialized randomly (no exact match in checkpoint).")
    return model, classifier

# ---------------- SmoothGrad core ----------------
def smoothgrad_attribution(model, classifier, xb, target, sg_std=0.10, sg_samples=25):
    """
    xb: (B, T, C) normalized tensor requiring grad
    target: (B,) target class indices
    Returns: attributions (B, T, C) averaged over samples (abs gradients)
    """
    B, T, C = xb.shape
    # collect grads over noisy samples
    agg = torch.zeros_like(xb)

    for s in range(sg_samples):
        noisy = xb + torch.randn_like(xb) * sg_std
        noisy.requires_grad_(True)

        logits = classifier(model(noisy).mean(dim=1) if model(noisy).ndim == 3 else classifier(model(noisy)))
        # gather logit per target
        selected = logits.gather(1, target.view(-1, 1)).sum()
        grads = torch.autograd.grad(selected, noisy, retain_graph=False, create_graph=False)[0]
        agg += grads.abs()

        # cleanup
        del noisy, logits, selected, grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return agg / float(sg_samples)

# ---------------- Main ----------------
if __name__ == "__main__":
    # cuDNN safety
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading dataset: {args.dataset}")
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, args.n_feature, args.len_sw)
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)

    model, classifier = load_flm(args, DEVICE)
    axis_names = axis_lookup(args.n_feature)

    T, C = args.len_sw, args.n_feature
    MICRO = max(1, args.batch_size // 8)

    all_attr = []
    model.eval()
    classifier.eval()

    with torch.no_grad():
        dummy = torch.randn(1, T, C).to(DEVICE)
        _ = model(dummy)

    print(f"[INFO] Running SmoothGrad: std={args.sg_std}, samples={args.sg_samples}")
    for xb, yb, _ in test_loader:
        # micro-batch to control memory
        for i in range(0, xb.size(0), MICRO):
            x = xb[i:i+MICRO].to(DEVICE)
            y = yb[i:i+MICRO].to(DEVICE)
            target = y.argmax(dim=1) if y.ndim > 1 else y

            # per-channel z-normalization for stability
            stds = x.std(dim=(0,1), keepdim=True) + 1e-8
            x_norm = (x / stds).detach()  # (B,T,C)
            x_norm.requires_grad_(False)

            # SG runs require grad on inputs each time inside function
            attr = smoothgrad_attribution(model, classifier, x_norm, target,
                                          sg_std=args.sg_std, sg_samples=args.sg_samples)
            all_attr.append(attr.detach().cpu().numpy())

            del x, y, target, x_norm, attr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------- Aggregate & Normalize ----------------
    all_attr = np.concatenate(all_attr, axis=0)    # (N, T, C)
    attr_abs = np.abs(all_attr).mean(axis=0)       # (T, C)

    # Per-channel normalization to get percentage within each channel over time,
    # then flatten across (T,C) for a global top-k ranking.
    per_channel_sum = attr_abs.sum(axis=0, keepdims=True) + 1e-12
    attr_norm = 100.0 * (attr_abs / per_channel_sum)   # (T, C) sum over t for each c is 100%

    flat_importance = attr_norm.reshape(-1)            # (T*C,)

    # ---------------- Save ----------------
    os.makedirs("./results/sg_outputs", exist_ok=True)
    out_csv = f'./results/sg_outputs/sg_wide_{args.dataset}_flm.csv'
    rows = [(f"f{i}", i // C, i % C, float(v)) for i, v in enumerate(flat_importance)]
    pd.DataFrame(rows, columns=["Feature", "Timestep", "Channel", "Importance (%)"]).to_csv(out_csv, index=False)
    print(f"[INFO] Saved SmoothGrad results → {out_csv}")

    # ---------------- Print Top-K decoded ----------------
    topk = max(1, args.topk)
    top_idx = np.argsort(flat_importance)[::-1][:topk]
    print(f"\n[INFO] Top {topk} most influential features (SG, FLM, Wide-Timestep):")
    for idx in top_idx:
        t, ch, axis = decode_feature_index(idx, T, C, axis_names)
        print(f"  f{idx} (t={t}, ch={ch}, axis={axis}): {flat_importance[idx]:.2f}%")
