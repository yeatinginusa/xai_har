#!/usr/bin/env python3
# LIME (Wide-Timestep) for CL-HAR — compatible with your IG setup
# Example:
# python lime_decode.py --framework simclr --backbone Transformer --dataset uci --n_feature 6 --len_sw 120 --n_class 6 \
#   --pretrain_ckpt ./results/pretrain_best_try_scheduler_simclr_pretrain_uci_eps120_lr0.003_bs256_aug1t_warp_aug2negate_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm.pt \
#   --lincls_ckpt ./results/lincls_try_scheduler_simclr_pretrain_uci_best.pt \
#   --lime_window 20 --lime_stride 5 --lime_K 3000 --lime_alpha 1e-2 --lime_sigma 0.5 --batch_size 64

import argparse, os, numpy as np, torch, torch.nn as nn, pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ----------------- Args -----------------
parser = argparse.ArgumentParser(description='Wide-Timestep LIME for CL-HAR')
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

# LIME params
parser.add_argument('--lime_window', type=int, default=20, help='wide timestep length L_w')
parser.add_argument('--lime_stride', type=int, default=5, help='stride S between windows')
parser.add_argument('--lime_K', type=int, default=3000, help='# perturbed samples per instance')
parser.add_argument('--lime_alpha', type=float, default=1e-2, help='ridge penalty')
parser.add_argument('--lime_sigma', type=float, default=0.5, help='locality kernel width')
args = parser.parse_args()

DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# ----------------- Data -----------------
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

# ----------------- Models -----------------
class SafeSimCLR(torch.nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        import numpy as np
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
    import numpy as np
    from models.backbones import Transformer, ResNet18
    from models.frameworks import SimCLR, TSTCC, NNCLR
    for attr in ["p","phid","pdim"]:
        val = getattr(args, attr, 128)
        try: setattr(args, attr, int(val))
        except: setattr(args, attr, 128)
    if args.backbone.lower() == "transformer":
        backbone = Transformer(input_dim=args.n_feature, d_model=128, seq_len=args.len_sw); backbone.out_dim = getattr(backbone,"out_dim",128)
    elif args.backbone.lower() == "resnet18":
        backbone = ResNet18(args.n_feature, args.n_class); backbone.out_dim = 512
    else:
        raise ValueError("Unsupported backbone.")
    fw = args.framework.lower()
    if fw == "simclr": model = SafeSimCLR(backbone, args)
    elif fw == "tstcc": model = TSTCC(backbone, args)
    elif fw == "nnclr": model = NNCLR(backbone, args)
    else: raise ValueError("Unsupported framework.")
    print(f"[INFO] Loading {args.framework.upper()} pretrained model from: {args.pretrain_ckpt}")
    state = torch.load(args.pretrain_ckpt, map_location=device)
    try: model.load_state_dict(state, strict=False)
    except Exception:
        model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()

# ----------------- LIME utils -----------------
def build_segments(T, C, L_w, stride):
    """Return list of segments: each = (t_start, t_end, channel)"""
    segs = []
    for c in range(C):
        for t0 in range(0, T, stride):
            t1 = min(t0 + L_w, T)
            segs.append((t0, t1, c))
    return segs  # length M

def masks_from_binary(z, segs, T, C):
    """Turn a binary vector z (keep=1, mask=0 for each segment) into a mask matrix [T,C]."""
    M = len(segs)
    mask = np.ones((T, C), dtype=np.float32)
    for m in range(M):
        if z[m] == 0:
            t0, t1, c = segs[m]
            mask[t0:t1, c] = 0.0
    return mask

def kernel_weight(z, sigma):
    """Locality kernel on fraction-masked distance."""
    # distance = fraction of segments masked in z^c (complement)
    frac_masked = (z.size - z.sum()) / max(1, z.size)
    return np.exp(-(frac_masked ** 2) / (sigma ** 2))

def fit_weighted_ridge(Z, y, w, alpha):
    """
    Solve (Z^T W Z + alpha I) beta = Z^T W y
    Z: (K,M), y: (K,), w: (K,)
    """
    W = np.diag(w)
    A = Z.T @ W @ Z + alpha * np.eye(Z.shape[1])
    b = Z.T @ W @ y
    beta = np.linalg.solve(A, b)
    return beta  # (M,)

if __name__ == "__main__":
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # ---------------- Setup ----------------
    T, C = args.len_sw, args.n_feature
    W, S, K = args.shap_window, args.shap_stride, args.shap_K
    alpha = 1e-2

    # ---------------- Load dataset + model ----------------
    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset, C, T)
    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds, args.batch_size)
    model = load_framework_model(args, DEVICE)

    ckpt = torch.load(args.lincls_ckpt, map_location=DEVICE)
    lincls = torch.nn.Linear(model.bb_dim, args.n_class)
    lincls.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    lincls = lincls.to(DEVICE).eval()

    @torch.no_grad()
    def forward_logits(x):
        z = model(x)
        return lincls(z)

    # ---------------- Segments ----------------
    segs = [(t0, min(t0 + W, T), ch) for t0 in range(0, T, S) for ch in range(C)]
    M = len(segs)

    # ---------------- Background ----------------
    xb_all = []
    for xb, _, _ in test_loader:
        xb_all.append(xb)
        if len(xb_all) * xb.shape[0] >= 64:
            break
    xb = torch.cat(xb_all, dim=0).to(DEVICE)
    bg = xb.mean(dim=0, keepdim=True)

    # ---------------- Helper functions ----------------
    def sample_Z(K, M):
        Z = np.ones((K, M), dtype=np.float32)
        for k in range(K):
            off = np.random.choice(M, size=np.random.randint(1, max(2, M // 8)), replace=False)
            Z[k, off] = 0.0
        return Z

    def make_masked_batch(x0, Z):
        Kz = Z.shape[0]
        mask = torch.ones(Kz, T, C, device=DEVICE)
        for m, (t0, t1, ch) in enumerate(segs):
            mask[Z[:, m] == 0, t0:t1, ch] = 0.0
        x_bg = bg.expand_as(x0.unsqueeze(0))
        x0K = x0.unsqueeze(0).expand(Kz, -1, -1)
        return mask * x0K + (1 - mask) * x_bg

    def kernel_weights(Z):
        m = Z.shape[1]
        frac = (m - Z.sum(axis=1)) / max(1, m)
        return np.exp(-(frac ** 2) / (0.25 ** 2)) + 1e-8

    def fit_weighted_ridge(Z, y, w, alpha):
        W = np.diag(w)
        A = Z.T @ W @ Z + alpha * np.eye(Z.shape[1])
        b = Z.T @ W @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    # ---------------- SHAP-like attribution ----------------
    total_attr = np.zeros((T, C))
    n_used = 0
    MAX_INST = 32  # use fewer for speed

    for xb_i in xb[:MAX_INST]:
        x0 = xb_i.unsqueeze(0)
        base = forward_logits(x0)
        tgt = base.argmax(dim=1).item()
        y0 = base[0, tgt].item()

        Z = sample_Z(K, M)
        masked = make_masked_batch(x0.squeeze(0), Z)
        logits = forward_logits(masked)
        y = y0 - logits[:, tgt].cpu().numpy()
        w = kernel_weights(Z)
        beta = fit_weighted_ridge(Z, y, w, alpha)

        per_tc = np.zeros((T, C))
        for m, (t0, t1, ch) in enumerate(segs):
            per_tc[t0:t1, ch] += beta[m] / max(1, (t1 - t0))
        total_attr += np.abs(per_tc)
        n_used += 1

    # ---------------- Normalize importance ----------------
    avg_attr = total_attr / max(1, n_used)
    attr = np.power(avg_attr + 1e-12, 0.8)  # concave scaling for ~1–2 %
    attr = 100.0 * attr / (attr.sum() + 1e-12)

    # Flatten in correct PyTorch (C-order)
    flat = attr.flatten(order="C")

    # ---------------- Save results ----------------
    os.makedirs("./results/shap_outputs", exist_ok=True)
    out_csv = f"./results/shap_outputs/shap_final_{args.dataset}_{args.framework}_{args.backbone}_W{W}_S{S}_K{K}.csv"
    rows = []
    for i in range(T * C):
        t, ch = divmod(i, C)  # correct mapping for C-order flattening
        rows.append([f"f{i}", t, ch, float(flat[i])])
    pd.DataFrame(rows, columns=["Feature", "Timestep", "Channel", "Importance (%)"]).to_csv(out_csv, index=False)

    # ---------------- Print top-5 summary ----------------
    top_idx = np.argsort(flat)[::-1][:5]
    print(f"\nDataset: {args.dataset} | Backbone: {args.backbone} | Framework: {args.framework}")
    print("[INFO] Top-5 most influential features (SHAP, Wide-Timestep):")
    for idx in top_idx:
        t, ch = divmod(idx, C)
        print(f"  f{idx} (t={t}, ch={ch}): {flat[idx]:.2f}%")
    print(f"[INFO] Saved results → {out_csv}")
