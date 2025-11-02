import argparse
import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
from trainer import *

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epoch', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_cls', type=float, default=1e-3)
parser.add_argument('--scheduler', type=bool, default=True)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dataset', type=str, default='uci',
                    choices=['uci', 'hhar', 'motion', 'shoaib'])
parser.add_argument('--n_feature', type=int, default=9)
parser.add_argument('--len_sw', type=int, default=120)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--cases', type=str, default='random',
                    choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'])
parser.add_argument('--split_ratio', type=float, default=0.2)
parser.add_argument('--target_domain', type=str, default='0')
parser.add_argument('--aug1', type=str, default='jit_scal')
parser.add_argument('--aug2', type=str, default='resample')
parser.add_argument('--framework', type=str, default='byol',
                    choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc'])
parser.add_argument(
    '--backbone', type=str, default='Transformer',
    choices=['FCN', 'DCL', 'LSTM', 'ResNet18','Transformer'],
    help='backbone network'
)
parser.add_argument('--criterion', type=str, default='cos_sim',
                    choices=['cos_sim', 'NTXent'])
parser.add_argument('--p', type=int, default=128)
parser.add_argument('--phid', type=int, default=128)
parser.add_argument('--logdir', type=str, default='log/')
parser.add_argument('--lr_mul', type=float, default=10.0)
parser.add_argument('--EMA', type=float, default=0.996)
parser.add_argument('--mmb_size', type=int, default=1024)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=1.0)
parser.add_argument('--temp_unit', type=str, default='tsfm',
                    choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'])
parser.add_argument('--device', type=str, default='Phones',
                    choices=['Phones', 'Watch'])
parser.add_argument('--plt', type=bool, default=False)


from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os

def load_npy_dataset(dataset_name, test_ratio=0.2, val_ratio=0.1):
    data_dir = os.path.join("./data", dataset_name)
    data_path = os.path.join(data_dir, "data_20_120.npy")
    label_path = os.path.join(data_dir, "label_20_120.npy")

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing .npy files for dataset: {dataset_name}")

    X = np.load(data_path)
    y = np.load(label_path)
    print(f"[INFO] Loaded {dataset_name}: X={X.shape}, y={y.shape}")

    if y.ndim == 3 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    if y.ndim == 2:
        y = y[:, -1]

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

def build_loaders(train_ds, val_ds, test_ds, args):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader

def infer_num_classes_from_splits(train_ds, val_ds, test_ds):
    base_ds = train_ds.dataset
    y_all = base_ds.tensors[1]
    idxs = []
    for ds in [train_ds, val_ds, test_ds]:
        idxs.extend(ds.indices)
    idxs = torch.tensor(idxs, dtype=torch.long)
    y = y_all[idxs]
    uniques = torch.unique(y)
    num_classes = int(uniques.max().item() + 1)
    return num_classes

if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = load_npy_dataset(args.dataset)
    train_loaders, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, args)
    args.n_class = infer_num_classes_from_splits(train_ds, val_ds, test_ds)
    
    total_len = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"[INFO] Total Samples: {total_len:,}")

    '''
    sample_x = train_ds.dataset.tensors[0]
    args.n_feature = sample_x.shape[-1]
    
    model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)

    # ================================================================
    # Determine feature dimension safely (framework-aware)
    # ================================================================
    model.eval()
    with torch.no_grad():
        sample_batch = torch.randn(8, args.len_sw, args.n_feature).to(DEVICE)

        # --- Framework-aware forward ---
        # SSL frameworks (SimCLR, BYOL, SimSiam, NNCLR, TSTCC) require two inputs
        if args.framework in ['simclr', 'byol', 'simsiam', 'nnclr', 'tstcc']:
            output = model(sample_batch, sample_batch)
        else:
            output = model(sample_batch)

        # --- Extract feature tensor robustly ---
        if isinstance(output, (tuple, list)):
            for o in reversed(output):
                if isinstance(o, torch.Tensor):
                    features = o
                    break
            else:
                raise ValueError("No tensor output found in model forward() result.")
        else:
            features = output

        feat_dim = features.reshape(features.size(0), -1).shape[1]

    
    best_pretrain_model = train(
        train_loaders=train_loaders,
        val_loader=val_loader,
        model=model,
        logger=logger,
        fitlog=fitlog,
        DEVICE=DEVICE,
        optimizers=optimizers,
        schedulers=schedulers,
        criterion=criterion,
        args=args,
    )

    # ================================================================
    # Stage 2: Linear Evaluation
    # ================================================================
    print("\n=== Stage 2: Linear Evaluation ===")
    trained_backbone = lock_backbone(model, args)
    best_lincls = train_lincls(
        train_loaders=train_loaders,
        val_loader=val_loader,
        trained_backbone=trained_backbone,
        classifier=classifier,
        logger=logger,
        fitlog=fitlog,
        DEVICE=DEVICE,
        optimizer_cls=optimizer_cls,
        criterion_cls=criterion_cls,
        args=args
    )

    # ================================================================
    # Stage 3: Testing
    # ================================================================
    print("\n=== Stage 3: Testing ===")
    print(f"[INFO] Using dataset: {args.dataset} | Backbone: {args.backbone} | Framework: {args.framework}")
    test_lincls(
        test_loader=test_loader,
        trained_backbone=trained_backbone,
        best_lincls=best_lincls,
        logger=logger,
        fitlog=fitlog,
        DEVICE=DEVICE,
        criterion_cls=criterion_cls,
        args=args,
        plt=args.plt
    )
    '''
