import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
from augmentations import gen_aug
from utils import tsne, mds, _logger
import time
from models.frameworks import *
from models.backbones import *
from models.loss import *

from sklearn.metrics import f1_score
import seaborn as sns
import fitlog
from copy import deepcopy
import torch.optim as optim


# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def setup_dataloaders(args):
    if args.dataset == 'uci':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)

    return train_loaders, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(args, DEVICE, classifier=True):
    """Setup model, backbone, and optimizers."""

    # ====================== Backbone Setup ======================
    if args.backbone == 'FCN':
        backbone = FCN(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            backbone=True
        )
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            conv_kernels=64,
            kernel_size=5,
            LSTM_units=128,
            backbone=True
        )
    elif args.backbone == 'LSTM':
        backbone = LSTM(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            LSTM_units=128,
            backbone=True
        )
    elif args.backbone == 'Transformer':
        backbone = Transformer(
        input_dim=args.n_feature,   
        seq_len=args.len_sw,
        d_model=128,
        depth=4,
        heads=4,
        mlp_dim=64,
        dropout=0.1,
        out_dim=128
    )
    elif args.backbone == 'ResNet18':
        backbone = ResNet18(
        n_channels=args.n_feature,
        n_classes=args.n_class,
        out_channels=128,
        backbone=True
    )
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")

    # ====================== Framework Setup ======================
    if args.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        optimizers = [optimizer]

    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        optimizers = [optimizer]

    elif args.framework == 'tstcc':
        model = TSTCC(
            backbone=backbone,
            DEVICE=DEVICE,
            temp_unit=args.temp_unit,
            tc_hidden=100
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay
        )
        optimizers = [optimizer]

    else:
        raise NotImplementedError(f"Framework {args.framework} not implemented.")

    # Move model to device
    model = model.to(DEVICE)

    # ====================== Classifier Setup ======================
    if classifier:
        classifier_layer = nn.Linear(backbone.out_dim, args.n_class).to(DEVICE)
        optimizer_cls = torch.optim.Adam(
            classifier_layer.parameters(),
            lr=args.lr_cls,
            weight_decay=args.weight_decay
        )
        return model, classifier_layer, [optimizer, optimizer_cls]

    return model, None, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)


def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 0.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        #args.backbone = 'FCN'
        args.weight_decay = 3e-4

    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)

    args.model_name = 'try_scheduler_' + args.framework + '_pretrain_' + args.dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)

    # fitlog
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None

    return model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls

'''
def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):
    aug_sample1 = gen_aug(sample, args.aug1)
    aug_sample2 = gen_aug(sample, args.aug2)
    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(
        DEVICE).long()
    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if args.framework == 'nnclr':
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)
        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        elif args.criterion == 'NTXent':
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'simclr':
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        loss = criterion(z1, z2)
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    return loss
'''

def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):
    """
    Gradient-safe loss calculator for contrastive frameworks.
    Supports SimCLR, TSTCC, BYOL, SimSiam, NNCLR.
    """
    # ---------------- Augmentation ----------------
    aug_sample1 = gen_aug(sample, args.aug1).to(DEVICE).float()
    aug_sample2 = gen_aug(sample, args.aug2).to(DEVICE).float()

    # ---------------- SimCLR ----------------
    if args.framework == 'simclr':
        z1, z2 = model(aug_sample1, aug_sample2)
        loss = criterion(z1, z2)

    # ---------------- NNCLR / BYOL / SimSiam ----------------
    elif args.framework in ['byol', 'simsiam', 'nnclr']:
        p1, p2, z1, z2 = model(aug_sample1, aug_sample2)

        if args.framework == 'nnclr' and nn_replacer is not None:
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)

        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        else:
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5

    # ---------------- TSTCC ----------------
    elif args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(aug_sample1, aug_sample2)
        loss = (nce1 + nce2) * args.lambda1 + criterion(p1, p2) * args.lambda2

    else:
        raise NotImplementedError(f"Unsupported framework: {args.framework}")

    # ---------------- Safety Checks ----------------
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError(f"Loss must be a Tensor, got {type(loss)}")
    return loss

def train(train_loaders, val_loader, model, logger, fitlog,
          DEVICE, optimizers, schedulers, criterion, args,
          recon=None, nn_replacer=None):
    """
    Simplified gradient-safe training loop.
    Fixes 'loss has no grad_fn' by ensuring .item() is used only after backward.
    Also saves a best-model checkpoint for later linear evaluation (lincls_ckpt).
    """
    best_model = None
    min_val_loss = float("inf")

    for epoch in range(args.n_epoch):
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0.0
        n_batches = 0

        model.train()
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                if sample.size(0) != args.batch_size:
                    continue

                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

                # compute differentiable loss
                loss = calculate_model_loss(
                    args, sample, target, model, criterion, DEVICE,
                    recon=recon, nn_replacer=nn_replacer
                )

                # ---- critical fix ----
                if not loss.requires_grad:
                    raise RuntimeError("Loss detached from graph. Check model forward().")

                # backward + optimize
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                if args.framework in ['byol', 'simsiam']:
                    model.update_moving_average()

                total_loss += loss.detach().item()
                n_batches += 1

        # scheduler step
        fitlog.add_loss(optimizers[0].param_groups[0]['lr'], name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        avg_train_loss = total_loss / max(1, n_batches)
        logger.debug(f'Train Loss : {avg_train_loss:.4f}')
        fitlog.add_loss(avg_train_loss, name="pretrain training loss", step=epoch)

        # Save per-epoch model checkpoint
        model_dir = os.path.join(model_dir_name, f'pretrain_{args.model_name}_epoch{epoch}.pt')
        print(f'Saving model checkpoint to {model_dir}')
        torch.save({'model_state_dict': model.state_dict()}, model_dir)

        # ---------- Validation ----------
        if args.cases in ['subject', 'subject_large']:
            best_model = copy.deepcopy(model.state_dict())
        else:
            model.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for idx, (sample, target, domain) in enumerate(val_loader):
                    if sample.size(0) != args.batch_size:
                        continue
                    loss = calculate_model_loss(
                        args, sample, target, model, criterion, DEVICE,
                        recon=recon, nn_replacer=nn_replacer
                    )
                    val_total += loss.detach().item()
                    val_batches += 1

            avg_val_loss = val_total / max(1, val_batches)
            logger.debug(f'Val Loss : {avg_val_loss:.4f}')
            fitlog.add_loss(avg_val_loss, name="pretrain validation loss", step=epoch)

            # Save best model based on validation loss
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_model = copy.deepcopy(model.state_dict())
                print(f"✅ Validation improved at epoch {epoch}: {avg_val_loss:.4f}")
                
                # === NEW ADDITION ===
                best_path = os.path.join(model_dir_name,
                    f'pretrain_best_{args.model_name}.pt')
                torch.save({'model_state_dict': best_model}, best_path)
                print(f"[INFO] Saved best model checkpoint to {best_path}")

    # === NEW ADDITION ===
    # Save final best model again for clarity (for use as lincls_ckpt)
    final_best_path = os.path.join(model_dir_name,
    f'lincls_try_scheduler_{args.framework}_pretrain_{args.dataset}_best.pt')

    torch.save({'model_state_dict': best_model}, final_best_path)
    print(f"[✅] Final best model saved for linear classifier: {final_best_path}")

    return best_model

'''
def train(train_loaders, val_loader, model, logger, fitlog, DEVICE, optimizers, schedulers, criterion, args):
    best_model = None
    min_val_loss = 1e8

    for epoch in range(args.n_epoch):
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 0
        model.train()
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if args.framework in ['byol', 'simsiam']:
                    model.update_moving_average()
        fitlog.add_loss(optimizers[0].param_groups[0]['lr'], name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        # save model
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'model_state_dict': model.state_dict()}, model_dir)

        logger.debug(f'Train Loss     : {total_loss / n_batches:.4f}')

        fitlog.add_loss(total_loss / n_batches, name="pretrain training loss", step=epoch)

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_model = copy.deepcopy(model.state_dict())
        else:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                n_batches = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    if sample.size(0) != args.batch_size:
                        continue
                    n_batches += 1
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                    total_loss += loss.item()
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                    print('update')
                logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)
    return best_model
'''

def test(test_loader, best_model, logger, fitlog, DEVICE, criterion, args):
    model, classifier, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        for idx, (sample, target, domain) in enumerate(test_loader):
            if sample.size(0) != args.batch_size:
                continue
            n_batches += 1
            loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
            total_loss += loss.item()
        logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
        fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss / n_batches}})

    return model


def lock_backbone(model, args):
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.framework in ['simclr', 'nnclr']:
        trained_backbone = model.encoder
    elif args.framework == 'tstcc':
        trained_backbone = model.backbone  
    else:
        NotImplementedError

    return trained_backbone

'''
def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion):
    _, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    output = classifier(feat)
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    return loss, predicted, feat
'''

def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion):
    """
    Compute linear classification loss and predictions.
    Works for backbones that return (h, z) or single z.
    """
    output = trained_backbone(sample)

    # Handle Transformer/TSTCC backbone outputs
    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            h, z = output
        else:
            z = output[-1]
    else:
        z = output

    # Ensure 2D features
    if z.ndim > 2:
        z = torch.mean(z, dim=1)  # global average pooling

    logits = classifier(z)
    loss = criterion(logits, target)
    predicted = torch.argmax(logits, dim=1)

    return loss, predicted, logits

'''
def train_lincls(train_loaders, val_loader, trained_backbone, classifier, logger, fitlog, DEVICE, optimizer, criterion, args):
    best_lincls = None
    min_val_loss = 1e8

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    for epoch in range(args.n_epoch):
        classifier.train()
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        total = 0
        correct = 0
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                total_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save model
        model_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)

        acc_train = float(correct) * 100.0 / total
        logger.debug(f'epoch train loss     : {total_loss:.4f}, train acc     : {acc_train:.4f}')
        fitlog.add_loss(total_loss, name="Train Loss", step=epoch)
        fitlog.add_metric({"dev": {"Train Acc": acc_train}}, step=epoch)

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                total_loss = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
                    total_loss += loss.item()
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
                    print('update')
                logger.debug(f'epoch val loss     : {total_loss:.4f}, val acc     : {acc_val:.4f}')
                fitlog.add_loss(total_loss, name="Val Loss", step=epoch)
                fitlog.add_metric({"dev": {"Val Acc": acc_val}}, step=epoch)
    return best_lincls
'''

def train_lincls(train_loaders, val_loader, trained_backbone, classifier,
                 logger, fitlog, DEVICE, optimizer_cls, criterion_cls, args):
    """
    Train linear classifier on top of a frozen pretrained backbone.
    Automatically adjusts feature dimension for Transformer/TSTCC backbones.
    """
    best_model_state = None
    best_val_loss = float("inf")
    trained_backbone.eval()

    for p in trained_backbone.parameters():
        p.requires_grad = False

    feature_dim_fixed = False  # flag to resize classifier dynamically

    for epoch in range(args.n_epoch):
        classifier.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for train_loader in train_loaders:
            for sample, target, domain in train_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                # Forward through backbone
                with torch.no_grad():
                    output = trained_backbone(sample)
                    if isinstance(output, (tuple, list)):
                        if len(output) == 2:
                            h, z = output
                        else:
                            z = output[-1]
                    else:
                        z = output

                    if z.ndim > 2:
                        z = torch.mean(z, dim=1)  # global average pooling if 3D

                # Dynamically fix classifier input dim (once)
                if not feature_dim_fixed:
                    in_features = z.shape[-1]
                    if classifier.in_features != in_features:
                        print(f"[INFO] Adjusting classifier: {classifier.in_features} → {in_features}")
                        classifier = nn.Linear(in_features, args.n_class).to(DEVICE)
                        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)
                    feature_dim_fixed = True

                # Forward + backward
                logits = classifier(z)
                loss = criterion_cls(logits, target)
                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()

                total_loss += loss.item() * sample.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += sample.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        print(f"Epoch : {epoch}")
        print(f"Train loss : {train_loss:.4f}, Train acc : {train_acc:.4f}")

        # ---------- Validation ----------
        classifier.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for sample, target, domain in val_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                output = trained_backbone(sample)
                if isinstance(output, (tuple, list)):
                    if len(output) == 2:
                        h, z = output
                    else:
                        z = output[-1]
                else:
                    z = output

                if z.ndim > 2:
                    z = torch.mean(z, dim=1)

                logits = classifier(z)
                loss = criterion_cls(logits, target)
                val_loss += loss.item() * sample.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == target).sum().item()
                val_samples += sample.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        print(f"Val loss : {val_loss:.4f}, Val acc : {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = classifier.state_dict().copy()
            print("✅ Validation improved — updated best classifier")

    return best_model_state

'''
def test_lincls(test_loader, trained_backbone, best_lincls, logger, fitlog, DEVICE, criterion, args, plt=False):
    classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
    classifier.load_state_dict(best_lincls)
    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    trgs = np.array([])
    preds = np.array([])
    with torch.no_grad():
        classifier.eval()
        for idx, (sample, target, domain) in enumerate(test_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
            total_loss += loss.item()
            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            trgs = np.append(trgs, target.data.cpu().numpy())
            preds = np.append(preds, predicted.data.cpu().numpy())
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_test = float(correct) * 100.0 / total

        miF = f1_score(trgs, preds, average='micro') * 100
        maF = f1_score(trgs, preds, average='weighted') * 100

        logger.debug(f'epoch test loss     : {total_loss:.4f}, test acc     : {acc_test:.4f}, miF     : {miF:.4f}, maF     : {maF:.4f}')

        fitlog.add_best_metric({"dev": {"Test Loss": total_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc": acc_test}})
        fitlog.add_best_metric({"dev": {"miF": miF}})
        fitlog.add_best_metric({"dev": {"maF": maF}})

        logger.debug(confusion_matrix)
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)
'''

def test_lincls(test_loader, trained_backbone, best_lincls, logger,
                fitlog, DEVICE, criterion_cls, args, plt=False):

    # Rebuild classifier with same shape
    sample_batch = next(iter(test_loader))[0].to(DEVICE)
    output = trained_backbone(sample_batch)
    if isinstance(output, (tuple, list)):
        feat = output[-1]
    else:
        feat = output
    feat_dim = feat.shape[-1]

    classifier = torch.nn.Linear(feat_dim, args.n_class).to(DEVICE)

    try:
        classifier.load_state_dict(best_lincls)
    except RuntimeError:
        if "model_state_dict" in best_lincls:
            classifier.load_state_dict(best_lincls["model_state_dict"])
        else:
            classifier.load_state_dict(best_lincls)

    trained_backbone.eval()
    classifier.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for sample, target, domain in test_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

            output = trained_backbone(sample)
            if isinstance(output, (tuple, list)):
                feat = output[-1]
            else:
                feat = output

            logits = classifier(feat)
            loss = criterion_cls(logits, target)

            total_loss += loss.item() * sample.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += sample.size(0)

    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples
    print(f"Test Loss     : {test_loss:.4f}, Test Acc     : {test_acc:.4f}")
