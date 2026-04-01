"""
Yosinski-style per-layer linear probe diagnostic.

For each CNN layer, fits a binary logistic regression (ID vs OOD) on
max-pooled features and reports AUROC. Run this before tuning the
multi-layer feature concatenation to see which layers help near-OOD
vs far-OOD, informing MRL-style layer weights.

Usage:
    python scripts/layer_probe_diagnostic.py --dataset cifar100 --seed 0
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.postprocessors import KernelAttentionPostprocessor
from openood.utils.config import Config

DATASET_CONFIG = {
    'cifar10': {
        'network_class': ResNet18_32x32,
        'num_classes': 10,
        'id_name': 'cifar10',
        'ckpt_dir': 'cifar10_resnet18_32x32_base_e100_lr0.1_default',
    },
    'cifar100': {
        'network_class': ResNet18_32x32,
        'num_classes': 100,
        'id_name': 'cifar100',
        'ckpt_dir': 'cifar100_resnet18_32x32_base_e100_lr0.1_default',
    },
    'imagenet200': {
        'network_class': ResNet18_224x224,
        'num_classes': 200,
        'id_name': 'imagenet200',
        'ckpt_dir': 'imagenet200_resnet18_224x224_base_e90_lr0.1_default',
    },
}

LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'penultimate']


def build_default_ckpt_path(dataset: str, seed: int) -> str:
    cfg = DATASET_CONFIG[dataset]
    return os.path.join(
        ROOT_DIR, 'results', 'checkpoints',
        cfg['ckpt_dir'], f's{seed}', 'best.ckpt'
    )


@torch.no_grad()
def extract_layer_features(net: nn.Module, loader, device: str = 'cuda') -> dict:
    """Max-pool spatial features at each layer for every batch in loader.

    feature_list indices (ResNet18):
        [1] → layer1: (B,  64, H, W)
        [2] → layer2: (B, 128, H, W)
        [3] → layer3: (B, 256, H, W)
        [4] → penultimate: (B, 512, 1, 1)

    Returns dict: layer_name -> np.ndarray (N, C)
    """
    net.eval()
    buffers = {name: [] for name in LAYER_NAMES}

    for batch in tqdm(loader, desc='  features', leave=False):
        data = batch['data'].to(device)
        _, feature_list = net(data, return_feature_list=True)

        for idx, name in enumerate(LAYER_NAMES[:3]):   # layer1/2/3
            feat = feature_list[idx + 1]               # indices 1, 2, 3
            B, C, H, W = feat.shape
            pooled = feat.view(B, C, -1).max(dim=2).values  # (B, C) — max over spatial
            buffers[name].append(pooled.cpu().numpy())

        # penultimate already reduced by global avg pool
        pen = feature_list[4].view(feature_list[4].size(0), -1)
        buffers['penultimate'].append(pen.cpu().numpy())

    return {name: np.concatenate(bufs, axis=0) for name, bufs in buffers.items()}


def probe_auroc(id_feats: np.ndarray, ood_feats: np.ndarray,
                max_per_class: int = 5000) -> float:
    """Fit a logistic regression on (id, ood) features; return held-out AUROC."""
    rng = np.random.default_rng(42)
    if len(id_feats) > max_per_class:
        id_feats = id_feats[rng.choice(len(id_feats), max_per_class, replace=False)]
    if len(ood_feats) > max_per_class:
        ood_feats = ood_feats[rng.choice(len(ood_feats), max_per_class, replace=False)]

    X = np.concatenate([id_feats, ood_feats], axis=0)
    y = np.concatenate([np.ones(len(id_feats)), np.zeros(len(ood_feats))], axis=0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--seed', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--data-root', type=str,
                        default='/storage/ice-shared/cs8903onl/kernel-datasets/data')
    parser.add_argument('--config-root', type=str,
                        default=os.path.join(ROOT_DIR, 'configs'))
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    ds_cfg   = DATASET_CONFIG[args.dataset]
    ckpt_path = args.ckpt_path or build_default_ckpt_path(args.dataset, args.seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("Per-Layer Linear Probe Diagnostic (Yosinski-style)")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Checkpoint : {ckpt_path}")
    print("=" * 60)

    # --- Network ---
    net = ds_cfg['network_class'](num_classes=ds_cfg['num_classes'])
    net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    net.cuda().eval()

    # --- Dataloaders via Evaluator (no eval_ood called, just for loader setup) ---
    config = Config(os.path.join(args.config_root, 'postprocessors', 'kernel_attention.yml'))
    postprocessor = KernelAttentionPostprocessor(config)
    postprocessor.APS_mode = False
    postprocessor.hyperparam_search_done = False

    evaluator = Evaluator(
        net=net,
        id_name=ds_cfg['id_name'],
        data_root=args.data_root,
        config_root=args.config_root,
        postprocessor=postprocessor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- ID test features (extracted once, reused for every OOD set) ---
    print(f"\nExtracting ID test features ({ds_cfg['id_name']})...")
    id_feats = extract_layer_features(net, evaluator.dataloader_dict['id']['test'])
    n_id = len(next(iter(id_feats.values())))
    print(f"  {n_id} samples | dims: " +
          ", ".join(f"{name}={id_feats[name].shape[1]}" for name in LAYER_NAMES))

    # --- Per-layer probes for each OOD dataset ---
    results = {}   # (split, ds_name) -> {layer_name: auroc}

    for split in ('near', 'far'):
        for ds_name, loader in evaluator.dataloader_dict['ood'][split].items():
            print(f"\n[{split}-OOD] {ds_name}")
            ood_feats = extract_layer_features(net, loader)
            n_ood = len(next(iter(ood_feats.values())))
            print(f"  {n_ood} OOD samples")

            aurocs = {}
            for layer in LAYER_NAMES:
                auroc = probe_auroc(id_feats[layer], ood_feats[layer])
                aurocs[layer] = auroc
                print(f"    {layer:<14}: {auroc * 100:5.1f}%")
            results[(split, ds_name)] = aurocs

    # --- Summary table ---
    col_w  = 13
    header = f"{'Dataset':<20} {'Split':<8}" + \
             "".join(f"{l:>{col_w}}" for l in LAYER_NAMES)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SUMMARY — Per-Layer AUROC")
    print("=" * len(header))
    print(header)
    print(sep)

    near_vals = {l: [] for l in LAYER_NAMES}
    far_vals  = {l: [] for l in LAYER_NAMES}

    for (split, ds_name), aurocs in results.items():
        row = f"{ds_name:<20} {split:<8}" + \
              "".join(f"{aurocs[l] * 100:>{col_w}.1f}" for l in LAYER_NAMES)
        print(row)
        for l in LAYER_NAMES:
            (near_vals if split == 'near' else far_vals)[l].append(aurocs[l])

    print(sep)
    for label, vals in (('near avg', near_vals), ('far avg', far_vals)):
        print(f"{label:<20} {'':8}" +
              "".join(f"{np.mean(vals[l]) * 100:>{col_w}.1f}" for l in LAYER_NAMES))
    print("=" * len(header))

    print("\nInterpretation guide:")
    print("  Higher AUROC at a layer = that layer's features better separate ID from that OOD set.")
    print("  Compare near avg vs far avg rows to see the per-layer near/far tradeoff.")


if __name__ == '__main__':
    main()
