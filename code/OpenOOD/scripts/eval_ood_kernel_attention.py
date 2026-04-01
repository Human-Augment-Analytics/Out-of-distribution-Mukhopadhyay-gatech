import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.postprocessors import KernelAttentionPostprocessor
from openood.utils.config import Config, merge_configs

# Per-dataset configuration: network class, num_classes, id_name for the
# Evaluator, and the expected checkpoint directory name under results/checkpoints/.
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


def build_default_ckpt_path(dataset: str, seed: int) -> str:
    cfg = DATASET_CONFIG[dataset]
    return os.path.join(
        ROOT_DIR, 'results', 'checkpoints',
        cfg['ckpt_dir'], f's{seed}', 'best.ckpt'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=list(DATASET_CONFIG.keys()),
                        help='In-distribution dataset')
    parser.add_argument('--seed', type=int, default=0, choices=[0, 1, 2],
                        help='Checkpoint seed (selects s0/s1/s2 subfolder)')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Path to checkpoint. Auto-derived from --dataset '
                             'and --seed if not specified.')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='RBF kernel bandwidth. Ignored when --auto-sigma is set.')
    parser.add_argument('--auto-sigma', action='store_true',
                        help='Estimate sigma via median global pairwise distance.')
    parser.add_argument('--rff-dim', type=int, default=2048)
    parser.add_argument('--sigma-scale', type=float, default=1.0,
                        help='Multiplicative scale applied to sigma after estimation.')
    parser.add_argument('--score-mode', type=str, default='per_class_threshold',
                        choices=['max', 'per_class_threshold'],
                        help='Scoring mode: max or per_class_threshold.')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--kernel-weighted', action='store_true',
                        help='Use kernel-weighted class embeddings v_c = M_c mu_hat_c.')
    parser.add_argument('--dual-head', action='store_true',
                        help='Use separate near (layer3+pen) and far (4-layer MRL) scoring heads.')
    parser.add_argument('--head-alpha', type=float, default=0.5,
                        help='Weight for far head score; near head gets (1 - head_alpha).')
    parser.add_argument('--data-root', type=str,
                        default='/storage/ice-shared/cs8903onl/kernel-datasets/data')
    parser.add_argument('--config-root', type=str,
                        default=os.path.join(ROOT_DIR, 'configs'))
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIG[args.dataset]

    # Resolve checkpoint path
    ckpt_path = args.ckpt_path or build_default_ckpt_path(args.dataset, args.seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"For '{args.dataset}' you may need to train a model first or "
            f"supply --ckpt-path explicitly."
        )

    sigma_mode = 'median' if args.auto_sigma else 'fixed'

    print("=" * 60)
    print("Kernel-Attention OOD Evaluation")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  σ mode     : {sigma_mode}" +
          (f" (value {args.sigma})" if sigma_mode == 'fixed' else ""))
    print(f"  D          : {args.rff_dim}")
    print("=" * 60)

    config_path = os.path.join(args.config_root, 'postprocessors',
                               'kernel_attention.yml')
    config = Config(config_path)
    override = Config(**{
        'postprocessor': {
            'APS_mode': False,
            'postprocessor_args': {
                'rff_dim': args.rff_dim,
                'sigma': args.sigma,
                'sigma_mode': sigma_mode,
                'sigma_scale': args.sigma_scale,
                'score_mode': args.score_mode,
                'alpha': args.alpha,
                'kernel_weighted': args.kernel_weighted,
                'dual_head': args.dual_head,
                'head_alpha': args.head_alpha,
            },
        }
    })
    config = merge_configs(config, override)

    postprocessor = KernelAttentionPostprocessor(config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False

    net = ds_cfg['network_class'](num_classes=ds_cfg['num_classes'])
    net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    net.cuda()
    net.eval()

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

    metrics = evaluator.eval_ood()

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(metrics)

    # === PI Debugging Diagnostics (test splits) ===
    print("\n" + "=" * 60)
    print("PI Debugging Diagnostics")
    print("=" * 60)

    # Check 1: φ(x) mean/std for test-ID and first near/far OOD
    print("\n[Check 1] φ(x) mean/std (test splits):")
    id_phi = postprocessor.get_phi_stats(net, evaluator.dataloader_dict['id']['test'])
    print(f"  test-ID              : mean={id_phi['mean']:.4f}  std={id_phi['std']:.4f}")
    for split in ('near', 'far'):
        for ds_name, loader in list(evaluator.dataloader_dict['ood'][split].items())[:1]:
            s = postprocessor.get_phi_stats(net, loader)
            print(f"  test-OOD ({ds_name:12s}): mean={s['mean']:.4f}  std={s['std']:.4f}")

    # Check 2: score distributions — ID vs each OOD set
    print("\n[Check 2] Score distributions:")
    id_conf = evaluator.scores['id']['test'][1]   # numpy array
    print(f"  ID   : mean={id_conf.mean():.4f}  std={id_conf.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ('near', 'far')):
        ax.hist(id_conf, bins=60, alpha=0.6, label='ID', color='steelblue', density=True)
        for ds_name, (_, ood_conf, _) in evaluator.scores['ood'][split].items():
            print(f"  {ds_name:16s}: mean={ood_conf.mean():.4f}  std={ood_conf.std():.4f}")
            ax.hist(ood_conf, bins=60, alpha=0.45, label=ds_name, density=True)
        ax.axvline(postprocessor.threshold, color='black', linestyle='--',
                   label=f'τ={postprocessor.threshold:.3f}')
        ax.set_title(f'{split}-OOD vs ID')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.suptitle(
        f'{args.dataset} | σ={postprocessor.sigma:.4f}  D={args.rff_dim}  α={args.alpha}'
    )
    plt.tight_layout()
    plot_path = f'score_dist_{args.dataset}_s{args.seed}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {plot_path}")

    # === Latency Benchmark ===
    print("\n" + "=" * 60)
    print("Latency Benchmark  (ID test loader, GPU)")
    print("=" * 60)
    id_test_loader = evaluator.dataloader_dict['id']['test']
    lat = postprocessor.benchmark_latency(net, id_test_loader,
                                          n_warmup=5, n_batches=20)
    print(f"  Batch size              : {lat['batch_size']}")
    print(f"  Batches timed           : 20  ({lat['n_samples']} samples)")
    print(f"  Per-sample latency (ms) :")
    print(f"    mean ± std            : {lat['mean_ms']:.3f} ± {lat['std_ms']:.3f}")
    print(f"    p50 / p95 / p99       : {lat['p50_ms']:.3f} / "
          f"{lat['p95_ms']:.3f} / {lat['p99_ms']:.3f}")
    print(f"  Batch latency mean (ms) : {lat['batch_ms_mean']:.1f}")
    print(f"  Throughput (samples/s)  : {1000.0 / lat['mean_ms']:.0f}")


if __name__ == '__main__':
    main()
