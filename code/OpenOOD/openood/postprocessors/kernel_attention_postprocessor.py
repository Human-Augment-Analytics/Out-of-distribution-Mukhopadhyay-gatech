from typing import Any
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class KernelAttentionPostprocessor(BasePostprocessor):
    """
    Kernel-Attention OOD Detection via Random Fourier Features (RFF).

    sigma_mode:
        'fixed'  — use self.sigma as-is
        'median' — estimate sigma via median global pairwise distance

    score_mode:
        'max'                 — score(x) = max_c mu_hat_c^T phi(x)
        'per_class_threshold' — tau_c = alpha-quantile of same-class cal scores;
                                score(x) = max_c (mu_hat_c^T phi(x) - tau_c)
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.rff_dim         = self.args.rff_dim
        self.sigma           = self.args.sigma
        self.alpha           = self.args.alpha
        self.sigma_mode      = getattr(self.args, 'sigma_mode', 'fixed')
        self.sigma_scale     = getattr(self.args, 'sigma_scale', 1.0)
        self.score_mode      = getattr(self.args, 'score_mode', 'max')
        self.kernel_weighted = getattr(self.args, 'kernel_weighted', False)
        # MRL-style per-layer weights [layer1, layer2, layer3, penultimate]
        # Defaults derived from Yosinski linear probe: product of near_avg x far_avg AUROC
        self.layer_weights   = list(getattr(self.args, 'layer_weights',
                                            [0.712, 0.734, 0.771, 0.790]))
        # Dual-head: separate near (layer3+penultimate) and far (all 4 layers) scoring heads
        self.dual_head       = getattr(self.args, 'dual_head', False)
        self.head_alpha      = getattr(self.args, 'head_alpha', 0.5)  # weight for far head

        # Computed during setup — far head (or single head when dual_head=False)
        self.omega            = None   # (D, feat_dim)
        self.bias             = None   # (D,)
        self.mu_hat           = None   # (C, D)
        self.threshold        = None   # global threshold (max mode) or 0.0 (per_class_threshold)
        self.class_thresholds = None   # (C,) — only used in per_class_threshold mode

        # Computed during setup — near head (only when dual_head=True)
        self.sigma_near             = None
        self.omega_near             = None   # (D, 1024)
        self.bias_near              = None   # (D,)
        self.mu_hat_near            = None   # (C, D)
        self.class_thresholds_near  = None   # (C,)

        self.setup_flag = False

    # ------------------------------------------------------------------
    # Hyperparameter sweep interface
    # ------------------------------------------------------------------

    def set_hyperparam(self, hyperparam: list):
        self.sigma   = hyperparam[0]
        self.rff_dim = hyperparam[1]

    def get_hyperparam(self):
        return [self.sigma, self.rff_dim]

    # ------------------------------------------------------------------
    # Sigma estimation
    # ------------------------------------------------------------------

    def _compute_median_sigma(self, features: torch.Tensor,
                               subsample: int = 2000) -> float:
        """Estimate σ via the median global pairwise distance heuristic."""
        n = features.shape[0]
        if n > subsample:
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))[:subsample]
            features = features[idx]
        sq_norms = (features ** 2).sum(dim=1)
        dists_sq = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * (features @ features.T)
        dists_sq = dists_sq.clamp(min=0)
        mask = torch.triu(torch.ones(len(features), len(features), dtype=torch.bool), diagonal=1)
        return max(dists_sq[mask].sqrt().median().item(), 1e-6)

    # ------------------------------------------------------------------
    # RFF parameter initialisation
    # ------------------------------------------------------------------

    def _init_rff_params(self, feature_dim: int):
        """Single (D, feat_dim) omega and (D,) bias."""
        D   = self.rff_dim
        rng = torch.Generator().manual_seed(42)
        self.omega = torch.randn(D, feature_dim, generator=rng) / self.sigma
        self.bias  = torch.rand(D, generator=rng) * 2 * np.pi

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, net: nn.Module, data: torch.Tensor):
        """Multi-layer feature extraction with MRL-style per-layer weighting.

        Concatenates (all scaled by probe-derived weights before concat):
          - layer1 min+max: 2 ×  64 = 128-d  (weight layer_weights[0])
          - layer2 min+max: 2 × 128 = 256-d  (weight layer_weights[1])
          - layer3 min+max: 2 × 256 = 512-d  (weight layer_weights[2])
          - penultimate   :       512-d       (weight layer_weights[3])
        Total: 1408-d (globally L2-normalised downstream).
        Returns (logits, features).
        """
        logits, feature_list = net(data, return_feature_list=True)
        # feature_list[1]: (batch,  64, H, W) — layer1
        # feature_list[2]: (batch, 128, H, W) — layer2
        # feature_list[3]: (batch, 256, H, W) — layer3
        # feature_list[4]: (batch, 512, 1, 1) — penultimate

        parts = []
        for idx, w in enumerate(self.layer_weights[:3]):   # layer1/2/3
            feat = feature_list[idx + 1]
            B, C, H, W = feat.shape
            flat = feat.view(B, C, -1)                     # (batch, C, H·W)
            parts.append(w * flat.min(dim=2).values)       # (batch, C)
            parts.append(w * flat.max(dim=2).values)       # (batch, C)

        # penultimate
        feat_pen = feature_list[4].view(feature_list[4].size(0), -1)  # (batch, 512)
        parts.append(self.layer_weights[3] * feat_pen)

        return logits, torch.cat(parts, dim=1)             # (batch, 1408)

    # ------------------------------------------------------------------
    # RFF feature computation
    # ------------------------------------------------------------------

    def _compute_rff_features(self, x: torch.Tensor) -> torch.Tensor:
        """φ(x) = sqrt(2/D) cos(Ωx + b).  Returns (batch, D)."""
        omega = self.omega.to(x.device)
        bias  = self.bias.to(x.device)
        proj  = x @ omega.T + bias.unsqueeze(0)
        return torch.sqrt(torch.tensor(2.0 / self.rff_dim, device=x.device)) * torch.cos(proj)

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    def _score(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: (batch, D) → (batch,) score."""
        raw = phi @ self.mu_hat.to(phi.device).T   # (batch, C)
        if self.score_mode == 'per_class_threshold':
            return (raw - self.class_thresholds.to(phi.device).unsqueeze(0)).max(dim=1).values
        return raw.max(dim=1).values

    # ------------------------------------------------------------------
    # Near-head feature extraction and scoring (dual_head mode)
    # ------------------------------------------------------------------

    def _extract_features_near(self, net: nn.Module, data: torch.Tensor):
        """Layer3 + penultimate features (1024-d) for the near-OOD head."""
        logits, feature_list = net(data, return_feature_list=True)
        feat_pen = feature_list[4].view(feature_list[4].size(0), -1)   # (batch, 512)
        layer3   = feature_list[3]
        flat     = layer3.view(layer3.size(0), layer3.size(1), -1)      # (batch, 256, H·W)
        return logits, torch.cat(
            [feat_pen, flat.min(dim=2).values, flat.max(dim=2).values], dim=1
        )   # (batch, 1024)

    def _extract_features_both(self, net: nn.Module, data: torch.Tensor):
        """Single forward pass returning near (1024-d) and far (1408-d) features."""
        logits, feature_list = net(data, return_feature_list=True)

        feat_pen = feature_list[4].view(feature_list[4].size(0), -1)   # (batch, 512)

        # Near: layer3 min/max + penultimate
        layer3 = feature_list[3]
        flat3  = layer3.view(layer3.size(0), layer3.size(1), -1)
        feat_near = torch.cat(
            [feat_pen, flat3.min(dim=2).values, flat3.max(dim=2).values], dim=1
        )   # (batch, 1024)

        # Far: all 4 layers with MRL weights
        parts = []
        for idx, w in enumerate(self.layer_weights[:3]):
            feat = feature_list[idx + 1]
            B, C, H, W = feat.shape
            flat = feat.view(B, C, -1)
            parts.append(w * flat.min(dim=2).values)
            parts.append(w * flat.max(dim=2).values)
        parts.append(self.layer_weights[3] * feat_pen)
        feat_far = torch.cat(parts, dim=1)   # (batch, 1408)

        return logits, feat_near, feat_far

    def _compute_rff_near(self, x: torch.Tensor) -> torch.Tensor:
        """RFF embedding using near-head parameters."""
        omega = self.omega_near.to(x.device)
        bias  = self.bias_near.to(x.device)
        proj  = x @ omega.T + bias.unsqueeze(0)
        return torch.sqrt(torch.tensor(2.0 / self.rff_dim, device=x.device)) * torch.cos(proj)

    def _score_near(self, phi: torch.Tensor) -> torch.Tensor:
        """Score using near-head class embeddings."""
        raw = phi @ self.mu_hat_near.to(phi.device).T   # (batch, C)
        if self.score_mode == 'per_class_threshold':
            return (raw - self.class_thresholds_near.to(phi.device).unsqueeze(0)).max(dim=1).values
        return raw.max(dim=1).values

    # ------------------------------------------------------------------
    # Diagnostics helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_phi_stats(self, net: nn.Module, loader) -> dict:
        """Compute mean/std of φ(x) for a loader (diagnostic check 1)."""
        net.eval()
        all_phi = []
        for batch in loader:
            data = batch['data'].cuda()
            _, feat = self._extract_features(net, data)
            feat = F.normalize(feat, p=2, dim=1)
            all_phi.append(self._compute_rff_features(feat).cpu())
        all_phi = torch.cat(all_phi)
        return {'mean': all_phi.mean().item(), 'std': all_phi.std().item()}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @torch.no_grad()
    def setup(self, net: nn.Module, id_loader_dict: dict, ood_loader_dict: dict):
        if self.setup_flag:
            return
        net.eval()

        # --- Train/cal split ---
        train_dataset = id_loader_dict['train'].dataset
        n_total = len(train_dataset)
        n_cal   = int(0.1 * n_total)
        n_fit   = n_total - n_cal

        fit_dataset, cal_dataset = torch.utils.data.random_split(
            train_dataset, [n_fit, n_cal],
            generator=torch.Generator().manual_seed(42)
        )
        batch_size  = id_loader_dict['train'].batch_size
        num_workers = id_loader_dict['train'].num_workers
        fit_loader  = torch.utils.data.DataLoader(
            fit_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        cal_loader  = torch.utils.data.DataLoader(
            cal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"Kernel-Attention: Fitting on {n_fit} samples, calibrating on {n_cal} samples")

        # --- Extract fit features ---
        print("Kernel-Attention: Extracting features from fitting set...")
        fit_features, fit_labels = [], []
        fit_features_near_buf = [] if self.dual_head else None
        for batch in tqdm(fit_loader, desc='Fitting'):
            data = batch['data'].cuda()
            if self.dual_head:
                _, feat_near, feat_far = self._extract_features_both(net, data)
                fit_features_near_buf.append(feat_near.cpu())
                fit_features.append(feat_far.cpu())
            else:
                _, feat = self._extract_features(net, data)
                fit_features.append(feat.cpu())
            fit_labels.append(batch['label'])
        fit_features = F.normalize(torch.cat(fit_features), p=2, dim=1)
        fit_labels   = torch.cat(fit_labels)
        if self.dual_head:
            fit_features_near = F.normalize(torch.cat(fit_features_near_buf), p=2, dim=1)

        feature_dim = fit_features.shape[1]
        classes     = fit_labels.unique().sort().values
        n_classes   = len(classes)
        D           = self.rff_dim
        print(f"Kernel-Attention: Feature dimension = {feature_dim} "
              f"(MRL-weighted 4-layer concat, weights={self.layer_weights}), "
              f"Classes = {n_classes}")

        # --- Sigma + RFF init ---
        if self.sigma_mode == 'median':
            self.sigma = self._compute_median_sigma(fit_features)
            print(f"Kernel-Attention: Median-heuristic σ = {self.sigma:.4f}")
        if self.sigma_scale != 1.0:
            self.sigma *= self.sigma_scale
            print(f"Kernel-Attention: Scaled σ × {self.sigma_scale} = {self.sigma:.4f}")
        self._init_rff_params(feature_dim)

        # --- Per-class mean embeddings ---
        print("Kernel-Attention: Computing per-class mean embeddings...")
        class_sums   = torch.zeros(n_classes, D)
        class_counts = torch.zeros(n_classes)
        chunk_size   = 1000
        for start in range(0, len(fit_features), chunk_size):
            end          = min(start + chunk_size, len(fit_features))
            chunk_phi    = self._compute_rff_features(fit_features[start:end])
            chunk_labels = fit_labels[start:end]
            for i, c in enumerate(classes):
                mask = chunk_labels == c
                if mask.any():
                    class_sums[i]   += chunk_phi[mask].sum(dim=0)
                    class_counts[i] += mask.sum()
        self.mu_hat = class_sums / class_counts.unsqueeze(1)

        # --- Second pass: v_c = M_c mu_hat_c (distance version) ---
        if self.kernel_weighted:
            print("Kernel-Attention: Computing kernel-weighted embeddings v_c = M_c μ̂_c (second pass)...")
            v_sums   = torch.zeros(n_classes, D)
            v_counts = torch.zeros(n_classes)
            for start in range(0, len(fit_features), chunk_size):
                end          = min(start + chunk_size, len(fit_features))
                chunk_phi    = self._compute_rff_features(fit_features[start:end])
                chunk_labels = fit_labels[start:end]
                weights      = chunk_phi @ self.mu_hat.T   # (chunk_size, C)
                for i, c in enumerate(classes):
                    mask = chunk_labels == c
                    if mask.any():
                        # weight each phi(x_i) by its similarity to the class center
                        v_sums[i]   += (chunk_phi[mask] * weights[mask, i].unsqueeze(1)).sum(dim=0)
                        v_counts[i] += mask.sum()
            self.mu_hat = v_sums / v_counts.unsqueeze(1)

        # --- Near head fit (dual_head mode only) ---
        if self.dual_head:
            print("Kernel-Attention: Fitting near-OOD head (layer3 + penultimate)...")
            if self.sigma_mode == 'median':
                self.sigma_near = self._compute_median_sigma(fit_features_near) * self.sigma_scale
                print(f"  Near head σ = {self.sigma_near:.4f}")
            else:
                self.sigma_near = self.args.sigma * self.sigma_scale

            rng_near = torch.Generator().manual_seed(43)   # different seed from far head
            self.omega_near = torch.randn(
                D, fit_features_near.shape[1], generator=rng_near) / self.sigma_near
            self.bias_near  = torch.rand(D, generator=rng_near) * 2 * np.pi

            class_sums_near   = torch.zeros(n_classes, D)
            class_counts_near = torch.zeros(n_classes)
            for start in range(0, len(fit_features_near), chunk_size):
                end          = min(start + chunk_size, len(fit_features_near))
                chunk_phi    = self._compute_rff_near(fit_features_near[start:end])
                chunk_labels = fit_labels[start:end]
                for i, c in enumerate(classes):
                    mask = chunk_labels == c
                    if mask.any():
                        class_sums_near[i]   += chunk_phi[mask].sum(dim=0)
                        class_counts_near[i] += mask.sum()
            self.mu_hat_near = class_sums_near / class_counts_near.unsqueeze(1)

        # --- Calibration ---
        print(f"Kernel-Attention: Calibrating (score_mode={self.score_mode})...")
        cal_raw_list, cal_labels_list, cal_phi_batches = [], [], []
        cal_near_raw_list = [] if self.dual_head else None

        for batch in tqdm(cal_loader, desc='Calibrating'):
            data   = batch['data'].cuda()
            labels = batch['label']
            if self.dual_head:
                _, feat_near, feat_far = self._extract_features_both(net, data)
                feat_near = F.normalize(feat_near, p=2, dim=1)
                feat_far  = F.normalize(feat_far,  p=2, dim=1)
                phi_far   = self._compute_rff_features(feat_far).cpu()
                phi_near  = self._compute_rff_near(feat_near).cpu()
                cal_phi_batches.append(phi_far)
                cal_raw_list.append(phi_far @ self.mu_hat.T)
                cal_near_raw_list.append(phi_near @ self.mu_hat_near.T)
                cal_labels_list.append(labels)
            else:
                _, feat = self._extract_features(net, data)
                feat   = F.normalize(feat, p=2, dim=1)
                phi    = self._compute_rff_features(feat).cpu()
                cal_phi_batches.append(phi)
                cal_raw_list.append(phi @ self.mu_hat.T)
                cal_labels_list.append(labels)

        cal_labels  = torch.cat(cal_labels_list)
        cal_raw_all = torch.cat(cal_raw_list, dim=0)   # (n_cal, C)

        if self.score_mode == 'per_class_threshold':
            self.class_thresholds = torch.zeros(n_classes)
            for i, c in enumerate(classes):
                mask = cal_labels == c
                if mask.any():
                    self.class_thresholds[i] = torch.quantile(cal_raw_all[mask, i], self.alpha)
            self.threshold = 0.0
        else:
            cal_scores     = cal_raw_all.max(dim=1).values
            self.threshold = torch.quantile(cal_scores, self.alpha).item()

        # --- Near head calibration thresholds ---
        if self.dual_head:
            cal_near_raw_all = torch.cat(cal_near_raw_list, dim=0)   # (n_cal, C)
            if self.score_mode == 'per_class_threshold':
                self.class_thresholds_near = torch.zeros(n_classes)
                for i, c in enumerate(classes):
                    mask = cal_labels == c
                    if mask.any():
                        self.class_thresholds_near[i] = torch.quantile(
                            cal_near_raw_all[mask, i], self.alpha)
            else:
                cal_scores_near = cal_near_raw_all.max(dim=1).values
                self.class_thresholds_near = torch.quantile(cal_scores_near, self.alpha)

        # --- Summary ---
        print(f"Kernel-Attention: Setup complete")
        print(f"  - Score mode          = {self.score_mode}")
        print(f"  - Kernel weighted     = {self.kernel_weighted}")
        print(f"  - Dual head           = {self.dual_head}" +
              (f"  (α_far={self.head_alpha})" if self.dual_head else ""))
        print(f"  - Sigma mode          = {self.sigma_mode}")
        print(f"  - Far head σ          = {self.sigma:.4f}")
        if self.dual_head:
            print(f"  - Near head σ         = {self.sigma_near:.4f}")
        print(f"  - RFF dimension D     = {self.rff_dim}")
        print(f"  - Number of classes C = {n_classes}")
        if self.score_mode == 'per_class_threshold':
            print(f"  - Far τ_c range       = [{self.class_thresholds.min():.4f}, {self.class_thresholds.max():.4f}]")
            if self.dual_head:
                print(f"  - Near τ_c range      = [{self.class_thresholds_near.min():.4f}, {self.class_thresholds_near.max():.4f}]")
        else:
            print(f"  - Threshold τ (α={self.alpha}) = {self.threshold:.6f}")

        # --- PI Diagnostics ---
        print("\n--- PI Debugging Diagnostics ---")

        # Check 1: phi mean/std
        fit_phi_sample = self._compute_rff_features(fit_features[:min(2000, len(fit_features))])
        cal_phi_all    = torch.cat(cal_phi_batches)
        print(f"  [1] φ(x) mean / std:")
        print(f"      train-fit : mean={fit_phi_sample.mean():.4f}  std={fit_phi_sample.std():.4f}")
        print(f"      train-cal : mean={cal_phi_all.mean():.4f}  std={cal_phi_all.std():.4f}")
        print(f"      (run get_phi_stats() for test-ID and test-OOD)")

        # Check 3: approximate kernel values
        n   = len(fit_features)
        rng = torch.Generator().manual_seed(99)
        idx_i = torch.randint(0, n, (1000,), generator=rng)
        idx_j = torch.randint(0, n, (1000,), generator=rng)
        phi_i = self._compute_rff_features(fit_features[idx_i])
        phi_j = self._compute_rff_features(fit_features[idx_j])
        kvals = (phi_i * phi_j).sum(dim=1)
        print(f"  [3] k(x_i, x_j) over 1000 random train pairs:")
        print(f"      mean={kvals.mean():.4f}  std={kvals.std():.4f}  "
              f"min={kvals.min():.4f}  max={kvals.max():.4f}")

        self.setup_flag = True

    # ------------------------------------------------------------------
    # Latency benchmarking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def benchmark_latency(self, net: nn.Module, loader,
                          n_warmup: int = 5, n_batches: int = 20) -> dict:
        """Measure per-sample inference latency (feature extraction + RFF + scoring).

        Uses CUDA synchronisation so GPU-side work is fully counted.
        Returns a dict with mean_ms, std_ms, p50_ms, p95_ms, p99_ms,
        batch_ms_mean, batch_size, n_samples.
        """
        net.eval()
        batch_times = []
        batch_size  = None
        use_cuda    = torch.cuda.is_available()
        it          = iter(loader)

        def _next_batch():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(loader)
                return next(it)

        # Warm-up: fill GPU caches, exclude JIT / kernel-launch overhead
        for _ in range(n_warmup):
            batch = _next_batch()
            data  = batch['data'].cuda()
            if batch_size is None:
                batch_size = data.shape[0]
            if use_cuda:
                torch.cuda.synchronize()
            self.postprocess(net, data)
            if use_cuda:
                torch.cuda.synchronize()

        # Timed runs
        for _ in range(n_batches):
            batch = _next_batch()
            data  = batch['data'].cuda()
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.postprocess(net, data)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)   # ms per batch

        batch_times = np.array(batch_times)
        per_sample  = batch_times / batch_size        # ms per sample

        return {
            'mean_ms':      float(per_sample.mean()),
            'std_ms':       float(per_sample.std()),
            'p50_ms':       float(np.percentile(per_sample, 50)),
            'p95_ms':       float(np.percentile(per_sample, 95)),
            'p99_ms':       float(np.percentile(per_sample, 99)),
            'batch_ms_mean': float(batch_times.mean()),
            'batch_size':   int(batch_size),
            'n_samples':    int(n_batches * batch_size),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: torch.Tensor):
        if self.dual_head:
            logits, feat_near, feat_far = self._extract_features_both(net, data)
            feat_near = F.normalize(feat_near, p=2, dim=1)
            feat_far  = F.normalize(feat_far,  p=2, dim=1)
            score_near = self._score_near(self._compute_rff_near(feat_near))
            score_far  = self._score(self._compute_rff_features(feat_far))
            conf = (1.0 - self.head_alpha) * score_near + self.head_alpha * score_far
        else:
            logits, feat = self._extract_features(net, data)
            feat = F.normalize(feat, p=2, dim=1)
            conf = self._score(self._compute_rff_features(feat))
        pred = logits.argmax(dim=1)
        return pred, conf
