from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


def _select_from_list(values, layer_idx, pos, default='none'):
    if values is None:
        return default
    if not isinstance(values, (list, tuple)):
        return values
    if len(values) == 0:
        return default
    if len(values) == 1:
        return values[0]
    if pos < len(values):
        return values[pos]
    if layer_idx < len(values):
        return values[layer_idx]
    return default


def _aggregate_scores(score_tensor: torch.Tensor,
                      mode: str,
                      weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if score_tensor.ndim != 2:
        raise ValueError('score_tensor must be [batch, layers]')

    if score_tensor.shape[1] == 0:
        return torch.zeros(score_tensor.shape[0], device=score_tensor.device)

    if mode == 'max':
        return score_tensor.max(dim=1).values
    if mode == 'mean':
        return score_tensor.mean(dim=1)
    if mode == 'weighted':
        if weights is None or weights.numel() != score_tensor.shape[1]:
            weights = torch.ones(score_tensor.shape[1], device=score_tensor.device)
        denom = weights.abs().sum()
        if denom.item() < 1e-8:
            weights = torch.ones_like(weights)
            denom = weights.sum()
        w = weights / denom
        return (score_tensor * w.unsqueeze(0)).sum(dim=1)

    return score_tensor.max(dim=1).values


def _safe_std(x: np.ndarray, eps: float = 1e-8) -> float:
    if x.size == 0:
        return 1.0
    s = float(np.std(x))
    return s if s > eps else 1.0


def reduce_feature_dim(feature_list_full,
                       label_list_full,
                       feature_process,
                       kept_dim=None):
    if feature_process == 'none':
        transform_matrix = np.eye(feature_list_full.shape[1], dtype=np.float32)
        kept_dim = feature_list_full.shape[1]
    else:
        if kept_dim is None:
            feature_process, kept_dim_str = feature_process.split('_')
            kept_dim = int(kept_dim_str)

        if feature_process == 'pca':
            pca = PCA(n_components=min(kept_dim, feature_list_full.shape[1]))
            pca.fit(feature_list_full)
            transform_matrix = pca.components_.T.astype(np.float32)
        elif feature_process == 'lda':
            lda = LinearDiscriminantAnalysis(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :min(
                kept_dim, lda.scalings_.shape[1])].astype(np.float32)
        else:
            transform_matrix = np.eye(
                feature_list_full.shape[1], dtype=np.float32)
            kept_dim = feature_list_full.shape[1]

    return transform_matrix.astype(np.float32), int(kept_dim)


def alpha_selector(data_in, data_out):
    label_in = np.ones(len(data_in))
    label_out = np.zeros(len(data_out))
    data = np.concatenate([data_in, data_out], axis=0)
    label = np.concatenate([label_in, label_out], axis=0)

    try:
        lr = LogisticRegressionCV(
            n_jobs=-1, max_iter=1000).fit(data, label)
        alpha_list = lr.coef_.reshape(-1)
    except Exception:
        alpha_list = np.ones(data_in.shape[1], dtype=np.float32)

    print(f'[RFF-MultiLayer-Hybrid] Alpha list: {alpha_list}')
    return alpha_list


def _compute_pairwise_median_sigma(features_np: np.ndarray,
                                   max_points: int = 2000) -> float:
    n = features_np.shape[0]
    if n == 0:
        return 1.0
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
        features_np = features_np[idx]
    x = torch.from_numpy(features_np).float()
    d = torch.cdist(x, x, p=2)
    vals = d[d > 0]
    if vals.numel() == 0:
        return 1.0
    return max(vals.median().item(), 1e-6)


def _compute_far_layer_score(similarities: torch.Tensor,
                             score_mode_far: str,
                             class_thresholds: Optional[torch.Tensor]) -> torch.Tensor:
    if score_mode_far == 'per_class_threshold' and class_thresholds is not None:
        return (similarities - class_thresholds.unsqueeze(0)).max(dim=1).values
    return similarities.max(dim=1).values


class RFFMultiLayerPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RFFMultiLayerPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.kernel_bandwidth = self.args.kernel_bandwidth
        self.feature_dim = self.args.feature_dim
        self.target_rate = self.args.target_rate

        self.layer_indices = list(self.args.layer_indices)
        self.reduce_dim_list = list(self.args.reduce_dim_list)

        # Existing knobs
        self.near_ood_layers = list(
            getattr(self.args, 'near_ood_layers', self.layer_indices))
        self.far_ood_layers = list(
            getattr(self.args, 'far_ood_layers', self.layer_indices))
        self.scoring_mode = getattr(self.args, 'scoring_mode', 'max')

        # Friend-style knobs
        self.sigma_mode = getattr(self.args, 'sigma_mode', 'auto')
        self.sigma_scale = float(getattr(self.args, 'sigma_scale', 1.0))
        self.far_score_mode = getattr(
            self.args, 'far_score_mode', 'per_class_threshold')
        self.far_threshold_alpha = float(
            getattr(self.args, 'far_threshold_alpha', self.target_rate))
        self.layer_weights = list(
            getattr(self.args, 'layer_weights', [1.0] * len(self.layer_indices)))

        # Fusion knobs (NearOOD-safe)
        self.dual_head = bool(getattr(self.args, 'dual_head', True))
        self.head_alpha = float(getattr(self.args, 'head_alpha', 0.5))
        self.gate_margin = float(getattr(self.args, 'gate_margin', 0.10))
        self.near_preserve = bool(getattr(self.args, 'near_preserve', True))
        self.apply_layer_weights = bool(
            getattr(self.args, 'apply_layer_weights', True))
        self.use_ood_for_alpha = bool(
            getattr(self.args, 'use_ood_for_alpha', True))

        self.layer_to_pos = {layer_idx: i for i,
                             layer_idx in enumerate(self.layer_indices)}

        # Calibrated
        self.layer_params = {}
        self.class_mean_embeddings_per_layer = {}
        self.class_thresholds_per_layer = {}
        self.alpha_list = []

        self.near_score_mean = 0.0
        self.near_score_std = 1.0
        self.far_score_mean = 0.0
        self.far_score_std = 1.0

        # Keep old names for compatibility
        self.score_mean = 0.0
        self.score_std = 1.0

        self.setup_flag = False
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def _alpha_for_layer(self, layer_idx: int) -> float:
        pos = self.layer_to_pos[layer_idx]
        if pos < len(self.alpha_list):
            return float(self.alpha_list[pos])
        return 1.0

    def _layer_weight_vector(self, active_layers: List[int], device):
        vals = []
        for layer_idx in active_layers:
            pos = self.layer_to_pos[layer_idx]
            if pos < len(self.layer_weights):
                vals.append(float(self.layer_weights[pos]))
            else:
                vals.append(1.0)
        w = torch.tensor(vals, dtype=torch.float32, device=device)
        if w.abs().sum().item() < 1e-8:
            w = torch.ones_like(w)
        return w

    def _alpha_weight_vector(self, active_layers: List[int], device):
        vals = [self._alpha_for_layer(layer_idx) for layer_idx in active_layers]
        w = torch.tensor(vals, dtype=torch.float32, device=device)
        if w.abs().sum().item() < 1e-8:
            w = torch.ones_like(w)
        return w

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        num_classes = None

        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'num_classes'):
            num_classes = self.config.dataset.num_classes
        elif hasattr(self.config, 'num_classes'):
            num_classes = self.config.num_classes

        if num_classes is None:
            for attr in ['fc', 'head', 'classifier', 'linear', 'output']:
                layer = getattr(net, attr, None)
                if layer is not None and hasattr(layer, 'out_features'):
                    num_classes = layer.out_features
                    break

        if num_classes is None and 'train' in id_loader_dict:
            try:
                sample_batch = next(iter(id_loader_dict["train"]))
                labels = sample_batch["label"]
                num_classes = int(labels.max().item()) + 1
            except Exception:
                pass

        if num_classes is None:
            dataset_name = ''
            if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'name'):
                dataset_name = str(self.config.dataset.name).lower()
            if 'cifar100' in dataset_name:
                num_classes = 100
            elif 'cifar10' in dataset_name:
                num_classes = 10
            elif 'imagenet' in dataset_name:
                num_classes = 1000
            else:
                num_classes = 100

        print(f"[RFF-MultiLayer-Hybrid] num_classes = {num_classes}")

        (
            self.layer_params,
            self.class_mean_embeddings_per_layer,
            self.alpha_list,
            self.class_thresholds_per_layer,
            self.near_score_mean,
            self.near_score_std,
            self.far_score_mean,
            self.far_score_std,
        ) = calibrate_multi_layer_model(
            model=net,
            train_loader=id_loader_dict["train"],
            val_loader_id=id_loader_dict["val"],
            val_loader_ood=ood_loader_dict["val"],
            target_rate=self.target_rate,
            kernel_bandwidth=self.kernel_bandwidth,
            feature_dim=self.feature_dim,
            layer_indices=self.layer_indices,
            reduce_dim_list=self.reduce_dim_list,
            near_ood_layers=self.near_ood_layers,
            far_ood_layers=self.far_ood_layers,
            scoring_mode=self.scoring_mode,
            num_classes=num_classes,
            sigma_mode=self.sigma_mode,
            sigma_scale=self.sigma_scale,
            far_score_mode=self.far_score_mode,
            far_threshold_alpha=self.far_threshold_alpha,
            use_ood_for_alpha=self.use_ood_for_alpha,
            layer_weights=self.layer_weights,
            apply_layer_weights=self.apply_layer_weights,
        )

        self.score_mean = self.near_score_mean
        self.score_std = self.near_score_std
        self.setup_flag = True

    def _extract_all_features(self, model, data):
        logits, all_features = model(data, return_feature_list=True)

        processed_features = []
        for layer_idx in self.layer_indices:
            if layer_idx < len(all_features):
                feat = all_features[layer_idx]
                feat = torch.flatten(feat, start_dim=1)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                processed_features.append(feat)
            else:
                warnings.warn(f"Layer index {layer_idx} not available in model")
                processed_features.append(None)

        return logits, processed_features

    def _compute_rff(self, features, layer_idx):
        if features is None:
            return None

        params = self.layer_params[layer_idx]
        random_omega = params['random_omega'].to(features.device)
        random_bias = params['random_bias'].to(features.device)
        feature_dim = params['feature_dim']
        transform_matrix = params['transform_matrix']

        if isinstance(transform_matrix, np.ndarray):
            transform_matrix = torch.from_numpy(
                transform_matrix).float().to(features.device)
        else:
            transform_matrix = transform_matrix.float().to(features.device)

        features_reduced = features @ transform_matrix

        phi = torch.sqrt(torch.tensor(
            2.0 / feature_dim, device=features.device)) * torch.cos(
            features_reduced @ random_omega.T + random_bias)
        phi = torch.nn.functional.normalize(phi, p=2, dim=1)
        return phi

    @torch.no_grad()
    def postprocess(self, model: nn.Module, data: Any):
        logits, features_list = self._extract_all_features(model, data)

        batch_size = logits.size(0)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        energy = torch.logsumexp(logits, dim=1)
        max_prob = probs.max(dim=1).values

        active_near_layers = [l for l in self.near_ood_layers if l in self.layer_indices]
        active_far_layers = [l for l in self.far_ood_layers if l in self.layer_indices]
        if len(active_near_layers) == 0:
            active_near_layers = list(self.layer_indices)
        if len(active_far_layers) == 0:
            active_far_layers = list(self.layer_indices)

        near_sim_scores = []
        near_energy_scores = []
        near_dist_scores = []
        near_maxprob_scores = []
        near_layer_order = []

        far_scores = []
        far_layer_order = []

        for i, layer_idx in enumerate(self.layer_indices):
            feat_i = features_list[i]
            if feat_i is None:
                continue

            phi_x = self._compute_rff(feat_i, layer_idx)
            if phi_x is None:
                continue

            class_means = self.class_mean_embeddings_per_layer[layer_idx].to(logits.device)
            alpha = self._alpha_for_layer(layer_idx)
            similarities = alpha * (phi_x @ class_means.T)

            if layer_idx in active_near_layers:
                pred_sims = similarities[torch.arange(batch_size, device=logits.device), pred]
                raw_conf = pred_sims - entropy
                batch_means = class_means[pred]
                dist = torch.norm(phi_x - batch_means, p=2, dim=1)

                near_sim_scores.append(raw_conf)
                near_energy_scores.append(energy)
                near_dist_scores.append(-dist)
                near_maxprob_scores.append(max_prob)
                near_layer_order.append(layer_idx)

            if layer_idx in active_far_layers:
                thresholds = None
                if layer_idx in self.class_thresholds_per_layer:
                    thresholds = self.class_thresholds_per_layer[layer_idx].to(logits.device)
                far_score_layer = _compute_far_layer_score(
                    similarities=similarities,
                    score_mode_far=self.far_score_mode,
                    class_thresholds=thresholds,
                )
                far_scores.append(far_score_layer)
                far_layer_order.append(layer_idx)

        if len(near_sim_scores) == 0:
            near_conf = torch.zeros(batch_size, device=logits.device)
        else:
            sim_t = torch.stack(near_sim_scores, dim=1)
            en_t = torch.stack(near_energy_scores, dim=1)
            dist_t = torch.stack(near_dist_scores, dim=1)
            mp_t = torch.stack(near_maxprob_scores, dim=1)

            near_weights = None
            if self.scoring_mode == 'weighted':
                near_weights = self._alpha_weight_vector(
                    near_layer_order, logits.device)

            conf_sim = _aggregate_scores(sim_t, self.scoring_mode, near_weights)
            conf_en = _aggregate_scores(en_t, self.scoring_mode, near_weights)
            conf_dist = _aggregate_scores(dist_t, self.scoring_mode, near_weights)
            conf_mp = _aggregate_scores(mp_t, self.scoring_mode, near_weights)

            near_raw = 0.25 * conf_sim + 0.35 * conf_en + 0.25 * conf_dist + 0.15 * conf_mp
            near_conf = (near_raw - self.near_score_mean) / (self.near_score_std + 1e-8)

        if len(far_scores) == 0:
            far_conf = near_conf
        else:
            far_t = torch.stack(far_scores, dim=1)
            if self.apply_layer_weights and far_t.shape[1] > 1:
                far_weights = self._layer_weight_vector(
                    far_layer_order, logits.device)
                far_raw = _aggregate_scores(far_t, 'weighted', far_weights)
            else:
                far_raw = far_t.mean(dim=1)
            far_conf = (far_raw - self.far_score_mean) / (self.far_score_std + 1e-8)

        if not self.dual_head:
            conf = near_conf
        else:
            if self.near_preserve:
                # Only decrease confidence when far head is sufficiently lower.
                delta = far_conf - near_conf + self.gate_margin
                adjust = torch.clamp(delta, max=0.0)
                conf = near_conf + self.head_alpha * adjust
            else:
                conf = (1.0 - self.head_alpha) * near_conf + self.head_alpha * far_conf

        return pred, conf


@torch.no_grad()
def calibrate_multi_layer_model(
    model,
    train_loader,
    val_loader_id,
    val_loader_ood,
    target_rate,
    kernel_bandwidth,
    feature_dim,
    layer_indices,
    reduce_dim_list,
    near_ood_layers,
    far_ood_layers,
    scoring_mode,
    num_classes,
    sigma_mode='auto',
    sigma_scale=1.0,
    far_score_mode='per_class_threshold',
    far_threshold_alpha=0.05,
    use_ood_for_alpha=True,
    layer_weights=None,
    apply_layer_weights=True,
):
    model.eval()
    device = next(model.parameters()).device

    if layer_weights is None:
        layer_weights = [1.0] * len(layer_indices)
    layer_to_pos = {layer_idx: i for i, layer_idx in enumerate(layer_indices)}

    print(f"[RFF-MultiLayer-Hybrid] Extracting features from layers: {layer_indices}")

    train_features_per_layer = {i: [] for i in layer_indices}
    train_labels = []

    for batch in tqdm(train_loader, desc='[Hybrid] Train feature extraction'):
        data = batch['data'].to(device)
        labels = batch['label'].to(device)
        _, all_features = model(data, return_feature_list=True)
        train_labels.append(labels.cpu())

        for layer_idx in layer_indices:
            if layer_idx < len(all_features):
                feat = all_features[layer_idx]
                feat = torch.flatten(feat, start_dim=1)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                train_features_per_layer[layer_idx].append(feat.cpu())

    train_labels = torch.cat(train_labels, dim=0).numpy()

    val_features_per_layer = {i: [] for i in layer_indices}
    val_labels = []
    val_logits = []

    for batch in tqdm(val_loader_id, desc='[Hybrid] ID val feature extraction'):
        data = batch['data'].to(device)
        labels = batch['label'].to(device)
        logits, all_features = model(data, return_feature_list=True)

        val_labels.append(labels.cpu())
        val_logits.append(logits.cpu())

        for layer_idx in layer_indices:
            if layer_idx < len(all_features):
                feat = all_features[layer_idx]
                feat = torch.flatten(feat, start_dim=1)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                val_features_per_layer[layer_idx].append(feat.cpu())

    val_labels = torch.cat(val_labels, dim=0).numpy()
    val_logits = torch.cat(val_logits, dim=0)

    ood_features_per_layer = {i: [] for i in layer_indices}
    for batch in tqdm(val_loader_ood, desc='[Hybrid] OOD val feature extraction'):
        data = batch['data'].to(device)
        _, all_features = model(data, return_feature_list=True)

        for layer_idx in layer_indices:
            if layer_idx < len(all_features):
                feat = all_features[layer_idx]
                feat = torch.flatten(feat, start_dim=1)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                ood_features_per_layer[layer_idx].append(feat.cpu())

    layer_params = {}
    class_mean_embeddings_per_layer = {}
    transform_matrices = {}

    for pos, layer_idx in enumerate(layer_indices):
        train_features = torch.cat(train_features_per_layer[layer_idx], dim=0).numpy()
        reduce_method = _select_from_list(
            reduce_dim_list, layer_idx, pos, default='none')

        transform_matrix, reduced_dim = reduce_feature_dim(
            train_features, train_labels, reduce_method)
        transform_matrices[layer_idx] = transform_matrix

        train_features_reduced = train_features @ transform_matrix
        median_sigma = _compute_pairwise_median_sigma(train_features_reduced)

        best_sigma = median_sigma
        best_alpha = 1.0

        if sigma_mode == 'median':
            best_sigma = median_sigma
        elif isinstance(kernel_bandwidth, str) and kernel_bandwidth == 'auto':
            sigma_grid = [0.25 * median_sigma, 0.5 * median_sigma,
                          1.0 * median_sigma, 2.0 * median_sigma, 4.0 * median_sigma]
            alpha_grid = [1.0, 5.0, 10.0, 20.0, 50.0]
            best_nll = float('inf')

            val_features = torch.cat(
                val_features_per_layer[layer_idx], dim=0).numpy()
            val_features_reduced = val_features @ transform_matrix

            for sig in sigma_grid:
                omega = (torch.randn(feature_dim, reduced_dim, device=device) / sig)
                bias = (torch.rand(feature_dim, device=device) * 2 * np.pi)

                train_phi = []
                for i in range(0, len(train_features_reduced), 1024):
                    chunk = torch.from_numpy(
                        train_features_reduced[i:i + 1024]).float().to(device)
                    phi = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * torch.cos(
                        chunk @ omega.T + bias)
                    phi = torch.nn.functional.normalize(phi, p=2, dim=1)
                    train_phi.append(phi)
                train_phi = torch.cat(train_phi, dim=0)

                c_means = torch.zeros(num_classes, feature_dim, device=device)
                tlabels = torch.from_numpy(train_labels).to(device)
                for c in range(num_classes):
                    mask = (tlabels == c)
                    if mask.any():
                        c_means[c] = torch.nn.functional.normalize(
                            train_phi[mask].mean(dim=0), dim=0)

                val_phi = []
                for i in range(0, len(val_features_reduced), 1024):
                    chunk = torch.from_numpy(
                        val_features_reduced[i:i + 1024]).float().to(device)
                    phi = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * torch.cos(
                        chunk @ omega.T + bias)
                    phi = torch.nn.functional.normalize(phi, p=2, dim=1)
                    val_phi.append(phi)
                val_phi = torch.cat(val_phi, dim=0)

                sims = val_phi @ c_means.T
                for alp in alpha_grid:
                    nll = torch.nn.functional.cross_entropy(
                        sims * alp, torch.from_numpy(val_labels).long().to(device)).item()
                    if nll < best_nll:
                        best_nll = nll
                        best_sigma = sig
                        best_alpha = alp
        else:
            if isinstance(kernel_bandwidth, (float, int)):
                best_sigma = float(kernel_bandwidth)
            else:
                best_sigma = 1.0

        best_sigma = max(best_sigma * sigma_scale, 1e-6)
        print(f'[Hybrid] Layer {layer_idx}: sigma={best_sigma:.4f}, alpha={best_alpha:.4f}')

        random_omega = (torch.randn(feature_dim, reduced_dim, device=device) / best_sigma)
        random_bias = (torch.rand(feature_dim, device=device) * 2 * np.pi)

        layer_params[layer_idx] = {
            'random_omega': random_omega.detach().cpu(),
            'random_bias': random_bias.detach().cpu(),
            'feature_dim': feature_dim,
            'alpha': best_alpha,
            'reduced_dim': reduced_dim,
            'transform_matrix': transform_matrix.astype(np.float32),
        }

        train_phi = []
        for i in range(0, len(train_features_reduced), 1024):
            chunk = torch.from_numpy(
                train_features_reduced[i:i + 1024]).float().to(device)
            phi = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * torch.cos(
                chunk @ random_omega.T + random_bias)
            phi = torch.nn.functional.normalize(phi, p=2, dim=1)
            train_phi.append(phi.cpu())
        train_phi = torch.cat(train_phi, dim=0)

        class_mean_embeddings = torch.zeros(num_classes, feature_dim, device=device)
        train_labels_t = torch.from_numpy(train_labels)
        for c in range(num_classes):
            mask = (train_labels_t == c)
            if mask.any():
                class_mean = torch.nn.functional.normalize(
                    train_phi[mask].to(device).mean(dim=0), dim=0)
                class_mean_embeddings[c] = class_mean
        class_mean_embeddings_per_layer[layer_idx] = class_mean_embeddings.detach().cpu()

    id_sim_score = {i: [] for i in layer_indices}
    ood_sim_score = {i: [] for i in layer_indices}
    id_energy = {i: [] for i in layer_indices}
    ood_energy = {i: [] for i in layer_indices}
    id_dist = {i: [] for i in layer_indices}
    ood_dist = {i: [] for i in layer_indices}
    id_maxprob = {i: [] for i in layer_indices}
    ood_maxprob = {i: [] for i in layer_indices}

    id_similarity_full = {i: [] for i in layer_indices}
    ood_similarity_full = {i: [] for i in layer_indices}

    chunk_size = 1024

    for layer_idx in layer_indices:
        val_features = torch.cat(val_features_per_layer[layer_idx], dim=0).numpy()
        val_features_reduced = val_features @ transform_matrices[layer_idx]

        params = layer_params[layer_idx]
        random_omega = params['random_omega'].to(device)
        random_bias = params['random_bias'].to(device)
        feature_dim_layer = params['feature_dim']
        class_means = class_mean_embeddings_per_layer[layer_idx].to(device)
        alpha = params['alpha']

        for i in range(0, len(val_features_reduced), chunk_size):
            chunk = torch.from_numpy(
                val_features_reduced[i:i + chunk_size]).float().to(device)
            chunk_logits = val_logits[i:i + chunk_size].to(device)

            phi_x = torch.sqrt(torch.tensor(2.0 / feature_dim_layer, device=device)) * torch.cos(
                chunk @ random_omega.T + random_bias)
            phi_x = torch.nn.functional.normalize(phi_x, p=2, dim=1)

            similarities = alpha * (phi_x @ class_means.T)
            id_similarity_full[layer_idx].append(similarities.detach().cpu())

            probs = torch.softmax(chunk_logits, dim=1)
            preds = chunk_logits.argmax(dim=1)
            pred_sims = similarities[torch.arange(chunk.size(0), device=device), preds]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            id_sim_score[layer_idx].append((pred_sims - entropy).detach().cpu())
            id_energy[layer_idx].append(torch.logsumexp(chunk_logits, dim=1).detach().cpu())
            batch_means = class_means[preds]
            id_dist[layer_idx].append((-torch.norm(phi_x - batch_means, p=2, dim=1)).detach().cpu())
            id_maxprob[layer_idx].append(probs.max(dim=1).values.detach().cpu())

        if layer_idx in ood_features_per_layer and len(ood_features_per_layer[layer_idx]) > 0:
            ood_features = torch.cat(ood_features_per_layer[layer_idx], dim=0).numpy()
            ood_features_reduced = ood_features @ transform_matrices[layer_idx]

            for i in range(0, len(ood_features_reduced), chunk_size):
                chunk = torch.from_numpy(
                    ood_features_reduced[i:i + chunk_size]).float().to(device)

                phi_x = torch.sqrt(torch.tensor(2.0 / feature_dim_layer, device=device)) * torch.cos(
                    chunk @ random_omega.T + random_bias)
                phi_x = torch.nn.functional.normalize(phi_x, p=2, dim=1)

                similarities = alpha * (phi_x @ class_means.T)
                ood_similarity_full[layer_idx].append(similarities.detach().cpu())

                preds = similarities.argmax(dim=1)
                pred_sims = similarities[torch.arange(chunk.size(0), device=device), preds]

                ood_sim_score[layer_idx].append(pred_sims.detach().cpu())
                ood_energy[layer_idx].append(torch.logsumexp(similarities, dim=1).detach().cpu())
                batch_means = class_means[preds]
                ood_dist[layer_idx].append((-torch.norm(phi_x - batch_means, p=2, dim=1)).detach().cpu())
                ood_probs = torch.softmax(similarities, dim=1)
                ood_maxprob[layer_idx].append(ood_probs.max(dim=1).values.detach().cpu())

    def _concat_feature_matrix(score_dict):
        mats = []
        for layer_idx in layer_indices:
            if len(score_dict[layer_idx]) > 0:
                mats.append(torch.cat(score_dict[layer_idx], dim=0).numpy().reshape(-1, 1))
        if len(mats) == 0:
            return None
        return np.concatenate(mats, axis=1)

    id_concat = _concat_feature_matrix(id_sim_score)
    ood_concat = _concat_feature_matrix(ood_sim_score)

    if use_ood_for_alpha and id_concat is not None and ood_concat is not None and len(
            ood_concat) > 0:
        alpha_list = alpha_selector(id_concat, ood_concat)
    else:
        alpha_list = np.ones(len(layer_indices), dtype=np.float32)

    def _build_near_raw(split_sim, split_energy, split_dist, split_maxprob, active_layers):
        sim_l, en_l, dist_l, mp_l = [], [], [], []
        for layer_idx in active_layers:
            if len(split_sim[layer_idx]) == 0:
                continue
            sim_l.append(torch.cat(split_sim[layer_idx], dim=0))
            en_l.append(torch.cat(split_energy[layer_idx], dim=0))
            dist_l.append(torch.cat(split_dist[layer_idx], dim=0))
            mp_l.append(torch.cat(split_maxprob[layer_idx], dim=0))

        if len(sim_l) == 0:
            return np.zeros((1,), dtype=np.float32)

        sim_t = torch.stack(sim_l, dim=1).float()
        en_t = torch.stack(en_l, dim=1).float()
        dist_t = torch.stack(dist_l, dim=1).float()
        mp_t = torch.stack(mp_l, dim=1).float()

        w = None
        if scoring_mode == 'weighted':
            aw = []
            for layer_idx in active_layers:
                pos = layer_to_pos[layer_idx]
                aw.append(float(alpha_list[pos]) if pos < len(alpha_list) else 1.0)
            w = torch.tensor(aw, dtype=torch.float32)

        conf_sim = _aggregate_scores(sim_t, scoring_mode, w)
        conf_en = _aggregate_scores(en_t, scoring_mode, w)
        conf_dist = _aggregate_scores(dist_t, scoring_mode, w)
        conf_mp = _aggregate_scores(mp_t, scoring_mode, w)

        near_raw = 0.25 * conf_sim + 0.35 * conf_en + 0.25 * conf_dist + 0.15 * conf_mp
        return near_raw.numpy()

    active_near_layers = [l for l in near_ood_layers if l in layer_indices]
    active_far_layers = [l for l in far_ood_layers if l in layer_indices]
    if len(active_near_layers) == 0:
        active_near_layers = list(layer_indices)
    if len(active_far_layers) == 0:
        active_far_layers = list(layer_indices)

    near_id_raw = _build_near_raw(
        id_sim_score, id_energy, id_dist, id_maxprob, active_near_layers)
    near_ood_raw = _build_near_raw(
        ood_sim_score, ood_energy, ood_dist, ood_maxprob, active_near_layers)
    near_all = np.concatenate([near_id_raw, near_ood_raw], axis=0)
    near_mean = float(np.mean(near_all))
    near_std = _safe_std(near_all)

    class_thresholds_per_layer = {}
    val_labels_t = torch.from_numpy(val_labels).long()

    for layer_idx in layer_indices:
        if layer_idx not in active_far_layers:
            continue

        if len(id_similarity_full[layer_idx]) == 0:
            class_thresholds_per_layer[layer_idx] = torch.zeros(num_classes)
            continue

        sims_id = torch.cat(id_similarity_full[layer_idx], dim=0).float()
        thresholds = torch.zeros(num_classes, dtype=torch.float32)

        if far_score_mode == 'per_class_threshold':
            for c in range(num_classes):
                mask = (val_labels_t == c)
                if mask.any():
                    thresholds[c] = torch.quantile(
                        sims_id[mask, c], q=far_threshold_alpha)
                else:
                    thresholds[c] = 0.0
        class_thresholds_per_layer[layer_idx] = thresholds

    def _build_far_raw(sim_full_dict, active_layers):
        per_layer = []
        used_layers = []

        for layer_idx in active_layers:
            if len(sim_full_dict[layer_idx]) == 0:
                continue

            sims = torch.cat(sim_full_dict[layer_idx], dim=0).float()
            thresholds = class_thresholds_per_layer.get(layer_idx, None)
            score_l = _compute_far_layer_score(
                similarities=sims,
                score_mode_far=far_score_mode,
                class_thresholds=thresholds,
            )
            per_layer.append(score_l)
            used_layers.append(layer_idx)

        if len(per_layer) == 0:
            return np.zeros((1,), dtype=np.float32)

        t = torch.stack(per_layer, dim=1)
        if apply_layer_weights and t.shape[1] > 1:
            lw = []
            for layer_idx in used_layers:
                pos = layer_to_pos[layer_idx]
                lw.append(float(layer_weights[pos]) if pos < len(layer_weights) else 1.0)
            lw = torch.tensor(lw, dtype=torch.float32)
            if lw.abs().sum().item() < 1e-8:
                lw = torch.ones_like(lw)
            far_raw = _aggregate_scores(t, 'weighted', lw)
        else:
            far_raw = t.mean(dim=1)

        return far_raw.numpy()

    far_id_raw = _build_far_raw(id_similarity_full, active_far_layers)
    far_ood_raw = _build_far_raw(ood_similarity_full, active_far_layers)
    far_all = np.concatenate([far_id_raw, far_ood_raw], axis=0)
    far_mean = float(np.mean(far_all))
    far_std = _safe_std(far_all)

    print(
        '[RFF-MultiLayer-Hybrid] Score stats: '
        f'near_mean={near_mean:.4f}, near_std={near_std:.4f}, '
        f'far_mean={far_mean:.4f}, far_std={far_std:.4f}'
    )

    return (
        layer_params,
        class_mean_embeddings_per_layer,
        alpha_list.tolist(),
        class_thresholds_per_layer,
        near_mean,
        near_std,
        far_mean,
        far_std,
    )