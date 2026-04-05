from __future__ import annotations
from typing import Any, Optional, Union
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import power_transform
from .base_postprocessor import BasePostprocessor
from torch.utils.data import DataLoader, TensorDataset


import contextlib
import math
from typing import Optional, Tuple
import cuml
from cuml.manifold.umap import fuzzy_simplicial_set, UMAP
from cuml.decomposition import IncrementalPCA
from cuml import KMeans


class ParametricUMAP(torch.nn.Module):
    def __init__(self, input_dim, z_dim=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, z_dim),
        )

    def forward(self, x):
        return self.encoder(x)


def umap_loss(pos_dist, neg_dist, weights, a=1.57, b=0.89, neg_sample_rate=5, eps=1e-7):
    # pos_dist, neg_dist: squared L2 distances (batch_size,)
    # weights: edge weights w_ij from sparse graph (batch_size,)

    # --- Attraction: -w_ij * log(q_ij) ---
    # q = 1 / (1 + a*d^2b)  =>  log(q) = -log(1 + a*d^2b)
    log_q_pos = -torch.log(1.0 + a * torch.pow(pos_dist, b) + eps)
    attraction = -weights * log_q_pos

    # --- Repulsion: -lambda * (1 - w_ij) * log(1 - q_ij) ---
    # 1 - q = a*d^2b / (1 + a*d^2b)
    # w_ij ≈ 0 for random negatives, so (1 - w_ij) ≈ 1
    d_b = a * torch.pow(neg_dist, b) + eps
    log_one_minus_q = torch.log(d_b / (1.0 + d_b))
    repulsion = neg_sample_rate * (-log_one_minus_q)

    return torch.mean(attraction + repulsion)


class FeatureStandardizer:
    def __init__(self, device="cuda"):
        self.device = device
        self.n_samples = 0
        self.running_sum = None
        self.running_sum_sq = None
        self.mean = None
        self.std = None

    def update(self, x):
        # Initialize tensors on the first batch
        if self.running_sum is None:
            self.running_sum = torch.zeros(x.size(1), device=self.device)
            self.running_sum_sq = torch.zeros(x.size(1), device=self.device)

        self.n_samples += x.size(0)
        self.running_sum += torch.sum(x, dim=0)
        self.running_sum_sq += torch.sum(x**2, dim=0)

    def finalize(self, eps=1e-8):
        self.mean = self.running_sum / self.n_samples
        # Variance formula: E[X^2] - (E[X])^2
        variance = (self.running_sum_sq / self.n_samples) - (self.mean**2)
        self.std = torch.sqrt(torch.clamp(variance, min=0) + eps)

    def transform(self, x):
        return (x - self.mean) / self.std


class RFFPostprocessor(BasePostprocessor):
    """
    Kernel Attention OOD Detection using Random Fourier Features.

    This method approximates a Gaussian kernel mean embedding for dataset-free
    inference. The OOD score is the "attention mass" = max_c(μ̂_c^T φ(x)), where
    μ̂_c is the per-class mean embedding and φ(x) is the RFF map.

    For universal kernels (e.g., Gaussian), the attention mass vanishes outside
    the in-distribution support, providing principled OOD guarantees.

    Reference:
        Random Fourier Features approximate the Gaussian kernel:
        k(x, y) = exp(-||x-y||² / 2σ²) ≈ φ(x)^T φ(y)
        where φ(x)_j = √(2/D) · cos(Ω_j^T x + b_j)
    """

    def __init__(self, config):
        super(RFFPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # Hyperparameters
        self.sigma = self.args.sigma  # Kernel bandwidth
        self.D = self.args.D  # RFF dimension
        self.alpha = self.args.alpha  # Target FPR for threshold
        self.K = self.args.K
        self.means = self.args.means
        self.B = self.args.B

        # Feature space: 'penultimate', 'all', or 'input'
        self.feature_space = getattr(self.args, "feature_space", "penultimate")

        # Whether to L2 normalize features (makes sigma transferable across datasets)
        self.normalize = getattr(self.args, "normalize", True)

        # Scoring mode: 'max' = max class score, 'margin' = max - 2nd max
        self.score_mode = getattr(self.args, "score_mode", "max")

        # Whether to normalize class scores by per-class variance
        self.variance_weighted = getattr(self.args, "variance_weighted", True)

        # Learned parameters (set during setup)
        self.omega = None  # [D, feature_dim] - RFF frequencies
        self.b = None  # [D] - RFF phases
        self.mu_hat = None  # [num_classes, D] - Per-class mean embeddings
        self.var_hat = None  # [num_classes] - Per-class score variance
        self.num_classes = None  # Number of classes
        self.threshold = None  # Scalar threshold (for current score_mode)
        self.max_threshold = None  # Max-based threshold (for diagnostics)
        self.feature_dim = None

        # Stored features and labels for hyperparameter search (avoid re-extraction)
        self.X_train = None  # Training features for mean embedding
        self.y_train = None  # Training labels
        self.X_val = None  # Validation features for threshold
        self.y_val = None  # Validation labels

        # Diagnostic mode: track class score distributions
        self.diagnose = getattr(self.args, "diagnose", False)
        self._diag_accum = {
            "n_above_threshold": [],
            "max_score": [],
            "margin": [],
            "mean_score": [],
        }

        self.setup_flag = False

    def _get_features(
        self, net: nn.Module, data: torch.Tensor, return_preds=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets certain features from neural network, allows for statistics before RFF embedding
        """
        if self.feature_space in ["minmax", "pcaall", "pcalayer", "UMAP"]:
            preds, all_features = net(
                data, return_feature_list=True
            )  # layers, Batch_size,  C, H, W
            penultimate_features = all_features[-1].flatten(start_dim=1)
            other_features = [f.flatten(start_dim=1) for f in all_features[:-1]]
            features = other_features, penultimate_features
        elif self.feature_space == "all":
            preds, all_features = net(data, return_feature_list=True)
            features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        elif self.feature_space == "input":
            features = torch.flatten(data, start_dim=1)
            preds = net(data)
        elif self.feature_space == "penultimate":
            preds, features = net(data, return_feature=True)

        if return_preds:
            return features, preds
        return features

    def _extract_features(self, net: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Extract features based on configured feature space.

        Args:
            net: Neural network model
            data: Input batch [batch_size, C, H, W]

        Returns:
            features: [batch_size, feature_dim]
        """
        preds = None
        if self.feature_space == "input":
            features, preds = self._get_features(net, data)
        elif self.feature_space == "all":
            # Concatenate flattened features from all layers
            features, preds = self._get_features(net, data)
        elif self.feature_space == "minmax":
            (other_features, penultimate_features), preds = self._get_features(
                net, data, True
            )

            flat_other = [
                self.norms[i].transform(f.flatten(start_dim=1))
                for i, f in enumerate(other_features)
            ]
            flat_max = torch.cat(
                [f.max(dim=1, keepdim=True)[0] for f in flat_other], dim=1
            )
            flat_min = torch.cat(
                [f.min(dim=1, keepdim=True)[0] for f in flat_other], dim=1
            )
            flat_minmax = torch.cat([flat_min, flat_max], dim=1)
            transformed_minmax = torch.tensor(
                power_transform(flat_minmax.cpu().numpy()),
                dtype=penultimate_features.dtype,
                device=penultimate_features.device,
            )

            features = torch.cat([penultimate_features, transformed_minmax], dim=1)
        elif self.feature_space == "pcalayer":
            (other_features, penultimate_features), preds = self._get_features(
                net, data, True
            )
            transformed = []
            for i, pca in enumerate(self.pcas):
                # Standardize
                other_features = [
                    self.norms[i].transform(f) for i, f in enumerate(other_features)
                ]
                # PCA Layerwise
                transforms = pca.transform(other_features[i])
                transforms = torch.tensor(transforms, dtype=penultimate_features.dtype)
                # Normalize
                transforms = torch.nn.functional.normalize(transforms, p=2, dim=1)
                transformed.append(transforms)
            # Normalize
            penultimate_features = torch.nn.functional.normalize(
                penultimate_features, p=2, dim=1
            )
            # Fuse
            features = torch.cat(transformed + [penultimate_features], dim=1)
        elif self.feature_space == "pcaall":
            (other_features, penultimate_features), preds = self._get_features(
                net, data, True
            )

            # Standardize
            other_features = [
                self.norms[i].transform(f) for i, f in enumerate(other_features)
            ]
            # PCA
            transformed = self.pcas.transform(torch.cat(other_features, dim=1))
            penultimate_features = self.pen_norm.transform(penultimate_features)

            # L2 Normalization
            penultimate_features = torch.nn.functional.normalize(
                penultimate_features, p=2, dim=1
            )
            transformed = torch.tensor(transformed, dtype=penultimate_features.dtype)
            transformed = torch.nn.functional.normalize(transformed, p=2, dim=1)

            # Fuse
            features = torch.cat([penultimate_features, transformed], dim=1)
        elif self.feature_space == "UMAP":
            (other_features, penultimate_features), preds = self._get_features(
                net, data, True
            )
            # Standardize
            other_features = [
                self.norms[i].transform(f) for i, f in enumerate(other_features)
            ]
            # UMAP
            feat_1 = self.param_umap.transform(
                torch.cat(other_features[:-3], dim=1),
            )
            feat_1 = torch.tensor(feat_1, dtype=penultimate_features.dtype)
            penultimate_features = self.pen_norm.transform(penultimate_features)
            # Fuse
            features = torch.cat([penultimate_features, feat_1], dim=1)
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            features, preds = self._get_features(net, data, True)
            features = self.pen_norm.transform(features)

        # L2 normalize features if enabled (makes sigma transferable across datasets)
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return preds, features

    def _sample_rff_params(self, feature_dim: int, device: torch.device):
        """Sample Random Fourier Feature parameters for Gaussian kernel."""
        self.feature_dim = feature_dim
        # Ω ~ N(0, σ^{-2} I) for Gaussian kernel
        self.omega = (torch.randn(self.D, feature_dim) / self.sigma).to(device)
        # b ~ Uniform[0, 2π]
        self.b = (torch.rand(self.D) * 2 * torch.pi).to(device)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RFF feature map.
        φ(x)_j = √(2/D) · cos(Ω_j^T x + b_j)

        Args:
            x: [batch_size, feature_dim] torch tensor

        Returns:
            [batch_size, D] torch tensor
        """
        device = x.device
        omega = self.omega.to(device)
        b = self.b.to(device)

        proj = x @ omega.T + b  # [batch_size, D]
        return torch.sqrt(torch.tensor(2.0 / self.D, device=device)) * torch.cos(proj)

    def _compute_rff_embedding(self, device: torch.device):
        """
        Recompute RFF parameters and per-class mean embeddings using stored features.
        Called when hyperparameters change during APS.
        """
        if self.X_train is None:
            return

        # Sample new RFF parameters with current sigma
        self._sample_rff_params(self.feature_dim, device)

        # Compute RFF features for all training samples
        phi_train = self._phi(self.X_train)  # [n_train, D]

        # Compute per-class mean embeddings
        if self.score_mode == "kmeans":
            self.class_centroids = []
            for c in range(self.num_classes):
                mask = self.y_train == c
                self.class_centroids.append(
                    torch.tensor(
                        KMeans(n_clusters=self.means)
                        .fit(phi_train[mask])
                        .cluster_centers_,
                        dtype=self.X_train.dtype,
                        device=device,
                    )
                )
        else:
            self.mu_hat = torch.zeros(self.num_classes, self.D, device=device)

            for c in range(self.num_classes):
                mask = self.y_train == c
                if mask.sum() > 0:
                    self.mu_hat[c] = phi_train[mask].mean(dim=0)

            # Compute per-class score variance for normalization
            self.var_hat = torch.ones(self.num_classes, device=device)
            if self.variance_weighted:
                for c in range(self.num_classes):
                    mask = self.y_train == c
                    if mask.sum() > 1:
                        scores_c = phi_train[mask] @ self.mu_hat[c]  # [n_c]
                        self.var_hat[c] = scores_c.var().clamp(min=1e-8)

            # Compute validation class scores
            phi_val = self._phi(self.X_val)  # [n_val, D]
            val_class_scores = phi_val @ self.mu_hat.T  # [n_val, num_classes]
            if self.variance_weighted:
                val_class_scores = val_class_scores / torch.sqrt(self.var_hat)

            # Always compute max-based threshold for diagnostics
            self.max_threshold = torch.quantile(
                val_class_scores.max(dim=1).values, self.alpha
            )

            # Compute scoring threshold based on score_mode
            if self.score_mode == "margin":
                sorted_val = val_class_scores.sort(dim=1, descending=True).values
                val_scores = sorted_val[:, 0] - sorted_val[:, 1]
            else:
                val_scores = val_class_scores.max(dim=1).values

            self.threshold = torch.quantile(val_scores, self.alpha)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """
        Setup phase: compute RFF parameters, per-class mean embeddings, and threshold.

        Uses training data for mean embedding and validation data for threshold.
        """
        if self.setup_flag:
            return

        print("\n" + "=" * 50)
        print("Setting up RFF Kernel Attention OOD detector...")
        print(f"  sigma (bandwidth): {self.sigma}")
        print(f"  D (RFF dimension): {self.D}")
        print(f"  alpha (target FPR): {self.alpha}")
        print(f"  feature_space: {self.feature_space}")
        print(f"  normalize: {self.normalize}")
        print(f"  score_mode: {self.score_mode}")
        print(f"  variance_weighted: {self.variance_weighted}")
        print("=" * 50)
        print("cuda" if torch.cuda.is_available() else "cpu")
        net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pen_norm = FeatureStandardizer(device)
        self.norms = None

        with torch.no_grad():
            for batch in tqdm(
                id_loader_dict["train"],
                desc="Computing Feature Standardization Statistics",
            ):
                data = batch["data"].to(device).float()
                labels = batch["label"].to(device)
                if self.feature_space in ["minmax", "pcaall", "pcalayer", "UMAP"]:
                    other_features, pen = self._get_features(
                        net, data
                    )  # num Layers, B, C * H * W
                else:
                    pen = self._get_features(net, data)  # num Layers, B, C * H * W
                if self.feature_space in ["minmax", "pcaall", "pcalayer", "UMAP"]:
                    if self.norms is None:
                        self.norms = [
                            FeatureStandardizer(device)
                            for f in range(len(other_features))
                        ]
                        for i, feats in enumerate(other_features):
                            self.norms[i].update(feats)
                self.pen_norm.update(pen)
        if self.feature_space in ["minmax", "pcaall", "pcalayer", "UMAP"]:
            for i, feats in enumerate(other_features):
                self.norms[i].finalize()
        self.pen_norm.finalize()

        if self.feature_space == "UMAP":
            self.other_feat_training = []
            for batch in tqdm(
                id_loader_dict["train"],
                desc="Computing UMAP Features",
            ):
                data = batch["data"].to(device).float()
                labels = batch["label"].to(device)

                other_features, pen = self._get_features(
                    net, data
                )  # num Layers, B, C * H * W

                self.other_feat_training.append(
                    torch.cat(other_features[:-3], dim=1).detach().cpu()
                )
            self.other_feat_training = (
                torch.cat(self.other_feat_training, dim=0).detach().cpu()
            )
            self.param_umap = UMAP(n_components=self.K, metric="cosine").fit(
                self.other_feat_training.numpy()
            )
            # P_sparse = fuzzy_simplicial_set(
            #     X=self.other_feat_training.numpy(),
            #     n_neighbors=self.K,
            #     verbose=True,
            # )
            # P_coo = P_sparse.tocoo()
            # edges_i = torch.as_tensor(P_coo.row, device="cpu").long()
            # edges_j = torch.as_tensor(P_coo.col, device="cpu").long()
            # weights = torch.as_tensor(P_coo.data, device="cpu")

            # dataset = TensorDataset(edges_i, edges_j, weights)
            # loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)
            # self.param_umap = ParametricUMAP(
            #     input_dim=self.other_feat_training.shape[1], z_dim=self.K
            # ).to(device)
            # optimizer = torch.optim.Adam(self.param_umap.parameters(), lr=1e-3)
            # for epoch in range(8):
            #     for i_idx, j_idx, w in tqdm(loader, desc="Training Parametric UMAP"):
            #         optimizer.zero_grad()
            #         # A. Embed the points
            #         z_i = self.param_umap(self.other_feat_training[i_idx].to(device))
            #         z_j = self.param_umap(self.other_feat_training[j_idx].to(device))

            #         # # B. Negative Sampling: pick random indices for 'k'
            #         # random_indices = torch.randint(
            #         #     0, len(training_), (len(i_idx),), device=device
            #         # )
            #         # z_k = self.param_umap(training_[random_indices])

            #         # B. Positive distances
            #         pos_dist = torch.sum((z_i - z_j) ** 2, dim=1)

            #         # C. Negative sampling (5 per anchor, single forward pass)
            #         neg_sample_rate = 5
            #         all_rand_idx = torch.randint(
            #             0,
            #             len(self.other_feat_training),
            #             (neg_sample_rate * len(i_idx),),
            #             device="cpu",
            #         )
            #         z_k_all = self.param_umap(
            #             self.other_feat_training[all_rand_idx].to(device)
            #         )  # (5N, dim)
            #         z_k_all = z_k_all.view(
            #             neg_sample_rate, len(i_idx), -1
            #         )  # (5, N, dim)
            #         z_i_exp = z_i.unsqueeze(0).expand(
            #             neg_sample_rate, -1, -1
            #         )  # (5, N, dim)
            #         neg_dist = torch.sum((z_i_exp - z_k_all) ** 2, dim=2).mean(
            #             0
            #         )  # (N,)

            #         # # C. Calculate Distances (Euclidean squared)
            #         # pos_dist = torch.sum((z_i - z_j) ** 2, dim=1)
            #         # neg_dist = torch.sum((z_i - z_k) ** 2, dim=1)

            #         # D. Compute Loss and Backprop
            #         loss = umap_loss(
            #             pos_dist,
            #             neg_dist,
            #             w.to(device),
            #             neg_sample_rate=neg_sample_rate,
            #         )
            #         loss.backward()
            #         optimizer.step()

        # Extract features only if not already done (for APS reuse)
        # PreRFF statistics need to compute before _extract_features
        if self.feature_space in ["pcaall", "pcalayer"]:
            self.pcas = None

            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["train"],
                    desc="Computing Incremental PCA Dim Reduction",
                ):
                    data = batch["data"].to(device).float()
                    labels = batch["label"].to(device)
                    other_features, _ = self._get_features(net, data)  # B, C * H * W

                    if self.pcas is None:
                        if self.feature_space == "pcaall":
                            self.pcas = IncrementalPCA(n_components=self.K, whiten=True)
                        else:
                            self.pcas = [
                                IncrementalPCA(n_components=self.K, whiten=True)
                                for i in range(len(other_features))
                            ]

                    if self.feature_space == "pcaall":
                        other_features = torch.cat(other_features, dim=1)
                        self.pcas.partial_fit(other_features)
                    else:
                        for i, feats in enumerate(other_features):
                            self.pcas[i].partial_fit(feats)

            if self.feature_space == "pcaall":
                print("Explained Variance: ", self.pcas.explained_variance_)
                print("Explained Variance Ratio:", self.pcas.explained_variance_ratio_)
            elif self.feature_space == "pcalayer":
                for i, pca in enumerate(self.pcas):
                    print("Explained Variance: ", pca.explained_variance_)
                    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
        if self.X_train is None:
            # Extract training features and labels for mean embedding
            train_features = []
            train_labels = []
            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["train"], desc="Extracting train features"
                ):
                    data = batch["data"].to(device).float()
                    labels = batch["label"].to(device)
                    _, features = self._extract_features(net, data)
                    train_features.append(features)
                    train_labels.append(labels)

            self.X_train = torch.cat(train_features, dim=0)
            self.y_train = torch.cat(train_labels, dim=0)
            self.feature_dim = self.X_train.shape[1]
            self.num_classes = int(self.y_train.max().item()) + 1
            print(
                f"Extracted {self.X_train.shape[0]} train features of dim {self.feature_dim}"
            )
            print(
                f"  Feature norm (mean): {torch.linalg.norm(self.X_train, dim=1).mean():.4f}"
            )
            print(f"  Number of classes: {self.num_classes}")

            # Extract validation features and labels for threshold calibration
            val_features = []
            val_labels = []
            with torch.no_grad():
                for batch in tqdm(
                    id_loader_dict["val"], desc="Extracting val features"
                ):
                    data = batch["data"].to(device).float()
                    labels = batch["label"].to(device)
                    _, features = self._extract_features(net, data)
                    val_features.append(features)
                    val_labels.append(labels)

            self.X_val = torch.cat(val_features, dim=0)
            self.y_val = torch.cat(val_labels, dim=0)
            print(f"Extracted {self.X_val.shape[0]} val features for threshold")

        # Compute RFF embedding and threshold
        self._compute_rff_embedding(device)

        # print(
        #     f"Per-class embedding norms (mean): {torch.linalg.norm(self.mu_hat, dim=1).mean():.4f}"
        # )
        # if self.variance_weighted:
        #     print(f"Per-class score variance (mean): {self.var_hat.mean():.6f}")
        # print(
        #     f"Threshold ({self.score_mode}) at {self.alpha * 100:.1f}%: {self.threshold:.4f}"
        # )
        # print(
        #     f"Max threshold (for diagnostics) at {self.alpha * 100:.1f}%: {self.max_threshold:.4f}"
        # )

        self.setup_flag = True
        print("RFF setup complete.\n")

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        Compute OOD score for a batch.

        Returns:
            pred: predicted class labels [batch_size]
            conf: attention mass scores [batch_size] (higher = more in-distribution)
        """

        output, features = self._extract_features(net, data)

        _, pred = torch.max(output, dim=1)

        # Compute RFF features
        phi_x = self._phi(features)  # [batch_size, D]

        # Compute class scores
        if self.score_mode == "kmeans":
            centroids = self.class_centroids
            class_scores = []
            # phi_train = self._phi(self.X_train)  # (n, D)

            # for beta in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
            #     class_stds = []
            #     for c_idx, C_c in enumerate(centroids):  # each (K_c, D)
            #         # get only training points belonging to this class
            #         mask = self.y_train == c_idx
            #         phi_c = phi_train[mask]  # (n_c, D)

            #         S_c = phi_c @ C_c.T  # (n_c, K_c)
            #         scores_c = torch.logsumexp(beta * S_c, dim=1) / beta
            #         class_stds.append(scores_c.std())

            #     mean_std = torch.stack(class_stds).mean()
            #     print(beta, mean_std, [f"{s.item():.4f}" for s in class_stds])
            for centroid in centroids:
                S = phi_x @ centroid.to(phi_x.device).T  #  [num_centroids, D]
                conf = (1 / self.B) * torch.logsumexp(self.B * S, dim=1)
                class_scores.append(conf)
            class_scores = torch.stack(class_scores, dim=1)  # (n, num_classes)
            conf = class_scores.max(dim=1).values  # (n,)
        else:
            mu_hat = self.mu_hat.to(features.device)  # [num_classes, D]
            class_scores = phi_x @ mu_hat.T  # [batch_size, num_classes]
            if self.variance_weighted:
                var_hat = self.var_hat.to(features.device)  # [num_classes]
                class_scores = class_scores / torch.sqrt(var_hat)

            # Compute confidence based on score mode
            if self.score_mode == "margin":
                sorted_scores = class_scores.sort(dim=1, descending=True).values
                conf = sorted_scores[:, 0] - sorted_scores[:, 1]  # max - 2nd max
            elif self.score_mode == "pred-aware":
                # conf = (torch.softmax(output, dim=1) * class_scores).sum(dim=1)
                conf = (torch.softmax(output, dim=1) * class_scores).max(dim=1).values
                # conf = class_scores[torch.arange(class_scores.size(0)), pred]
            else:
                conf = class_scores.max(dim=1).values

        # Accumulate diagnostics if enabled
        if self.diagnose and self.max_threshold is not None:
            sorted_scores = class_scores.sort(dim=1, descending=True).values
            # Use max-based threshold for class counting (meaningful regardless of score_mode)
            self._diag_accum["n_above_threshold"].append(
                (class_scores > self.max_threshold).sum(dim=1).cpu()
            )
            self._diag_accum["max_score"].append(sorted_scores[:, 0].cpu())
            self._diag_accum["margin"].append(
                (sorted_scores[:, 0] - sorted_scores[:, 1]).cpu()
            )
            self._diag_accum["mean_score"].append(class_scores.mean(dim=1).cpu())

        return pred, conf

    def save_diagnostics(self, save_path: str):
        """Save accumulated diagnostic statistics to .npz file and print summary."""
        diag = {}
        for key, val_list in self._diag_accum.items():
            if val_list:
                diag[key] = torch.cat(val_list).numpy()

        if not diag:
            return

        np.savez(save_path, **diag)
        self._print_diag_summary(diag, save_path)

    def _print_diag_summary(self, diag: dict, label: str = ""):
        """Print aggregate diagnostic summary."""
        n_above = diag["n_above_threshold"]
        margin = diag["margin"]
        max_score = diag["max_score"]
        n = len(n_above)

        print(f"\n--- RFF Diagnostics ({n} samples) {label} ---")
        print(
            f"  Classes above threshold: "
            f"mean={n_above.mean():.1f}, "
            f"zero={100 * (n_above == 0).mean():.1f}%, "
            f"one={100 * (n_above == 1).mean():.1f}%, "
            f"multi(>1)={100 * (n_above > 1).mean():.1f}%"
        )
        print(f"  Max score: mean={max_score.mean():.4f}, std={max_score.std():.4f}")
        print(f"  Margin (max-2nd): mean={margin.mean():.4f}, std={margin.std():.4f}")

    def reset_diagnostics(self):
        """Reset diagnostic accumulators for next dataset."""
        for key in self._diag_accum:
            self._diag_accum[key] = []

    def set_hyperparam(self, hyperparam: list):
        """
        Update hyperparameters for APS (Automatic Parameter Search) mode.
        Recomputes RFF parameters and mean embedding using stored features.
        """
        self.sigma = hyperparam[0]
        if len(hyperparam) > 1:
            self.D = int(hyperparam[1])
        if len(hyperparam) == 3:
            self.K = int(hyperparam[2])
        if len(hyperparam) == 4:
            self.B = int(hyperparam[3])

        # Recompute RFF embedding with new hyperparameters
        if self.X_train is not None:
            device = self.X_train.device
            self._compute_rff_embedding(device)

    def get_hyperparam(self):
        """Return current hyperparameters."""
        return [self.sigma, self.D]
