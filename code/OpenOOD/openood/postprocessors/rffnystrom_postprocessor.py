from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import power_transform
from .base_postprocessor import BasePostprocessor


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix between X and Y.
    K(x, y) = exp(-gamma * ||x - y||^2)

    Args:
        X: (n, d)
        Y: (m, d)
        gamma: bandwidth parameter (larger = narrower kernel)

    Returns:
        K: (n, m)
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X_sq = (X**2).sum(dim=1, keepdim=True)  # (n, 1)
    Y_sq = (Y**2).sum(dim=1, keepdim=True)  # (m, 1)
    sq_dists = X_sq + Y_sq.T - 2.0 * (X @ Y.T)  # (n, m)
    sq_dists = sq_dists.clamp(min=0.0)  # numerical safety
    return torch.exp(-gamma * sq_dists)


# ── Nyström approximation ─────────────────────────────────────────────────────


class NystromRBF(nn.Module):
    """
    Nyström RBF approximation with a scalar weight vector.

    Parameters
    ----------
    n_components : int
        Number of landmark points.
    gamma : float
        RBF bandwidth.
    landmark_method : 'random' | 'kmeans'
        How to pick landmark points during fit().
    reg : float
        Ridge penalty added to K_nm^T K_nm for numerical stability.
    """

    def __init__(
        self,
        n_components: int = 100,
        gamma: float = 1.0,
        landmark_method: str = "kmeans",
        reg: float = 1e-6,
    ):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.landmark_method = landmark_method
        self.reg = reg

        # set after fit()
        self.landmarks: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None  # (m,)
        self.sqrt_w: torch.Tensor | None = None

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, X: torch.Tensor) -> "NystromRBF":
        """
        Solve for weights that minimises the mean-kernel MSE objective.

        Objective:
            (1/n) ||K_nm w − k_means||²,   k_means = K_nn 1 / n

        Args:
            X: (n, d) training data
        """
        n = X.shape[0]
        landmarks = self._select_landmarks(X)  # (m, d)
        m = landmarks.shape[0]
        I = torch.eye(m, dtype=X.dtype, device=X.device)

        K_nm = rbf_kernel(X, landmarks, self.gamma)  # (n, m)
        K_nn = rbf_kernel(X, X, self.gamma)  # (n, n)

        # Mean kernel vector
        k_bar = K_nn.sum(dim=1) / n  # (n,)

        # Closed form solution
        weights = torch.linalg.solve(
            K_nm.T @ K_nm + self.reg * I, K_nm.T @ k_bar
        )  # (m,)

        # Clamp negatives — negative weights break the PSD feature map
        # (small negatives arise only from numerical noise)
        weights_pos = weights.clamp(min=0.0)

        self.landmarks = landmarks
        self.weights = weights
        self.sqrt_w = weights_pos.sqrt()

        # Residual: how well does the approximation match k̄?
        k_bar_hat = K_nm @ weights
        self._fit_mse = ((k_bar - k_bar_hat) ** 2).mean().item()
        print("MSE of the approximation is: ", self._fit_mse)
        return self

    def attention(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates attention mass

        Args:
            X: (n, d)

        Returns:
            m: (n, )
        """
        if self.landmarks is None:
            raise RuntimeError("Call fit() before attention().")

        K_nm = rbf_kernel(X, self.landmarks, self.gamma)  # (n, m)
        return (K_nm * self.weights).sum(dim=1)

    # ── landmark selection ────────────────────────────────────────────────────

    def _select_landmarks(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        m = min(self.n_components, n)

        if self.landmark_method == "random":
            idx = torch.randperm(n, device=X.device)[:m]
            return X[idx]

        elif self.landmark_method == "kmeans":
            return _kmeans_landmarks(X, m)

        else:
            raise ValueError(f"Unknown landmark_method: {self.landmark_method!r}")


# ── k-means++ for landmark init ───────────────────────────────────────────────


def _kmeans_landmarks(
    X: torch.Tensor, k: int, n_iter: int = 50, seed: int = 0
) -> torch.Tensor:
    """
    K-Means++ initialisation followed by Lloyd's iterations.
    """
    torch.manual_seed(seed)
    n = X.shape[0]

    # ── K-Means++ seeding ─────────────────────────────────────────────────────
    # 1. Pick the first center uniformly at random
    first = torch.randint(n, (1,), device=X.device).item()
    centers = [X[first]]

    for _ in range(1, k):
        # Stack current centers: (c, d)
        C = torch.stack(centers)
        # Squared distance from every point to its nearest center: (n,)
        dists = torch.cdist(X, C).min(dim=1).values ** 2
        # Sample next center proportional to distance²
        probs = dists / dists.sum()
        idx = torch.multinomial(probs, 1).item()
        centers.append(X[idx])

    centers = torch.stack(centers)  # (k, d)

    for _ in range(n_iter):
        # assignment
        dists = torch.cdist(X, centers)  # (n, k)
        labels = dists.argmin(dim=1)  # (n,)

        # update
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=X.device)
        new_centers.scatter_add_(0, labels.unsqueeze(1).expand_as(X), X)
        counts.scatter_add_(0, labels, torch.ones(X.shape[0], device=X.device))
        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(1)
        new_centers[~mask] = centers[~mask]  # keep old center if empty
        centers = new_centers

    return centers


class RFFNystromPostprocessor(BasePostprocessor):
    """
    Kernel Attention OOD Detection using Nystrom Method.

    For universal kernels (e.g., Gaussian), the attention mass vanishes outside
    the in-distribution support, providing principled OOD guarantees.

    """

    def __init__(self, config):
        super(RFFNystromPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # Hyperparameters
        self.sigma = self.args.sigma  # Kernel bandwidth
        self.m = self.args.m  # RFF dimension
        self.alpha = self.args.alpha  # Target FPR for threshold
        self.ridge_penalty = self.args.ridge

        self.kernel = getattr(self.args, "kernel", "rbf")

        self.anchor_method = getattr(self.args, "anchor_method", "kmeans")

        # Feature space: 'penultimate', 'all', or 'input'
        self.feature_space = getattr(self.args, "feature_space", "penultimate")

        # Whether to L2 normalize features (makes sigma transferable across datasets)
        self.normalize = getattr(self.args, "normalize", True)

        # Learned parameters (set during setup)
        self.omega = None  # [D, feature_dim] - RFF frequencies
        self.b = None  # [D] - RFF phases
        self.mu_hat = None  # [num_classes, D] - Per-class mean embeddings
        self.var_hat = None  # [num_classes] - Per-class score variance
        self.num_classes = None  # Number of classes
        self.threshold = None  # Scalar threshold (for current score_mode)
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

    def _extract_features(self, net: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Extract features based on configured feature space.

        Args:
            net: Neural network model
            data: Input batch [batch_size, C, H, W]

        Returns:
            features: [batch_size, feature_dim]
        """
        if self.feature_space == "input":
            features = torch.flatten(data, start_dim=1)
        elif self.feature_space == "all":
            # Concatenate flattened features from all layers
            _, all_features = net(data, return_feature_list=True)
            features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        elif self.feature_space == "minmax":
            _, all_features = net(
                data, return_feature_list=True
            )  # layers, Batch_size,  C, H, W
            penultimate_features = all_features[-1].flatten(start_dim=1)
            # other_features = all_features[
            #     :-1
            # ]  # Potentially use both only other or all when doing min-max norm

            flat_all = [f.flatten(start_dim=1) for f in all_features]
            flat_max = torch.cat(
                [f.max(dim=1, keepdim=True)[0] for f in flat_all], dim=1
            )
            flat_min = torch.cat(
                [f.min(dim=1, keepdim=True)[0] for f in flat_all], dim=1
            )
            flat_minmax = torch.cat([flat_min, flat_max], dim=1)
            transformed_minmax = torch.tensor(
                power_transform(flat_minmax.cpu().numpy()),
                dtype=penultimate_features.dtype,
                device=penultimate_features.device,
            )
            features = torch.cat([penultimate_features, transformed_minmax], dim=1)
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            _, features = net(data, return_feature=True)

        # L2 normalize features if enabled (makes sigma transferable across datasets)
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features

    def _compute_nystrom(self, device: torch.device):
        """
        Recompute Anchors and Weights when HyperParameters change
        """
        if self.X_train is None:
            return
        self.nystrom = NystromRBF(
            self.m, 1 / (2 * self.sigma), self.anchor_method, self.ridge_penalty
        ).fit(self.X_train)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """
        Setup phase: compute RFF parameters, per-class mean embeddings, and threshold.

        Uses training data for mean embedding and validation data for threshold.
        """
        if self.setup_flag:
            return

        print("\n" + "=" * 50)
        print("Setting up RFF Kernel Attention OOD detector...")
        print(f"  sigma (rbf bandwidth): {self.sigma}")
        print(f"  Number of Anchors): {self.m}")
        print(f"  alpha (target FPR): {self.alpha}")
        print(f"  feature_space: {self.feature_space}")
        print(f"  normalize: {self.normalize}")
        print("=" * 50)

        net.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract features only if not already done (for APS reuse)
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
                    features = self._extract_features(net, data)
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
                    features = self._extract_features(net, data)
                    val_features.append(features)
                    val_labels.append(labels)

            self.X_val = torch.cat(val_features, dim=0)
            self.y_val = torch.cat(val_labels, dim=0)
            print(f"Extracted {self.X_val.shape[0]} val features for threshold")

        # Compute Nystrom
        self._compute_nystrom(device)

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
        # Get predictions and features
        if self.feature_space == "input":
            output = net(data)
            features = torch.flatten(data, start_dim=1)
        elif self.feature_space == "all":
            output, all_features = net(data, return_feature_list=True)
            features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        elif self.feature_space == "minmax":
            output, all_features = net(
                data, return_feature_list=True
            )  # layers, Batch_size,  C, H, W
            penultimate_features = all_features[-1].flatten(start_dim=1)
            # other_features = all_features[
            #     :-1
            # ]  # Potentially use both only other or all when doing min-max norm

            flat_all = [f.flatten(start_dim=1) for f in all_features]
            flat_max = torch.cat(
                [f.max(dim=1, keepdim=True)[0] for f in flat_all], dim=1
            )
            flat_min = torch.cat(
                [f.min(dim=1, keepdim=True)[0] for f in flat_all], dim=1
            )
            flat_minmax = torch.cat([flat_min, flat_max], dim=1)
            transformed_minmax = torch.tensor(
                power_transform(flat_minmax.cpu().numpy()),
                dtype=penultimate_features.dtype,
                device=penultimate_features.device,
            )
            features = torch.cat([penultimate_features, transformed_minmax], dim=1)
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            output, features = net(data, return_feature=True)

        # L2 normalize features if enabled
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        _, pred = torch.max(output, dim=1)

        # Compute Approximated Kernel
        attention_mass = self.nystrom.attention(features)  # [batch_size, D]

        return pred, attention_mass

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
            self.m = int(hyperparam[1])

        # Recompute RFF embedding with new hyperparameters
        if self.X_train is not None:
            device = self.X_train.device
            self._compute_nystrom(device)

    def get_hyperparam(self):
        """Return current hyperparameters."""
        return [self.sigma, self.m]
