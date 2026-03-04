from typing import Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class RFFPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RFFPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.kernel_bandwidth = self.args.kernel_bandwidth
        self.feature_dim = self.args.feature_dim
        self.target_rate = self.args.target_rate

        self.feature_space = self.args.feature_space
        self.setup_flag = False
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
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

            print(f"[RFF Postprocessor] num_classes = {num_classes}")

            (
                self.random_omega,
                self.random_bias,
                self.class_mean_embeddings,
                self.class_norms,
                self.global_mean_embedding,
                self.score_mean,
                self.score_std,
                self.alpha,
            ) = calibrate_model(
                net,
                id_loader_dict["train"],
                id_loader_dict["val"],
                self.target_rate,
                self.kernel_bandwidth,
                self.feature_dim,
                self.feature_space,
                num_classes=num_classes,
            )
            self.setup_flag = True

    def _extract_features(self, model, data):
        if self.feature_space == "input":
            features = torch.flatten(data, start_dim=1)
            logits = model(data)
        else:
            logits, all_features = model(data, return_feature_list=True)
            if self.feature_space == "all":
                features = torch.cat(
                    [f.flatten(start_dim=1) for f in all_features], dim=1
                )
            else:
                features = all_features[-1].view(all_features[-1].size(0), -1)
        return logits, features

    def _compute_rff(self, features):
        phi = torch.sqrt(torch.tensor(2.0 / self.feature_dim, device=features.device)) * \
              torch.cos(features @ self.random_omega.T + self.random_bias)
        phi = torch.nn.functional.normalize(phi, p=2, dim=1)
        return phi

    @torch.no_grad()
    def postprocess(self, model: nn.Module, data: Any):
        logits, features = self._extract_features(model, data)

        # L2 normalize raw features
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        phi_x = self._compute_rff(features)

        # Compute cosine similarity to each class mean in RFF space
        # class_mean_embeddings are already L2-normalized
        similarities = self.alpha * (phi_x @ self.class_mean_embeddings.T)  # [batch, num_classes]

        # --- Predictor-Aware Component ---
        # Instead of max similarity, use the similarity of the class predicted by the logits
        probs = torch.softmax(logits, dim=1)
        _, pred = torch.max(logits, dim=1)
        
        batch_size = logits.size(0)
        pred_similarities = similarities[torch.arange(batch_size), pred]

        # Entropy Component
        # Calculate Shannon entropy (higher entropy = more uncertain/OOD)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # Combine: High similarity is ID, High entropy is OOD
        raw_conf = pred_similarities - entropy

        # Z-score normalize using calibration statistics
        conf = (raw_conf - self.score_mean) / (self.score_std + 1e-8)

        return pred, conf
    
@torch.no_grad()
def calibrate_model(
    model,
    train_loader,
    val_loader,
    target_rate,
    kernel_bandwidth,
    feature_dim,
    feature_space,
    num_classes,
):
    model.eval()
    device = next(model.parameters()).device

    # Step 1: Collect all training features and labels
    all_features = []
    all_labels = []
    for batch in tqdm(train_loader, desc="[RFF] Extracting train features"):
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        if feature_space == "input":
            features = torch.flatten(data, start_dim=1)
        else:
            _, feat_list = model(data, return_feature_list=True)
            if feature_space == "all":
                features = torch.cat(
                    [f.flatten(start_dim=1) for f in feat_list], dim=1
                )
            else:
                features = feat_list[-1].view(feat_list[-1].size(0), -1)

        features = torch.nn.functional.normalize(features, p=2, dim=1)
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    data_dim = all_features.shape[1]

    # Extract validation features and labels for MLE tuning
    val_features = []
    val_labels = []
    val_logits = []
    for batch in tqdm(val_loader, desc="[RFF] Extracting val features"):
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        if feature_space == "input":
            features = torch.flatten(data, start_dim=1)
            logits = model(data)
        else:
            logits, feat_list = model(data, return_feature_list=True)
            if feature_space == "all":
                features = torch.cat(
                    [f.flatten(start_dim=1) for f in feat_list], dim=1
                )
            else:
                features = feat_list[-1].view(feat_list[-1].size(0), -1)

        features = torch.nn.functional.normalize(features, p=2, dim=1)
        val_features.append(features.cpu())
        val_labels.append(labels.cpu())
        val_logits.append(logits.cpu())

    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    val_logits = torch.cat(val_logits, dim=0)

    chunk_size = 1024
    alpha = 1.0

    #Step 2: Estimate kernel bandwidth and alpha using MLE Grid Search
    if isinstance(kernel_bandwidth, str) and kernel_bandwidth == 'auto':
        subset_idx = torch.randperm(len(all_features))[:min(2000, len(all_features))]
        subset = all_features[subset_idx].cuda()
        dists = torch.cdist(subset, subset, p=2)
        med_dist = dists[dists > 0].median().item()
        
        print(f"[RFF] Base median distance = {med_dist:.4f}. Starting MLE Grid Search...")
        
        # Grid for sigma (bandwidth) and alpha (temperature scaling)
        sigma_grid = [0.25 * med_dist, 0.5 * med_dist, 1.0 * med_dist, 2.0 * med_dist, 4.0 * med_dist]
        alpha_grid = [1.0, 5.0, 10.0, 20.0, 50.0]
        
        best_nll = float('inf')
        best_sigma = med_dist
        best_alpha = 1.0
        
        for sig in sigma_grid:
            omega = (torch.randn(feature_dim, data_dim) / sig).to(device)
            bias = (torch.rand(feature_dim) * 2 * np.pi).to(device)
            
            # Train class means
            train_phi = []
            for i in range(0, len(all_features), chunk_size):
                chunk = all_features[i:i+chunk_size].to(device)
                phi = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * \
                      torch.cos(chunk @ omega.T + bias)
                phi = torch.nn.functional.normalize(phi, p=2, dim=1)
                train_phi.append(phi)
            train_phi = torch.cat(train_phi, dim=0)
            
            c_means = torch.zeros(num_classes, feature_dim, device=device)
            for c in range(num_classes):
                mask = (all_labels == c)
                if mask.any():
                    c_mean = torch.nn.functional.normalize(train_phi[mask].mean(dim=0), dim=0)
                    c_means[c] = c_mean
                    
            # Val similarities
            val_phi = []
            for i in range(0, len(val_features), chunk_size):
                chunk = val_features[i:i+chunk_size].to(device)
                phi = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * \
                      torch.cos(chunk @ omega.T + bias)
                phi = torch.nn.functional.normalize(phi, p=2, dim=1)
                val_phi.append(phi)
            val_phi = torch.cat(val_phi, dim=0)
            
            sims = val_phi @ c_means.T
            
            for alp in alpha_grid:
                # MLE: Minimize Negative Log-Likelihood (Cross Entropy)
                nll = torch.nn.functional.cross_entropy(sims * alp, val_labels.to(device)).item()
                if nll < best_nll:
                    best_nll = nll
                    best_sigma = sig
                    best_alpha = alp
                    
        kernel_bandwidth = best_sigma
        alpha = best_alpha
        print(f"[RFF] MLE Grid Search found best sigma={kernel_bandwidth:.4f}, alpha={alpha:.4f} (NLL={best_nll:.4f})")
    else:
        print(f"[RFF] Using provided kernel_bandwidth = {kernel_bandwidth}")


    #Step 3: Generate RFF random projection
    random_omega = (torch.randn(feature_dim, data_dim) / kernel_bandwidth).to(device)
    random_bias = (torch.rand(feature_dim) * 2 * np.pi).to(device)

    #Step 4: Compute RFF embeddings for all training data
    chunk_size = 1024
    all_phi = []
    for i in range(0, len(all_features), chunk_size):
        chunk = all_features[i:i+chunk_size].to(device)
        phi = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * \
              torch.cos(chunk @ random_omega.T + random_bias)
        phi = torch.nn.functional.normalize(phi, p=2, dim=1)
        all_phi.append(phi.cpu())

    all_phi = torch.cat(all_phi, dim=0)

    class_mean_embeddings = torch.zeros(num_classes, feature_dim, device=device)
    class_norms = torch.zeros(num_classes, device=device)

    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.any():
            class_phi = all_phi[mask].to(device)
            class_mean = class_phi.mean(dim=0)
            class_mean = torch.nn.functional.normalize(class_mean, dim=0)
            class_mean_embeddings[c] = class_mean
            class_norms[c] = mask.sum().float()

    global_mean = all_phi.to(device).mean(dim=0)
    global_mean = torch.nn.functional.normalize(global_mean, dim=0)

    # Compute calibration scores on validation set
    val_scores = []
    for i in range(0, len(val_features), chunk_size):
        chunk = val_features[i:i+chunk_size].to(device)
        chunk_logits = val_logits[i:i+chunk_size].to(device)
        
        phi_x = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * \
                torch.cos(chunk @ random_omega.T + random_bias)
        phi_x = torch.nn.functional.normalize(phi_x, p=2, dim=1)

        similarities = alpha * (phi_x @ class_mean_embeddings.T)
        
        # Predictor-Aware Component
        probs = torch.softmax(chunk_logits, dim=1)
        _, preds = torch.max(chunk_logits, dim=1)
        pred_sims = similarities[torch.arange(chunk.size(0)), preds]
        
        # Entropy Component
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Combined Score
        combined_score = pred_sims - entropy
        val_scores.append(combined_score.cpu())

    val_scores = torch.cat(val_scores)

    # Compute mean and std for z-score normalization
    score_mean = val_scores.mean().to(device)
    score_std = val_scores.std().to(device)

    print(f"[RFF] Val score stats: mean={score_mean.item():.4f}, std={score_std.item():.4f}, "
          f"min={val_scores.min().item():.4f}, max={val_scores.max().item():.4f}")

    threshold = torch.quantile(val_scores, q=target_rate)
    print(f"[RFF] Threshold at {target_rate} quantile = {threshold.item():.4f}")

    return (random_omega, random_bias, class_mean_embeddings, class_norms,
            global_mean, score_mean, score_std, alpha)