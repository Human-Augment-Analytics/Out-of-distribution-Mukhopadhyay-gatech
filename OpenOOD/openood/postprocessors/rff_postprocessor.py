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
        similarities = phi_x @ self.class_mean_embeddings.T  # [batch, num_classes]

        # Maximum attention: best class similarity
        max_similarity, _ = torch.max(similarities, dim=1)

        # Z-score normalize using calibration statistics
        conf = (max_similarity - self.score_mean) / (self.score_std + 1e-8)

        _, pred = torch.max(logits, dim=1)
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

    #Step 1: Collect all training features and labels
    all_features = []
    all_labels = []

    for batch in tqdm(train_loader, desc="[RFF] Extracting train features"):
        data = batch["data"].cuda()
        labels = batch["label"].cuda()

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

    #Step 2: Estimate kernel bandwidth if 'auto'
    if isinstance(kernel_bandwidth, str) and kernel_bandwidth == 'auto':
        subset_idx = torch.randperm(len(all_features))[:min(2000, len(all_features))]
        subset = all_features[subset_idx].cuda()
        dists = torch.cdist(subset, subset, p=2)
        kernel_bandwidth = dists[dists > 0].median().item()
        print(f"[RFF] Auto kernel_bandwidth (median heuristic) = {kernel_bandwidth:.4f}")
    
    print(f"[RFF] Using kernel_bandwidth = {kernel_bandwidth}")

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

    for batch in tqdm(val_loader, desc="[RFF] Computing val scores"):
        data = batch["data"].cuda()

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

        phi_x = torch.sqrt(torch.tensor(2.0 / feature_dim, device=device)) * \
                torch.cos(features @ random_omega.T + random_bias)
        phi_x = torch.nn.functional.normalize(phi_x, p=2, dim=1)

        similarities = phi_x @ class_mean_embeddings.T
        batch_max_sim, _ = torch.max(similarities, dim=1)
        val_scores.append(batch_max_sim.cpu())

    val_scores = torch.cat(val_scores)

    # Compute mean and std for z-score normalization
    score_mean = val_scores.mean().to(device)
    score_std = val_scores.std().to(device)

    print(f"[RFF] Val score stats: mean={score_mean.item():.4f}, std={score_std.item():.4f}, "
          f"min={val_scores.min().item():.4f}, max={val_scores.max().item():.4f}")

    threshold = torch.quantile(val_scores, q=target_rate)
    print(f"[RFF] Threshold at {target_rate} quantile = {threshold.item():.4f}")

    return (random_omega, random_bias, class_mean_embeddings, class_norms,
            global_mean, score_mean, score_std)