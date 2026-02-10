from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class RFFKernelPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RFFKernelPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.kernel_bandwidth = self.args.kernel_bandwidth
        self.feature_dim = self.args.feature_dim
        self.target_rate = self.args.target_rate

        self.feature_space = self.args.feature_space
        self.setup_flag = False
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            (
                self.random_means,
                self.random_intersects,
                self.threshold,
                self.mean_embedding,
            ) = calibrate_model(
                net,
                id_loader_dict["train"],
                id_loader_dict["val"],
                self.target_rate,
                self.kernel_bandwidth,
                self.feature_dim,
                self.feature_space,
            )
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, model: nn.Module, data: Any):
        if self.feature_space == "input":
            # Flattened Input
            features = torch.flatten(data, start_dim=1)
        else:
            logits, all_features = model(data, return_feature_list=True)
            if self.feature_space == "all":
                # Flattened ALL Features
                features = torch.cat(
                    [f.flatten(start_dim=1) for f in all_features], dim=1
                )
            else:
                # Flattened Penultimate Features
                features = all_features[-1].view(all_features[-1].size(0), -1)

        phi_x = torch.sqrt(torch.tensor(2.0 / self.feature_dim)) * torch.cos(
            features @ self.random_means.T + self.random_intersects
        )

        attention_mass = phi_x @ self.mean_embedding
        _, pred = torch.max(logits, dim=1)
        conf = attention_mass
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
):
    model.eval()  # Placeholder if we want to use model features
    threshold = mean_embedding = size = 0
    random_init = False

    for batch in tqdm(train_loader, desc="Compute mean embedding"):
        data = batch["data"].cuda()

        if feature_space == "input":
            # Flattened Input
            features = torch.flatten(data, start_dim=1)
        else:
            # Flattened ALL Features
            _, all_features = model(data, return_feature_list=True)
            if feature_space == "all":
                features = torch.cat(
                    [f.flatten(start_dim=1) for f in all_features], dim=1
                )
            else:
                features = all_features[-1].view(all_features[-1].size(0), -1)

        batch_size, data_dim = features.shape
        if not random_init:
            random_means = (
                torch.randn(feature_dim, data_dim) * (1 / kernel_bandwidth)
            ).to(data.device)
            random_intersects = (torch.rand(feature_dim) * 2 * torch.pi).to(data.device)
            random_init = True

        phi_x = torch.sqrt(torch.tensor(2.0 / feature_dim)) * torch.cos(
            features @ random_means.T + random_intersects
        )

        mean_embedding += phi_x.sum(
            dim=0
        )  # First sum along batch dimension then add to mean embedding
        size += batch_size

    mean_embedding = mean_embedding / size
    scores = []

    for batch in tqdm(val_loader, desc="Compute threshold"):
        data = batch["data"].cuda()

        if feature_space == "input":
            # Flattened Input
            features = torch.flatten(data, start_dim=1)
        else:
            _, all_features = model(data, return_feature_list=True)
            if feature_space == "all":
                # Flattened ALL Features
                features = torch.cat(
                    [f.flatten(start_dim=1) for f in all_features], dim=1
                )
            else:
                # Flattened Penultimate Features
                features = all_features[-1].view(all_features[-1].size(0), -1)

        phi_x = torch.sqrt(torch.tensor(2.0 / feature_dim)) * torch.cos(
            features @ random_means.T + random_intersects
        )

        batch_scores = phi_x @ mean_embedding
        scores.append(batch_scores.cpu())

    scores = torch.cat(scores)
    threshold = torch.quantile(scores, q=target_rate)
    return random_means, random_intersects, threshold, mean_embedding
