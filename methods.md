# OOD (Out-of-Distribution) Detection Methods

Survey paper from 2025 : https://dl.acm.org/doi/epdf/10.1145/3760390


##  Supervised Methods

###  Requires Dataset During Inference

| Method | Paper | CIFAR-10 (Near / Far) | CIFAR-100 (Near / Far) | ImageNet | Description |
|-------|------|----------------------|------------------------|----------|------------|
|  |  |  |  |  |  |
|  |  |  |  |  |  |

---

###  No Dataset Needed During Inference

| Method | Paper | CIFAR-10 (Near / Far) | CIFAR-100 (Near / Far) | ImageNet | Description |
|--------|-------|----------------------|------------------------|----------|------------|
| **Outlier Exposure** | [OpenReview](https://openreview.net/pdf?id=HyxCxhRcY7) |  | **88.30 / 81.41** |  | Uses ImageNet-22k and 80M Tiny Images as outliers. MSP used as baseline preprocessor |
|  |  |  |  |  |  |

---

##  Unsupervised Methods

###  Requires Dataset During Inference

| Method | Paper | CIFAR-10 (Near / Far) | CIFAR-100 (Near / Far) | ImageNet | Description |
|--------|-------|----------------------|------------------------|----------|------------|
| **Reweight OOD** | [CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Regmi_ReweightOOD_Loss_Reweighting_for_Distance-based_OOD_Detection_CVPRW_2024_paper.pdf) |  | **71.27 / 91.12** |  | Loss reweighting to tighten embedding clusters within class. Enables KNN / Mahalanobis scoring |
| **Deep Nearest Neighbors** | [arXiv](https://arxiv.org/pdf/2204.06507) | Not in leaderboard |  |  | Requires K samples from dataset to estimate distance thresholds |

---

###  No Dataset Needed / Partial Data Needed

| Method | Paper | CIFAR-10 (Near / Far) | CIFAR-100 (Near / Far) | ImageNet | Description |
|--------|-------|----------------------|------------------------|----------|------------|
| **Rotation Prediction** | [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/file/a2b15837edac15df90721968986f7f8e-Paper.pdf) | **94.86 / 98.18** |  |  | Uses rotation prediction as auxiliary task. Combines KL divergence + BCE loss |
| **PixMix** | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Hendrycks_PixMix_Dreamlike_Pictures_Comprehensively_Improve_Safety_Measures_CVPR_2022_paper.pdf) | Same as above |  |  | Uses fractal images as structured noise augmentation |
| **Kernel PCA (RFF)** | [arXiv](https://arxiv.org/pdf/2402.02949) | Not in leaderboard |  |  | Applies PCA on nonlinear random Fourier features. Reconstruction error used as OOD score |
| **Gram Matrices** | [arXiv](https://arxiv.org/pdf/1912.12510) | Not in leaderboard |  |  | Uses CNN activation statistics (layer/channel variance). Threshold-based detection |
| **Diffusion-based OOD** | [CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf) | Not in leaderboard |  |  | Uses reconstruction error from diffusion steps with MSE / Mahalanobis distance |

---
