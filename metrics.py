import torch
import numpy as np
from scipy import linalg

def compute_pairwise_distance(data_x, data_y=None):
    """
    Compute pairwise Euclidean distance.
    If data_y is None, computes the distance between data_x and itself.
    Args:
        data_x: Tensor (N, feature_dim)
        data_y: Tensor (M, feature_dim) or None
    Returns:
        Tensor (N, M) of distances
    """
    if data_y is None:
        data_y = data_x
    
    # torch.cdist is optimized for computing Euclidean distance (p=2)
    dists = torch.cdist(data_x, data_y, p=2)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Retrieve the k-th smallest value along a given axis.
    Args:
        unsorted: Tensor of distances
        k: index of the k-th neighbor (e.g., k=3)
    """
    # topk with largest=False returns the smallest elements
    # We take k+1 because the distance to itself is 0 (included in the list)
    args = torch.topk(unsorted, k=k+1, dim=axis, largest=False)
    return args.values[:, -1]


def compute_precision_recall_manifold(real_features, gen_features, k=3):
    """
    Exact implementation of "Improved Precision and Recall Metric for Assessing Generative Models".
    
    Args:
        real_features (Tensor): Features of real images [N, Dim]
        gen_features (Tensor): Features of generated images [M, Dim]
        k (int): Number of neighbors to estimate the manifold (paper default: 3)
    
    Returns:
        (precision, recall): Tuple of floats
    """
    print(f"Computing P&R (Manifold) with k={k}...")

    # Ensure tensors are on the same device (CPU is often sufficient for <50k samples; otherwise GPU)
    device = real_features.device
    
    # --- 1. Estimation of the REAL data manifold (for Precision) ---
    # Compute distances between all real samples
    # Note: If N > 20,000, this can consume a lot of RAM.
    # For MNIST (10k test), this is manageable (10k*10k*4 bytes ~ 400MB).
    d_real_real = compute_pairwise_distance(real_features)
    
    # Manifold radius around each real point = distance to the k-th nearest neighbor
    # The 0-th neighbor (itself, dist=0) is ignored
    real_radii = get_kth_value(d_real_real, k=k)
    
    # --- 2. Precision computation ---
    # For each generated image, is it inside the real manifold?
    d_gen_real = compute_pairwise_distance(gen_features, real_features)
    
    # Test: Does there exist at least ONE real point 'r' such that Dist(g, r) <= Radius(r)?
    # real_radii is (N,), broadcast to compare with d_gen_real (M, N)
    is_in_real_manifold = (d_gen_real <= real_radii.unsqueeze(0)).any(dim=1)
    precision = is_in_real_manifold.float().mean().item()

    # --- 3. Estimation of the GENERATED data manifold (for Recall) ---
    d_gen_gen = compute_pairwise_distance(gen_features)
    gen_radii = get_kth_value(d_gen_gen, k=k)
    
    # --- 4. Recall computation ---
    # For each real image, is it inside the generated manifold?
    # Note: We can reuse the transposed d_gen_real to avoid recomputation
    d_real_gen = d_gen_real.t()  # (N, M)
    
    is_in_gen_manifold = (d_real_gen <= gen_radii.unsqueeze(0)).any(dim=1)
    recall = is_in_gen_manifold.float().mean().item()
    
    return precision, recall

def compute_fid(real_features, gen_features, eps=1e-6):
    """
    Compute FID (Fréchet Inception Distance) between real and generated features.
    Args:
        real_features: Tensor (N, D) - features of real images
        gen_features: Tensor (M, D) - features of generated images
        eps: small value to stabilize sqrt
    Returns:
        fid_value: float
    """
    # Convert to numpy for linalg operations
    mu_real = real_features.mean(dim=0).cpu().numpy()
    sigma_real = np.cov(real_features.cpu().numpy(), rowvar=False)

    mu_gen = gen_features.mean(dim=0).cpu().numpy()
    sigma_gen = np.cov(gen_features.cpu().numpy(), rowvar=False)

    # Numerical stability
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_gen += np.eye(sigma_gen.shape[0]) * eps

    # Squared difference of means
    diff = mu_real - mu_gen
    diff_squared = diff.dot(diff)

    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_value = diff_squared + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid_value)