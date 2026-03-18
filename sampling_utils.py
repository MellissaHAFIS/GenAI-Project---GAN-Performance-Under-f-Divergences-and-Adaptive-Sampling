import torch
from scipy.stats import truncnorm

def sample_soft_truncation(batch_size, latent_dim, psi, device='cpu'):
    """
    Sample a latent vector from a Normal distribution and apply soft truncation.

    Soft truncation scales latent vectors toward the mean (0).

    Args:
    batch_size (int): Number of images to generate.
    latent_dim (int): Size of the latent vector (e.g., 100).
    psi (float): Truncation strength (1.0 = no truncation, <1 = stronger truncation).
    device (torch.device): CPU or CUDA/MPS.

    Returns:
    torch.Tensor: Latent vector z with dimension (batch_size, latent_dim).
    """
    print(f"Sampling with Soft Truncation (psi={psi})...")

    # Sample standard normal
    z = torch.randn(batch_size, latent_dim, device=device)

    # Apply soft truncation (shrink toward 0)
    z = psi * z

    return z

def sample_hard_truncation(batch_size, latent_dim, threshold, device='cpu'):
    """
    Sample a latent vector from a Truncated Normal distribution.
    Values outside the range [-threshold, threshold] are clipped and re-sampled.

    Args:
    batch_size (int): Number of images to generate.
    latent_dim (int): Size of the latent vector (e.g., 100 for this project).
    threshold (float): The truncation threshold (often denoted psi, e.g., 1.0 or 0.5).
    device (torch.device): CPU or CUDA/MPS.

    Returns:
    torch.Tensor: Latent vector z with dimension (batch_size, latent_dim).
    """
    print(f"Sampling with Hard Truncation (threshold={threshold})...")
    
    # scipy's truncnorm takes standardized bounds: (bound - mean) / standard deviation
    # Since we want a standard normal N(0,1), the bounds are just -threshold and +threshold
    truncated_z = truncnorm.rvs(-threshold, threshold, size=(batch_size, latent_dim))
    
    # Convert to PyTorch tensor and send to the correct device
    z_tensor = torch.tensor(truncated_z, dtype=torch.float32, device=device)
    
    return z_tensor

# ==========================================================================================================================================
#                                       GDflow utils : Raffinement de z en utilisant les gradients du Discriminateur
# ==========================================================================================================================================
def get_f_prime(ratio, divergence='kl'):
    """
    Calcule la dérivée f'(ratio) selon la f-divergence choisie.
    ratio = exp(-logit)
    """
    if divergence == 'kl':
        return torch.log(ratio) + 1.0
    elif divergence == 'js':
        return torch.log(2 * ratio / (ratio + 1.0))
    elif divergence == 'logd':
        return torch.log(ratio + 1.0) + 1.0
    elif divergence == 'pearson':
        # f(u) = (u-1)², f'(u) = 2(u-1)
        # In logit space: ratio = 0.5 * ratio + 1, so f'= ratio (the logit itself)
        return ratio  # equivalent to 2*(0.5*ratio + 1 - 1) = ratio
    else:
        raise ValueError(f"Divergence {divergence} non supportée par DGflow.")

def dgflow_refine_z(G, D, z0, n_steps=25, step_size=0.1, gamma=0.01, divergence='kl'):
    """
    Raffine un batch de vecteurs latents z0 en utilisant les gradients du Discriminateur.
    
    Args:
        G, D : Générateur et Discriminateur (déjà sur le bon device et en mode eval)
        z0 : Vecteur latent initial (Tensor)
        n_steps : Nombre d'itérations de raffinement
        step_size : Le pas d'apprentissage (eta)
        gamma : Facteur de diffusion (0 = déterministe)
    """
    # On détache z0 du graphe précédent et on demande à PyTorch de suivre ses gradients
    z = z0.clone().detach().requires_grad_(True)
    
    # Astuce Pro : Utiliser SGD pour mettre à jour 'z' proprement
    optimizer = torch.optim.SGD([z], lr=step_size)
    
    for i in range(n_steps):
        optimizer.zero_grad()
        
        # 1. Générer l'image
        x = G(z)
        
        # 2. Obtenir le logit du discriminateur
        logits = D(x)
        
        # 3. Calculer le ratio e^{-d(x)}
        ratio = torch.exp(-logits)
        
        # 4. Appliquer f'
        f_prime = get_f_prime(ratio, divergence)
        
        # 5. La perte à MINIMISER est la somme des f' 
        # (L'optimiseur fera z = z - step_size * gradient)
        loss = torch.sum(f_prime)
        loss.backward()
        
        # 6. Mise à jour de z selon le gradient
        optimizer.step()
        
        # 7. Ajout du terme de diffusion (bruit de Langevin)
        if gamma > 0:
            with torch.no_grad():
                noise = torch.randn_like(z) * torch.sqrt(torch.tensor(2 * step_size * gamma, device=z.device))
                z.add_(noise)
                
    return z.detach()