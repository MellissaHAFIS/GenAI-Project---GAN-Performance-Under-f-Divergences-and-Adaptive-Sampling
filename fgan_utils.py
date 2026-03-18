"""
The file fgan_utils.py (The Architect of Mathematics).
Contains formulas for KL, Pearson χ², and a numerically stable JS.
Reference: Table 6 of the f-GAN paper (https://arxiv.org/pdf/1606.00709.pdf)
"""
import torch
import torch.nn.functional as F
import math

def get_activation(divergence_name):
    """
    Returns the activation function g_f(v) specific to the divergence.
    According to Table 6 of the f-GAN paper.
    """
    divergence_name = divergence_name.lower()

    if divergence_name in ['kl', 'pearson']:
        # Identity
        return lambda v: v

    elif divergence_name == 'js':
        # Stable JS activation:
        # g_f(v) = log(2) - softplus(-v)
        return lambda v: math.log(2.0) - F.softplus(-v)

    else:
        raise ValueError(f"Divergence '{divergence_name}' not supported.")


def get_conjugate(divergence_name):
    """
    Returns the Fenchel conjugate function f*(t) specific to the divergence.
    According to Table 6 of the f-GAN paper.
    """
    divergence_name = divergence_name.lower()

    if divergence_name == 'kl':
        def fused_kl(v):
            v_safe = torch.clamp(v, min=-10.0, max=20.0)
            return torch.exp(v_safe - 1.0)
        return fused_kl

    elif divergence_name == 'pearson':
        return lambda t: 0.25 * (t ** 2) + t

    elif divergence_name == 'js':

        def f_star_js(t):
            # Ensure numerical stability
            eps = 1e-6
            t = torch.clamp(t, max=math.log(2.0) - eps)
            return -torch.log(2.0 - torch.exp(t))

        return f_star_js

    else:
        raise ValueError(f"Divergence '{divergence_name}' not supported.")


def compute_fgan_loss_D(v_real, v_fake, divergence_name):
    """
    Computes the Discriminator loss for f-GAN.
    """
    divergence_name = divergence_name.lower()

    if divergence_name == 'js':
        # Fully stable JS form
        loss_d = torch.mean(F.softplus(-v_real)) + torch.mean(F.softplus(v_fake))
        return loss_d
    
    elif divergence_name == 'kl':
        # Stable: no clamping needed in D path since gradients flow through mean
        # Use log-sum-exp trick for the exp term
        exp_term = torch.exp(torch.clamp(v_fake - 1.0, max=10.0))
        loss_d = -(torch.mean(v_real) - torch.mean(exp_term))
        return loss_d  # ✅
    
    elif divergence_name == 'pearson':
        g_f = get_activation(divergence_name)
        f_star = get_conjugate(divergence_name)

        t_real = g_f(v_real)
        t_fake = g_f(v_fake)

        loss_d = -(torch.mean(t_real) - torch.mean(f_star(t_fake)))
        return loss_d


def compute_fgan_loss_G(v_fake, divergence_name):
    """
    Computes the Generator (G) loss for the f-GAN.
    Args:
    v_fake (Tensor): Raw output (logits) of the discriminator on fake images.
    divergence_name (str): 'kl' or 'pearson'.
    """
    divergence_name = divergence_name.lower()

    if divergence_name == 'js':
        # Stable JS generator loss
        loss_g = torch.mean(F.softplus(-v_fake))
        return loss_g
    elif divergence_name == 'kl':
        # Paper Eq. 10 alternative update: maximize E[g_f(T(fake))]
        # For KL, g_f(v) = v (identity), so: minimize -E[v_fake]
        # This avoids exp() in the generator path entirely → no explosion
        return -torch.mean(v_fake)
    elif divergence_name == 'pearson':
        # For Pearson: use the STANDARD f-GAN generator loss
        # Minimize -E[f*(T(x))]
        g_f = get_activation(divergence_name)
        f_star = get_conjugate(divergence_name)
        t_fake = g_f(v_fake)
        loss_g = -torch.mean(f_star(t_fake))
        
        return loss_g

# Remark about JS:

# 1. **Stabilité Numérique (JS)** : La fonction $f^*(t)$ pour JS contient un `log(2 - exp(t))`. 
# Si $t$ s'approche trop de $\log(2)$, cela peut causer des `NaN`.
#  Cependant, l'activation $g_f(v)$ pour JS est conçue pour que sa sortie soit toujours inférieure à $\log(2)$, ce qui protège 
# mathématiquement le calcul.

# 2. **Comparaison avec la Baseline** : En utilisant `js` dans ton `train_fgan.py`, tu devrais obtenir des résultats très proches de ton 
# `train.py` original (qui utilise `BCELoss`), car le vanilla GAN  minimise précisément la divergence de Jensen-Shannon.
#  C'est un excellent moyen de vérifier que votre implémentation du f-GAN est correcte avant de passer à Pearson ou KL.

# j'ai ajoute JS meme si c'est un type de divergence qui fonctionne tres bien mais apres avec gradient flow ca se stabilise mieux 
# donc il sera interessant de le tester et de voir les amelioration grace au Gradient flow
# , par contre pearson avec gradient flow ne vont pas tres bien ensemble (y a une explication derniere) 
# donc il sera aussi interessant de le tester.