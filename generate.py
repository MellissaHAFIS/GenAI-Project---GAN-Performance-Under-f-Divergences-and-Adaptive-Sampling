import torch 
import torchvision
import os
import argparse
import torch.nn as nn

from model import Generator, Discriminator, FGanDiscriminator
from utils import load_model
from sampling_utils import sample_soft_truncation, sample_hard_truncation, dgflow_refine_z


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from GAN.')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for training.")
    parser.add_argument("--divergence", type=str, default="js", choices=["js", "kl", "pearson", "Baseline"],
                        help="Select a model: js = fgan avec Jensen-Shannon, kl = fgan avec Kullback-Leibler, pearson = fgan avec Pearson chi-squared, Baseline = Gan classique avec log D et log(1-D), Note: utilise la divergence de Jensen-Shannon de manière implicite via la BCELoss")
    parser.add_argument("--sampling", type=str, default="dgflow",
                        choices=["normal", "hard", "soft", "dgflow"],
                        help="Sampling strategy.")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Threshold for hard truncation.")
    parser.add_argument("--psi", type=float, default=0.7,
                        help="Psi value for soft truncation.")
    parser.add_argument("--dgflow_steps", type=int, default=25, help="Number of DGflow steps.")
    parser.add_argument("--dgflow_eta", type=float, default=0.1, help="Step size for DGflow.")
    
    args = parser.parse_args()


    # ------------------ DEVICE ------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    # ------------------ LOAD MODELS ------------------
    # On utilise args.divergence comme suffixe pour charger le bon fichier
    suffix = args.divergence
    
    print(f"Loading Generator (suffix: {suffix})...")
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    G = load_model(G, 'checkpoints', device, suffix=suffix, is_discriminator=False)
    G.eval()

    # Charger le Discriminateur UNIQUEMENT si on utilise DGflow
    D = None
    
    if args.sampling == "dgflow":
        print(f"Loading Discriminator for DGflow (suffix: {suffix})...")
        if args.divergence == "Baseline":
            
            D = Discriminator(mnist_dim, use_sigmoid=False).to(device)
        else:
            D = FGanDiscriminator(mnist_dim, use_sigmoid=False).to(device)
        
        D = load_model(D, 'checkpoints', device, suffix=suffix, is_discriminator=True)
        D.eval()
        # Geler les poids pour économiser de la mémoire et accélérer le calcul des gradients de Z
        for param in G.parameters(): param.requires_grad = False
        for param in D.parameters(): param.requires_grad = False

    # ------------------ OUTPUT FOLDER ------------------
    # On nomme le dossier avec la méthode ET la divergence pour ne pas tout mélanger
    save_folder = "samples" # f"folder_of_samples/samples_{args.sampling}_{suffix}"
    os.makedirs(save_folder, exist_ok=True)

    print(f"Generating with sampling = {args.sampling}")

    # ------------------ GENERATION ------------------
    n_samples = 0
    
    # On ne met pas de no_grad() ici car DGflow a besoin des gradients
    while n_samples < 10000:

        # ---- SAMPLING STRATEGY ----
        if args.sampling == "normal":
            z = torch.randn(args.batch_size, 100, device=device)
        elif args.sampling == "hard":
            z = sample_hard_truncation(args.batch_size, 100, threshold=args.threshold, device=device)
        elif args.sampling == "soft":
            z = sample_soft_truncation(args.batch_size, 100, psi=args.psi, device=device)
        elif args.sampling == "dgflow":
            if args.divergence == "Baseline":

                z = torch.randn(args.batch_size, 100, device=device)
                z = z.clone().detach()
                for t in range(args.dgflow_steps):
                    z.requires_grad_(True)
                    x_fake = G(z)
                    D_score = D(x_fake)
                    loss = -torch.mean(torch.nn.functional.softplus(D_score))
                    grad = torch.autograd.grad(loss, z)[0]
                    z = z + args.dgflow_eta * grad + 0.01 * torch.randn_like(z)
                    z = z.detach()
            
            else:
            
                if args.divergence == "pearson":
                    dgflow_eta = 0.001 # réduisez drastiquement votre `step_size` ($\eta$) à $0.001$ ou moins pour éviter que vos vecteurs latents ne s'envolent vers l'infini
                else:
                    # kl and JS
                    dgflow_eta = args.dgflow_eta # default value used in the paper

                z = torch.randn(args.batch_size, 100, device=device)
                z = dgflow_refine_z(
                    G, D, z,
                    n_steps=args.dgflow_steps,
                    step_size=dgflow_eta,
                    gamma=0.01, # keep default value used in the paper.
                    divergence=args.divergence.lower()
                )

        # ---- GENERATE & SAVE ----
        with torch.no_grad():
            x = G(z)
            x = x.view(-1, 1, 28, 28)

            for k in range(x.shape[0]):
                if n_samples >= 10000:
                    break
                torchvision.utils.save_image(
                    x[k:k + 1],
                    os.path.join(save_folder, f'{n_samples}.png'),
                    normalize=True
                )
                n_samples += 1

    print(f"Generation finished. Samples saved in {save_folder}")