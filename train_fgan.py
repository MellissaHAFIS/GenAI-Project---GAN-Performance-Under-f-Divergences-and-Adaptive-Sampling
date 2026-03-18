import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
from model import Generator, FGanDiscriminator
from fgan_utils import compute_fgan_loss_D, compute_fgan_loss_G

# ------------------ ARGPARSE ------------------
parser = argparse.ArgumentParser(description="Train f-GAN with different divergences")

parser.add_argument(
    "--divergence",
    type=str,
    default="js",
    choices=["js", "kl", "pearson"],
    help="Choose divergence: js | kl | pearson"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size (default: 64)"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of training epochs (default: 100)"
)

parser.add_argument(
    "--gpus",
    type=int,
    default=-1,
    help="Number of GPUs to use (-1 for all available)"
)

args = parser.parse_args()

# ------------------ CONFIG (runtime) ------------------
DIV_NAME = args.divergence.lower()
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
NUM_GPUS = args.gpus

LATENT_DIM = 100
MNIST_DIM = 784  # 28*28
SAVE_PATH = "checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    print(f"🚀 Training with divergence: {DIV_NAME.upper()}")

    # ------------------ DATA ------------------
    # No data augmentation - matches Baseline conditions, reduces training variance
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # ------------------ MODEL ------------------
    G = Generator(g_output_dim=MNIST_DIM).to(DEVICE)
    if DIV_NAME in ("kl", "pearson"):

        g_max_grad_norm = 0.5 
        d_max_grad_norm = 1.0
        D = FGanDiscriminator(d_input_dim=MNIST_DIM, use_sigmoid=False).to(DEVICE)  
        optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    if DIV_NAME == "js":

        # must be 1.0 and equal. 
        g_max_grad_norm = 1.0
        d_max_grad_norm = 1.0
        D = FGanDiscriminator(d_input_dim=MNIST_DIM, use_sigmoid=False).to(DEVICE)  # f-GAN needs logits
        # Optimizers — asymmetric LR: D learns slower to stay competitive with G
        optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # ------------------ TRAIN LOOP ------------------
    for epoch in range(EPOCHS):
        G.train()
        D.train()
        total_loss_D = 0
        total_loss_G = 0

        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.view(-1, MNIST_DIM).to(DEVICE)
            batch_size = real_images.size(0)

            # ------------------ Train Discriminator ------------------
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = G(z).detach()  # detach: no G gradients needed here

            v_real = D(real_images)
            v_fake = D(fake_images)

            loss_D = compute_fgan_loss_D(v_real, v_fake, DIV_NAME)
            if torch.isnan(loss_D) or torch.isinf(loss_D):
                print(f"⚠️ Warning: Loss_D is NaN or Inf at batch {batch_idx}")
                continue

            optimizer_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=d_max_grad_norm)
            optimizer_D.step()
            total_loss_D += loss_D.item()

            # ------------------ Train Generator ------------------
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_images = G(z)
            v_fake = D(fake_images)

            loss_G = compute_fgan_loss_G(v_fake, DIV_NAME)
            if torch.isnan(loss_G) or torch.isinf(loss_G):
                print(f"⚠️ Warning: Loss_G is NaN or Inf at batch {batch_idx}")
                continue

            optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=g_max_grad_norm)
            optimizer_G.step()
            total_loss_G += loss_G.item()

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")

        avg_loss_D = total_loss_D / len(train_loader)
        avg_loss_G = total_loss_G / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}")

        # ------------------ Save every 10 epochs (matches Baseline) ------------------
        # GAN loss does not reliably reflect image quality, so periodic saving is safer
        if (epoch + 1) % 10 == 0:
            os.makedirs(SAVE_PATH, exist_ok=True)
            torch.save(G.state_dict(), os.path.join(SAVE_PATH, f"G_{DIV_NAME}.pth"))
            torch.save(D.state_dict(), os.path.join(SAVE_PATH, f"D_{DIV_NAME}.pth"))
            print(f"✅ Models saved at epoch {epoch+1} for divergence '{DIV_NAME}'")

    print("🎯 Training finished!")