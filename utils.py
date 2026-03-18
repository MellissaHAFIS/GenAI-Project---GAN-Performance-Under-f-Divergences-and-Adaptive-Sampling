import torch
import os

def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder, suffix=""):
    """
    Sauvegarde G et D avec un suffixe pour éviter d'écraser les autres expériences.
    Exemple : G_pearson.pth, D_pearson.pth
    """
    os.makedirs(folder, exist_ok=True)
    s = f"_{suffix}" if suffix else ""
    torch.save(G.state_dict(), os.path.join(folder, f'G{s}.pth'))
    torch.save(D.state_dict(), os.path.join(folder, f'D{s}.pth'))
    print(f"Modèles sauvegardés dans {folder} avec le suffixe '{suffix}'")


def load_model(model, folder, device, suffix="", is_discriminator=False):
    """
    Charge G ou D selon les besoins.
    is_discriminator: True pour charger D, False pour charger G.
    """
    prefix = "D" if is_discriminator else "G"
    s = f"_{suffix}" if suffix else ""
    ckpt_path = os.path.join(folder, f'{prefix}{s}.pth')
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Le fichier {ckpt_path} est introuvable.")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Nettoyage des clés (pour la compatibilité DataParallel)
    clean_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(clean_ckpt)
    
    return model


# ------------------------------------- Utilities for metrics ------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- Feature Extractor Architecture (Improved LeNet-style) ---
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMNISTFeatureExtractor(nn.Module):
    """
    Deeper CNN to extract rich features,
    inspired by the Precision & Recall paper approach
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Convolutional block 1 (28x28 -> 14x14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Convolutional block 2 (14x14 -> 7x7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Convolutional block 3 (7x7 -> 3x3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully-connected layers
        # 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, feature_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(feature_dim, 10)  # Classification
        
        self.feature_dim = feature_dim
    
    def forward(self, x, return_features=False):
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # High-level features
        features = F.relu(self.fc1(x))
        # L2 normalization so features lie on the unit hypersphere (useful for distances)
        # improves distance stability and makes radii more comparable
        features = F.normalize(features, dim=1)
        
        if return_features:
            return features  # Vector of dimension feature_dim (e.g., 512)
        
        # Classification
        x = self.dropout(features)
        x = self.fc2(x)
        return x


# --- Helper function to load the model ---
def initialize_feature_extractor(device='cpu', path='checkpoints/cnn_mnist_features_extractor.pkl'):
    """Load the pre-trained classification model in evaluation mode."""
    model = ImprovedMNISTFeatureExtractor().to(device)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File {path} does not exist. Run 'python train_feature_extractor.py' first."
        )
        
    # Load weights
    state = torch.load(path, map_location=device)
    # Support both checkpoint dict and plain state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    
    model.eval()  # Important: evaluation mode (freezes dropout, batchnorm, etc.)
    return model


def get_features(model, images, batch_size=100, device='cpu', normalize=True):
    """Pass a batch of images through the model to obtain features."""
    features_list = []
    n_samples = images.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = images[i:i+batch_size].to(device)
            # Ensure images have shape (N, 1, 28, 28)
            if len(batch.shape) == 2:
                batch = batch.view(-1, 1, 28, 28)
            
            # Feature extraction
            feat = model(batch, return_features=True)

            if normalize:
                feat = F.normalize(feat, dim=1)

            features_list.append(feat.cpu())
            
    return torch.cat(features_list, dim=0)
