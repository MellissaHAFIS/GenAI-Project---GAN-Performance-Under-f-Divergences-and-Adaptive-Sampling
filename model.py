import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    """
        Discriminator for Vanila GAN
    """
    def __init__(self, d_input_dim, use_sigmoid=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc4(x)
        return torch.sigmoid(x) if self.use_sigmoid else x

class FGanDiscriminator(nn.Module):
    """
    Adapted discriminator for the FGAN 
    """

    def __init__(self, d_input_dim, use_sigmoid=True):
        super(FGanDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.fc1 = spectral_norm(nn.Linear(d_input_dim, 1024))
        self.fc2 = spectral_norm(nn.Linear(self.fc1.out_features, self.fc1.out_features//2))
        self.fc3 = spectral_norm(nn.Linear(self.fc2.out_features, self.fc2.out_features//2))
        self.fc4 = spectral_norm(nn.Linear(self.fc3.out_features, 1))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)
        return torch.sigmoid(x) if self.use_sigmoid else x
            