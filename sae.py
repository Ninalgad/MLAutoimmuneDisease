import torch
from torch import nn

from utils import *


class JumpReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.jump = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inp):
        h = heaviside_step(inp - self.jump)
        out = inp * h
        sparcity = h.sum()
        return out, sparcity


class SoftSAE(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.enc = nn.Linear(input_dim, width)
        self.dec = nn.Linear(width, input_dim)

        with torch.no_grad():
            nn.init.zeros_(self.enc.bias)
            nn.init.zeros_(self.dec.bias)

            self.enc.weight = nn.Parameter(self.dec.weight.t().clone())

            nn.utils.parametrizations.weight_norm(self.dec, 'weight')

    def forward(self, inp):
        feat = self.enc(inp)
        feat = nn.Softplus()(feat - 4.0)
        rec = self.dec(feat)

        return {'reconstructed': rec, 'features': feat}


class SAENet(nn.Module):
    def __init__(self, network, network_output_dim, n_features):
        super().__init__()
        self.sae = SoftSAE(network_output_dim, n_features)
        self.network = network

        # freeze network weights
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, img, mask):
        activations = self.network(img, mask)['embedding']
        sae_output = self.sae(activations)
        sae_output['activations'] = activations
        return sae_output


class PartiallySupervisedSAENet(nn.Module):
    def __init__(self, activation_output_dim, reg_output_dim, n_features):
        super().__init__()
        self.sae = SoftSAE(activation_output_dim, n_features)
        self.reg_dim = reg_output_dim

    def forward(self, img):
        sae_output = self.sae(img)

        features = sae_output['features']

        output = {
            'reconstructed': sae_output['reconstructed'],
            'reg': features[:, :self.reg_dim],
            'features': features[:, self.reg_dim:]
        }
        return output
