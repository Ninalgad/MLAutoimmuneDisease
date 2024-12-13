import torch
from torch import nn
from torch.nn import functional as F

from utils import *


def mlp(d_in, d_out, dff):
    return nn.Sequential(
        nn.Linear(d_in, dff),
        nn.ReLU(),
        nn.Linear(dff, dff),
        nn.ReLU(),
        nn.Linear(dff, d_out)
    )


class EncoderNet(nn.Module):
    def __init__(self, features_encoder, encoder_output_dim,
                 output_dim=460, hidden_dim=128):
        super().__init__()

        self.reg_head = mlp(encoder_output_dim, output_dim, hidden_dim)
        self.encoder = features_encoder

    def forward(self, x):
        x = self.encoder(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, dim=(2, 3))

        r = torch.nn.Softplus()(self.reg_head(x))

        return {'reg': r}


class GateLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, m):
        return x + self.alpha * m


class ConjoinedNet(nn.Module):
    def __init__(self, features_encoder, encoder_output_dim,
                 output_dim=460, hidden_dim=128):
        super().__init__()
        self.reg_head = mlp(encoder_output_dim, output_dim, hidden_dim)
        self.encoder = features_encoder
        self.n_enc_layers = len(features_encoder)

        self.gates = nn.ModuleList([GateLayer() for _ in range(self.n_enc_layers)])

    def forward(self, img, mask):
        x, m = img, inflate_mask(mask)
        for i in range(self.n_enc_layers):
            u, v = self.encoder[i](x), self.encoder[i](m)

            x = self.gates[i](u, v)
            m = v

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, dim=(2, 3))
        emb = x

        r = torch.nn.Softplus()(self.reg_head(x))

        return {'reg': r, 'embedding': emb}
