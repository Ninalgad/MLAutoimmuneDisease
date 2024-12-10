import torch
from torch import nn
from torch.nn import functional as F


def mlp(d_in, d_out, dff):
    return nn.Sequential(
        nn.Linear(d_in, dff),
        nn.ReLU(),
        nn.Linear(dff, dff),
        nn.ReLU(),
        nn.Linear(dff, d_out)
    )


def sparse_mlp(d_in, d_out, dff):
    return nn.Sequential(
        nn.Linear(d_in, dff),
        JumpReLU(),
        nn.Linear(dff, dff),
        JumpReLU(),
        nn.Linear(dff, d_out)
    )


class EncoderNet(nn.Module):
    def __init__(self, features_encoder, encoder_output_dim,
                 output_dim=460, hidden_dim=128):
        super().__init__()

        self.reg_head = mlp(encoder_output_dim, output_dim, hidden_dim)
        self.bin_head = mlp(encoder_output_dim, output_dim, hidden_dim)
        self.encoder = features_encoder

    def forward(self, x):
        x = self.encoder(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x_pooled = torch.squeeze(x, dim=(2, 3))

        r = torch.nn.Softplus()(self.reg_head(x_pooled))

        b = torch.nn.Sigmoid()(self.bin_head(x_pooled))

        return {'reg': r, 'bin': b}


class JumpReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.jump = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        heaviside_step = (x > self.jump).float()
        return x * heaviside_step


class SparseEncoderNet(nn.Module):
    def __init__(self, features_encoder, encoder_output_dim,
                 output_dim=460, hidden_dim=128):
        super().__init__()

        self.frame_head = sparse_mlp(encoder_output_dim, output_dim, hidden_dim)
        self.encoder = features_encoder

    def forward(self, x):
        x = self.encoder(x)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x, dim=(2, 3))

        x = self.frame_head(x)
        x = torch.nn.Softplus()(x)

        return {'reg': x}
