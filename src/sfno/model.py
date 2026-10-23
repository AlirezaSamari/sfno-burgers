import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes, self.width = modes, width
        scale = 1 / (width * width)
        self.weights = nn.Parameter(
            scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat)
        )
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        if self.modes > min(x.size(-2), x.size(-1) // 2 + 1):
            raise ValueError("The input grid is too small for the retained Fourier modes.")
        bypass = self.w(x)
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.shape[0],
            self.width,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes, : self.modes] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes, : self.modes], self.weights
        )
        x_spatial = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return F.gelu(x_spatial + bypass)


class StochasticFNO(nn.Module):
    def __init__(self, modes=16, width=64, num_layers=4, projection_width=128):
        super().__init__()
        self.lifting = nn.Linear(4, width)
        self.fourier_layers = nn.ModuleList(
            [SpectralConv2d(width, modes) for _ in range(num_layers)]
        )
        self.projection = nn.Sequential(
            nn.Linear(width, projection_width), nn.GELU(), nn.Linear(projection_width, 1)
        )

    def forward(self, x_grid, t_grid, u0, xi):
        batch, nx, nt = xi.shape
        xx, tt = torch.meshgrid(x_grid, t_grid, indexing="ij")
        features = torch.stack(
            [
                xx.unsqueeze(0).expand(batch, -1, -1),
                tt.unsqueeze(0).expand(batch, -1, -1),
                u0.unsqueeze(-1).expand(batch, nx, nt),
                xi,
            ],
            dim=-1,
        )
        hidden = self.lifting(features).permute(0, 3, 1, 2)
        for layer in self.fourier_layers:
            hidden = layer(hidden)
        return self.projection(hidden.permute(0, 2, 3, 1)).squeeze(-1)


def make_model(config, device):
    architecture = config.architecture
    return StochasticFNO(
        architecture.modes,
        architecture.width,
        architecture.num_layers,
        architecture.projection_width,
    ).to(device)


def count_parameters(model):
    parameters = list(model.parameters())
    return {
        "parameter_entries": sum(p.numel() for p in parameters),
        "real_scalar_components": sum(p.numel() * (2 if p.is_complex() else 1) for p in parameters),
        "trainable_parameter_entries": sum(p.numel() for p in parameters if p.requires_grad),
    }
